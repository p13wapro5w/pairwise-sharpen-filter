"""线性空间亮度锐化滤镜。

修复点（与旧的 sRGB 全通道固定半径锐化相对应）：

1. 线性空间：先 sRGB -> linear，在线性光空间计算 luma 与高频，最后再转回 sRGB，
   避免暗部噪声被伽马曲线放大成颗粒。
2. 按分辨率定半径：``radius_scale`` 让半径随输出/源分辨率比例缩放，4K 输出不再失效。
3. 对比度自适应：``contrast_adaptive`` / ``amount_curve`` / ``threshold_band`` /
   ``edge_limit`` 在高对比边缘衰减强度、在平坦区完全不响应，抑制 halo。
4. 去噪后锐化：``post_denoise`` 约束管线顺序，时域降噪收敛后才锐化，不固化噪声。
5. 只锐化亮度：``chroma_skip`` + ``luma_sharpen`` 只处理 Rec.709 luma，色度按比例保留，
   紫边/彩噪不被增强。
6. UI 分离：``ui_mask`` 标记 UI 层像素，世界缓冲锐化跳过它们，文字不出现描边。
7. 与 TAA 顺序：``taa_order`` 保证 TAA 历史缓冲先收敛，静止画面不再闪烁。
8. 输出分辨率前处理：``pre_resize`` 保证先缩放到输出分辨率、再在最终分辨率上锐化，
   细线不会被锐化之后的二次插值抹掉。
9. 高光钳制：``highlight_clamp`` 把结果限制在白点以内，灯丝不会变成死白。
10. 负值截断：``negative_clamp`` 截断过冲负值，暗部不出现黑色描边。

另外 ``alpha_preserve`` 保证透明像素不被锐化、alpha 通道原样透传；
``kernel_build`` 生成归一化高斯核供 luma 高频分离使用。

正确的管线阶段顺序见 ``PIPELINE_ORDER``：
denoise -> taa -> resize -> sharpen -> ui(composite)。
"""

from __future__ import annotations

import math

__all__ = [
    "luma_sharpen",
    "radius_scale",
    "contrast_adaptive",
    "post_denoise",
    "chroma_skip",
    "ui_mask",
    "taa_order",
    "pre_resize",
    "highlight_clamp",
    "negative_clamp",
    "kernel_build",
    "amount_curve",
    "threshold_band",
    "alpha_preserve",
    "edge_limit",
    "PIPELINE_ORDER",
    "LUMA_R",
    "LUMA_G",
    "LUMA_B",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: Rec.709 线性 luma 权重
LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722

#: 修正后的管线阶段顺序（索引越小越早执行）
PIPELINE_ORDER = ("denoise", "taa", "resize", "sharpen", "ui")

#: 半径标定参考分辨率高度（1080p）
REFERENCE_HEIGHT = 1080.0

#: 对比度自适应的软膝盖 sigma：local_contrast 达到该值时强度衰减到约 0.6
EDGE_SIGMA = 0.25

#: 单像素 luma 过冲的默认上限（占 [0, 1] 动态范围的比例），用于防 halo
DEFAULT_HALO_LIMIT = 0.25

#: 平坦区噪声门限默认值（线性 luma 空间）
DEFAULT_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# 色彩空间辅助（IEC 61966-2-1 sRGB EOTF/逆 EOTF）
# ---------------------------------------------------------------------------

def srgb_to_linear(channel: float) -> float:
    """单个 sRGB 通道值 -> 线性光值。"""
    c = max(0.0, min(1.0, float(channel)))
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(channel: float) -> float:
    """单个线性光通道值 -> sRGB 值，输出钳制在 [0, 1]。"""
    c = max(0.0, float(channel))
    if c <= 0.0031308:
        v = 12.92 * c
    else:
        v = 1.055 * c ** (1.0 / 2.4) - 0.055
    return max(0.0, min(1.0, v))


def _linear_luma(r: float, g: float, b: float) -> float:
    return LUMA_R * r + LUMA_G * g + LUMA_B * b


# ---------------------------------------------------------------------------
# 1. 线性空间 + 5. 只锐化亮度（核心入口）
# ---------------------------------------------------------------------------

def luma_sharpen(
    image,
    amount,
    radius=1.0,
    sigma=1.0,
    threshold=DEFAULT_THRESHOLD,
    white_point=1.0,
    halo_limit=DEFAULT_HALO_LIMIT,
    ui_mask_grid=None,
):
    """在线性空间只对 Rec.709 luma 做 unsharp mask。

    :param image: 单个像素 ``[r, g, b]`` / ``[r, g, b, a]``（sRGB, 0..1），
        或二维图像（行列表，每个像素同样为 3/4 通道）。
    :param amount: 基础锐化强度（线性空间）。
    :param radius: 高斯模糊半径（输出分辨率像素，浮点，内部四舍五入取整）。
        跨分辨率时应先用 :func:`radius_scale` 换算。
    :param sigma: 高斯核标准差。
    :param threshold: 平坦区门限（线性 luma 高频幅度），低于它的细节视为噪声不锐化。
    :param white_point: 高光白点钳制值（线性）。
    :param halo_limit: 单像素 luma 过冲/下冲的绝对上限，防高对比边缘 halo。
    :param ui_mask_grid: 与图像同形的 0/1 网格，1 表示该像素属于 UI 层，
        世界缓冲锐化整体跳过（UI 文字不产生描边）。
    :returns: 与输入同结构的锐化结果（sRGB, 0..1）；alpha 通道原样透传。
    """
    single_pixel = not (image and isinstance(image[0], (list, tuple)))
    pixels = [[list(image)]] if single_pixel else [list(row) for row in image]
    if ui_mask_grid is None:
        mask_rows = [[0 for _ in row] for row in pixels]
    elif single_pixel:
        flat = ui_mask_grid[0] if isinstance(ui_mask_grid, (list, tuple)) else ui_mask_grid
        mask_rows = [[int(flat)]]
    else:
        mask_rows = [[int(v) for v in row] for row in ui_mask_grid]

    height = len(pixels)
    width = max((len(row) for row in pixels), default=0)

    # 1) 全部转到线性空间，并计算线性 luma
    linear = [[[srgb_to_linear(c) for c in px[:3]] for px in row] for row in pixels]
    alphas = [[px[3] if len(px) > 3 else 1.0 for px in row] for row in pixels]
    luma = [[_linear_luma(*px) for px in row] for row in linear]

    # 2) luma 平面上做可分离高斯模糊，得到高频分量（色度完全不参与）
    kernel = kernel_build(radius, sigma)
    blurred = _separable_blur(luma, kernel)

    out_rows = []
    for y in range(height):
        out_row = []
        for x in range(width):
            alpha = alphas[y][x]
            if mask_rows[y][x]:
                # UI 层像素：跳过世界缓冲锐化，原样返回
                out_row.append(_pack_srgb(linear[y][x], pixels[y][x], alpha))
                continue

            y0 = luma[y][x]
            highpass = y0 - blurred[y][x]
            contrast = abs(highpass)

            # 3) 对比度自适应：平坦区门限 + 高对比边缘软衰减 + 中间调响应曲线
            effective_amount = amount_curve(float(amount), y0)
            effective_amount *= contrast_adaptive(1.0, contrast)
            effective_amount *= threshold_band(contrast, threshold)
            # 透明像素按 alpha 衰减，完全透明处不锐化
            effective_amount = alpha_preserve(alpha, effective_amount)

            # 4) 过冲限制（halo），再做高光/负值钳制
            delta = edge_limit(effective_amount * highpass, halo_limit)
            y_sharp = highlight_clamp(y0 + delta, white_point)
            y_sharp = negative_clamp(y_sharp, 0.0)

            # 5) 只改 luma：按比例缩放线性 RGB，色度（色相/饱和度）原样保留
            if y0 > 1e-8:
                ratio = y_sharp / y0
                rgb = [c * ratio for c in linear[y][x]]
            else:
                rgb = list(linear[y][x])
            rgb = [highlight_clamp(negative_clamp(c, 0.0), white_point) for c in rgb]
            out_row.append(_pack_srgb(rgb, pixels[y][x], alpha))
        out_rows.append(out_row)

    return out_rows[0][0] if single_pixel else out_rows


def _pack_srgb(linear_rgb, original_pixel, alpha):
    px = [linear_to_srgb(c) for c in linear_rgb[:3]]
    if len(original_pixel) > 3:
        px.append(float(alpha))
    return px


def _separable_blur(plane, kernel):
    """对二维标量平面做可分离高斯模糊，边界用 clamp-to-edge 处理。"""
    height = len(plane)
    width = max((len(row) for row in plane), default=0)
    radius = len(kernel) // 2

    def clamp(v, hi):
        return 0 if v < 0 else hi - 1 if v >= hi else v

    tmp = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for k, w in enumerate(kernel):
                acc += w * plane[y][clamp(x + k - radius, width)]
            tmp[y][x] = acc

    out = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for k, w in enumerate(kernel):
                acc += w * tmp[clamp(y + k - radius, height)][x]
            out[y][x] = acc
    return out


# ---------------------------------------------------------------------------
# 2. 按分辨率定半径
# ---------------------------------------------------------------------------

def radius_scale(radius_at_source, source_height, output_height):
    """把源分辨率下的锐化半径换算成输出分辨率下的像素半径。

    旧实现半径写死为固定像素数，内部分辨率渲染再放大到 4K 时，半径相对于
    输出画面缩水，锐化几乎不可见。修正方式是让半径跟随输出/源高度比例：

        r_out = r_src * H_out / H_src
    """
    source_height = float(source_height)
    output_height = float(output_height)
    if source_height <= 0.0:
        source_height = REFERENCE_HEIGHT
    return max(0.0, float(radius_at_source) * output_height / source_height)


# ---------------------------------------------------------------------------
# 3. 对比度自适应
# ---------------------------------------------------------------------------

def contrast_adaptive(strength, local_contrast):
    """根据局部对比度衰减锐化强度。

    局部对比度（线性 luma 高频幅度）越大，越可能是硬边缘；高斯软膝盖把强度
    压下去，避免边缘两侧堆出一圈 halo。对比度为 0 的平坦区返回满强度。
    """
    c = max(0.0, float(local_contrast))
    falloff = math.exp(-(c * c) / (2.0 * EDGE_SIGMA * EDGE_SIGMA))
    return float(strength) * falloff


def amount_curve(base_amount, luma):
    """中间调响应曲线：暗部与高光衰减、中间调满额。

    暗部衰减防止线性空间里残余噪声被推成颗粒，高光衰减给白点钳制减负。
    ``luma`` 为线性 luma（0..1），返回该亮度下实际使用的锐化强度。
    """
    l = max(0.0, min(1.0, float(luma)))
    mid_emphasis = 4.0 * l * (1.0 - l)  # 0/1 处为 0，0.5 处为 1
    return float(base_amount) * (0.5 + 0.5 * mid_emphasis)


def threshold_band(local_contrast, threshold):
    """噪声门限带：低于门限返回 0，门限到两倍门限之间 smoothstep 软过渡，之上为 1。

    硬门限会在门限值附近产生可见的开关痕迹，因此用一段平滑过渡带。
    """
    c = max(0.0, float(local_contrast))
    t = float(threshold)
    if t <= 0.0 or c >= 2.0 * t:
        return 1.0 if t > 0.0 or c > 0.0 else 0.0
    if c <= t:
        return 0.0
    x = (c - t) / t
    return x * x * (3.0 - 2.0 * x)


def edge_limit(delta, limit):
    """把单像素 luma 过冲量截断在 ``[-limit, limit]``，硬边缘不出现 halo 圈。"""
    lo = -abs(float(limit))
    hi = abs(float(limit))
    return max(lo, min(hi, float(delta)))


# ---------------------------------------------------------------------------
# 4. 去噪后锐化 / 7. TAA 顺序 / 8. 输出分辨率前处理
# ---------------------------------------------------------------------------

def post_denoise(denoise_stage, sharpen_stage):
    """锐化阶段必须晚于降噪阶段（时域降噪收敛后才允许锐化，否则噪声被固化）。"""
    return sharpen_stage > denoise_stage


def taa_order(taa_stage, sharpen_stage):
    """锐化阶段必须晚于 TAA 阶段（历史缓冲先收敛，避免静止画面与历史互相打架闪烁）。"""
    return sharpen_stage > taa_stage


def pre_resize(resize_stage, sharpen_stage):
    """必须先缩放到输出分辨率，再在最终分辨率上锐化。

    旧管线锐化之后才做输出缩放，锐化出的细线被缩放插值二次抹掉；正确顺序是
    resize 在前、sharpen 在后（见 ``PIPELINE_ORDER``）。
    """
    return sharpen_stage > resize_stage


# ---------------------------------------------------------------------------
# 5. 只锐化亮度
# ---------------------------------------------------------------------------

def chroma_skip(channel_id, luma_channel_id=0):
    """色度通道返回 1（跳过锐化），luma 通道返回 0（允许锐化）。"""
    return 0 if channel_id == luma_channel_id else 1


# ---------------------------------------------------------------------------
# 6. UI 分离
# ---------------------------------------------------------------------------

def ui_mask(layer_id, ui_layer_id=None):
    """层 id 等于 UI 层时返回 1：该像素只参与 UI 合成，跳过世界缓冲锐化。

    默认 UI 层 id 取 ``PIPELINE_ORDER`` 中 ``"ui"`` 的阶段索引。
    """
    if ui_layer_id is None:
        ui_layer_id = PIPELINE_ORDER.index("ui")
    return 1 if layer_id == ui_layer_id else 0


# ---------------------------------------------------------------------------
# 9. 高光钳制 / 10. 负值截断
# ---------------------------------------------------------------------------

def highlight_clamp(value, white_point):
    """高光钳制：线性值不允许超过白点，防止灯丝等亮部被推成死白。"""
    return min(float(value), float(white_point))


def negative_clamp(value, floor=0.0):
    """负值截断：过冲不允许低于 ``floor``（默认 0），暗部不出现黑色描边。"""
    return max(float(value), float(floor))


# ---------------------------------------------------------------------------
# 高斯核
# ---------------------------------------------------------------------------

def kernel_build(radius, sigma):
    """构建归一化一维高斯核（可分离使用）。

    :param radius: 半径像素数（浮点会四舍五入，最小为 0，核长 2*r+1）。
    :param sigma: 高斯标准差；非正值按极小值处理，退化为中心冲激。
    """
    r = int(round(max(0.0, float(radius))))
    s = float(sigma)
    if s <= 0.0:
        s = 1e-6
    taps = [
        math.exp(-(float(i) * float(i)) / (2.0 * s * s))
        for i in range(-r, r + 1)
    ]
    total = sum(taps)
    return [t / total for t in taps]


# ---------------------------------------------------------------------------
# alpha 保留
# ---------------------------------------------------------------------------

def alpha_preserve(alpha, amount):
    """锐化强度按 alpha 缩放：透明像素不锐化，半透明像素按覆盖率部分锐化。"""
    return float(amount) * max(0.0, min(1.0, float(alpha)))
