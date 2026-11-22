# pairwise-sharpen-filter

## 领域

游戏、图形与多媒体

## 主题

锐化滤镜

## 说明

线性空间锐化、对比度自适应与 TAA 顺序修复。

初始状态下 `render/sharpen_filter.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
