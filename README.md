# pairwise-tilt-shift-blur

## 领域

游戏、图形与多媒体

## 主题

移轴虚化

## 说明

焦点带、比例半径与高位深中间结果。

初始状态下 `imaging/tilt_shift_blur.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
