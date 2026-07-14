# 字符区域提取阶段

本阶段对应流程图中的：

```text
prompt（字符 bound box 或中心点） -> MobileSAM（获取字符区域）
```

当前代码提供两个后端：

- `opencv`：本地冒烟测试后端，不依赖模型权重。
- `mobilesam`：远程满血验证预留后端，保持同一 CLI 接口。

## 输入

输入一张单字图或整片中裁剪出的字符候选区域，并提供一种 prompt：

- `--box xmin,ymin,xmax,ymax`
- `--point x,y`

推荐优先使用 `box`。如果只有中心点，脚本会根据图像尺寸生成一个局部提示框。

## 输出

- 灰度 mask：字符区域，边缘经过 feather，避免硬边贴图。
- RGBA cutout：可选，alpha 通道即字符 mask。
- JSON 元数据：记录输入、后端、prompt、mask 面积等。

## 本地冒烟测试

```powershell
python .\tools\smoke_test_character_region.py
```

冒烟测试会创建一张合成竹简背景图，加入一个字符形状和一个裂缝干扰项，然后检查 mask 和透明 cutout 是否能生成。

## 真实数据示例

```powershell
python .\tools\character_region_extractor.py `
  --image ".\天回\天回-单字\天回\一\123.bmp" `
  --box "10,10,100,100" `
  --out_mask ".\tmp\mask.png" `
  --out_cutout ".\tmp\cutout.png" `
  --backend opencv
```

## 当前阶段的风险控制

- 墨水晕染导致扣字不全：使用 Otsu 阈值、形态学闭运算和软 mask。
- 裂缝被误扣：通过连通域面积、长宽比和 prompt 区域约束过滤。
- 边缘过硬：对 mask 做 Gaussian feather，后续融合时可继续做颜色匹配。
- 复杂字被扣断：保留 GrabCut 和闭运算修正，但参数保持保守，避免改变字形结构。

远程环境接入 MobileSAM 时，建议仅替换 `extract_with_mobilesam()`，不要改变命令行参数和输出格式，这样后续增强流程可以复用。
