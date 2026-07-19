# 背景库生成阶段

本阶段对应流程图中的：

```text
背景图片生成
-> Inpainting（去除字符）
-> 背景纹理
-> 背景增强（亮度、色调、纹理扰动、噪声）
-> 背景库（四个来源）
```

## 目标

从整片图像和 XML 字符框中生成无字背景，并裁剪背景 patch。后续字符前景融合或风格迁移时，优先使用这些真实来源背景，而不是直接拼接纯色背景。

数据集整片图像外侧常带有白色衬底。脚本默认使用 `edge` 模式：通过 Canny 边缘检测计算行/列投影，找到真实医简或文字内容区域的外接框，再用裁剪后的真实区域做 mask、inpainting 和 patch 裁剪，避免后续背景库学到白色画布。

如果需要严格按纯白边框裁剪，也可以使用 `--crop_method pure_white`：从上下左右四个方向向内扫描，若整行或整列所有像素均为纯白色，则裁掉；一旦遇到任意非纯白像素就停止。

## 单字背景库

当前少样本补齐实验优先使用单字图提取背景，而不是整片图背景。原因是单字图的裁剪尺度、局部底纹和分类训练样本更一致，也能避免整片图中的大块空白、版面边缘和 inpainting 伪影进入背景库。

```powershell
python .\tools\build_single_char_background_library.py `
  --data_root . `
  --clean_samples .\tmp\label_index\clean_samples.csv `
  --out_root .\tmp\single_char_background_library `
  --patch_size 128
```

输出结构与整片背景库保持一致：

```text
single_char_background_library/
  manifest.csv
  summary.json
  tianhui/
    inpainted/
    masks/
    patches/
  zhangjiashan/
  wuwei/
  mawangdui/
```

正式补样时应检查四个来源下均有可用 `patches/*.png`，并在生成阶段使用 `--strict_background_sources`。

## 默认四个来源

脚本默认读取：

```text
天回/天回-整片爬取
张家山/张家山-整片
武威/武威_整片
马王堆/马王堆-整片爬取
```

每个来源目录下需要有：

```text
img/
label/
```

其中 `label/*.xml` 采用 VOC 风格，只要 `<filename>` 和 `<bndbox>` 坐标可读即可。由于数据集中部分 `<name>` 标签损坏，脚本采用宽松解析，不依赖完整 XML 树。

## 本地冒烟测试

```powershell
python .\tools\smoke_test_background_library.py
```

冒烟测试会创建一张合成整片图和一个 XML 框，验证是否能生成：

- `inpainted/*.png`
- `masks/*.png`
- `patches/*.png`
- `manifest.csv`
- `summary.json`

## 真实数据小规模测试

建议远程电脑先跑每个来源 2 张：

```powershell
C:\yijian_project\envs\yijian\Scripts\python.exe .\tools\build_background_library.py `
  --data_root C:\yijian_project\data\raw `
  --out_root C:\yijian_project\data\background_library_smoke `
  --limit 2 `
  --patches_per_image 4 `
  --patch_size 128
```

## 满血生成

```powershell
C:\yijian_project\envs\yijian\Scripts\python.exe .\tools\build_background_library.py `
  --data_root C:\yijian_project\data\raw `
  --out_root C:\yijian_project\data\background_library `
  --patches_per_image 8 `
  --patch_size 128
```

## 输出结构

```text
background_library/
  manifest.csv
  summary.json
  tianhui/
    inpainted/
    masks/
    patches/
  zhangjiashan/
    inpainted/
    masks/
    patches/
  wuwei/
  mawangdui/
```

## 参数说明

- `--mask_pad`：在 XML 框外扩若干像素，减少残留笔画。
- `--dilate`：对字符 mask 膨胀，进一步覆盖墨迹边缘。
- `--inpaint_radius`：OpenCV inpaint 半径。
- `--patches_per_image`：每张无字背景裁剪多少个 patch。
- `--patch_size`：背景 patch 尺寸。
- `--no_enhance`：关闭亮度、色调、噪声、纹理扰动。
- `--crop_method`：裁剪方法，默认 `edge`，可选 `edge`、`pure_white`、`none`。
- `--no_crop_white_border`：旧参数，等价于 `--crop_method none`。
- `--crop_margin`：裁剪真实图像区域时额外保留的边距，默认 32。edge 模式会先找到边缘内容区域，再向外保留这段真实背景，避免只贴着字符裁剪。
- `--canny_low`、`--canny_high`：edge 模式下的 Canny 阈值。
- `--edge_dilate`：edge 模式下边缘膨胀半径。
- `--max_patch_mask_ratio`：过滤与字符 mask 重叠过多的 patch，默认 0.02，减少 inpaint 伪影进入背景库。
- `--max_patch_white_ratio`：过滤白色扫描画布占比过高的 patch，默认 0.35。
- `--min_patch_std`：过滤过于平坦、缺少纹理的 patch，默认 4.0。
- `--patch_attempts`：每个目标 patch 的随机尝试次数，默认 40。过滤较严格时，实际输出 patch 数可能少于 `--patches_per_image`。

## 风险控制

- XML 损坏：只解析 `<bndbox>`，不依赖 `<name>`。
- 字符残留：通过 `mask_pad`、`dilate` 和 inpaint 半径控制。
- 白色衬底污染背景库：默认用边缘检测裁掉外侧画布，并同步平移 XML 框坐标。
- 无效 patch 污染背景库：采样时避开字符 mask，并过滤白底比例过高或纹理过弱的 patch。
- 背景过度增强：增强只作用于背景 patch，不作用于字符前景。
- 贴图感：后续融合阶段仍需做 alpha 羽化与颜色匹配。
