# 单字样本生成阶段

本阶段对应流程图左侧：

```text
prompt（字符 bound box 或中心点）
-> 字符区域 mask
-> GrabCut / OpenCV mask 精修
-> AlphaMatting 风格 soft alpha
-> 颜色归一化，消除原竹简背景残留色
-> 融合不同竹简背景
-> 边缘羽化 + 光照匹配
-> 新的字符样本
```

目前本地 smoke 使用 OpenCV 后端，不依赖 MobileSAM 权重。远程满血版可以后续把 MobileSAM 接入 `character_region_extractor.py`，但输出接口保持不变。

## 为什么之前看到的是整片图？

之前完成的是右侧“背景库生成”阶段，它必须读取整片图和 XML，把整片上的字符区域 inpaint 掉，再裁出背景 patch。

当前脚本 `generate_character_samples.py` 才是单字增强阶段：它读取 `clean_samples.csv` 中的单字图，抽取字符前景，再融合到背景库 patch 上。

## 前置输入

需要先有：

```text
label_index/
  clean_samples.csv
  rare_chars.csv

single_char_background_library/
  */patches/*.png
```

## 本地/远程 smoke

```powershell
python .\tools\smoke_test_character_sample_generation.py
```

## 真实数据小规模生成

```powershell
C:\yijian_project\envs\yijian\Scripts\python.exe .\tools\generate_character_samples.py `
  --data_root C:\yijian_project\data\raw `
  --clean_samples C:\yijian_project\data\label_index\clean_samples.csv `
  --rare_chars C:\yijian_project\data\label_index\rare_chars.csv `
  --background_root C:\yijian_project\data\background_library_smoke_edge `
  --out_dir C:\yijian_project\data\char_aug_smoke `
  --limit_chars 5 `
  --per_char 3 `
  --image_size 128
```

如果生成字符偏浅，可以加深墨色、减弱柔化：

```powershell
C:\yijian_project\envs\yijian\Scripts\python.exe .\tools\generate_character_samples.py `
  --data_root C:\yijian_project\data\raw `
  --clean_samples C:\yijian_project\data\label_index\clean_samples.csv `
  --rare_chars C:\yijian_project\data\label_index\rare_chars.csv `
  --background_root C:\yijian_project\data\background_library_smoke_edge `
  --out_dir C:\yijian_project\data\char_aug_smoke_dark `
  --limit_chars 5 `
  --per_char 3 `
  --image_size 128 `
  --mask_feather 1 `
  --ink_strength_min 190 `
  --ink_strength_max 260 `
  --alpha_power 0.45 `
  --darkness_gamma 0.6
```

## 四源背景轮转补齐到 20+

导师建议的“先对图像进行增强再提取字符”可以通过 `--pre_extract_enhance` 打开。当前脚本支持：

- `gamma`：Gamma 校正，默认 `--pre_extract_gamma 0.9`。
- `clahe`：限制对比度自适应直方图均衡化。
- `usm`：轻度 USM 锐化。
- `gamma_clahe`、`gamma_usm`、`guided_gamma_usm`、`median_gamma_usm`：组合增强。

增强阶段默认使用 `--background_source_policy cycle_all_sources`：每个待补字都会按 `tianhui -> zhangjiashan -> wuwei -> mawangdui` 的顺序轮转背景。也就是说，即使某个字只在天回出现，补样时也必须把四种医简背景全部用上，而不是只优先使用其他三种来源。`--target_count 20` 会按 `clean_samples.csv` 中已有数量计算缺口，只生成补齐部分。

```powershell
C:\yijian_project\envs\yijian\Scripts\python.exe .\tools\generate_character_samples.py `
  --data_root C:\yijian_project\data\raw `
  --clean_samples C:\yijian_project\data\label_index\clean_samples.csv `
  --rare_chars C:\yijian_project\data\label_index\rare_chars.csv `
  --background_root C:\yijian_project\data\single_char_background_library `
  --out_dir C:\yijian_project\data\char_aug_to20_preenhanced `
  --target_count 20 `
  --image_size 128 `
  --pre_extract_enhance gamma_usm `
  --pre_extract_gamma 0.9 `
  --mask_feather 1 `
  --ink_strength_min 190 `
  --ink_strength_max 260 `
  --alpha_power 0.45 `
  --darkness_gamma 0.6 `
  --source_bg_strength 0.45 `
  --white_alpha_suppression 0.85 `
  --strict_background_sources
```

如果只想随机抽背景做对照实验，可加 `--background_source_policy random`。

输出：

```text
char_aug_smoke/
  images/
  debug/
  generated_samples.csv
  summary.json
```

`generated_samples.csv` 使用 UTF-8 with BOM 写入，便于 Windows Excel/WPS 直接打开中文字段。表中同时保留 `char` 和 `char_code`：`char` 是源图片标注字符，`char_code` 是稳定的 Unicode 码点，例如 `丞 -> U4E1E`、`三十 -> U4E09_U5341`。生成图片目录优先使用真实字符名，例如 `images/丞/丞_0000.png`；只有遇到 Windows 文件名非法字符时才回退到 `char_code`。

`debug/` 里会保存部分 alpha mask 和 darkness 图，方便检查单字是否被正确提取。

## 墨色与边缘参数

- `--mask_feather`：字符 mask 羽化半径，默认 3。若字符太糊，降到 1。
- `--ink_strength_min/max`：融合时墨色压暗强度，默认 155-225。若字符太浅，提高到 190-260。
- `--alpha_power`：小于 1 会让 alpha 主体更实，默认 0.55。
- `--darkness_gamma`：小于 1 会加深淡墨区域，默认 0.72。
