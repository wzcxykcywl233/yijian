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

background_library_smoke_edge/
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

输出：

```text
char_aug_smoke/
  images/
  debug/
  generated_samples.csv
  summary.json
```

`debug/` 里会保存部分 alpha mask 和 darkness 图，方便检查单字是否被正确提取。
