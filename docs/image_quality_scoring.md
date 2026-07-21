# 图像质量评分路线

## 调整原因

当前阶段暂时搁置用 CNN 判断“是否需要图像增强后融合”的路线。原因是少样本字符本身类别多、单类样本少，CNN 的验证结果容易受到划分和少量样本波动影响。

新的主线改为：直接对图像本身打分，判断原始单字是否适合用于风格迁移，以及融合后样本是否正确有效。

## 评分对象

评分分为两类：

1. 原始单字图片
   - 目标：筛掉肉眼也难以辨别、过暗、过亮、过模糊、字符 mask 失败的源图。
   - 用途：避免这些图片继续作为字符来源参与风格迁移。

2. 数据增强图片
   - 目标：评价 simple_fusion 和图像预增强后融合的结果是否有效。
   - 用途：比较是否需要通过 gamma、CLAHE、USM、guided_gamma_usm 等图像增强来辅助风格迁移。

## 指标

来自 `图像增强.docx` 的核心指标：

- `PSNR`：反映增强图像相对原图的失真程度，越高表示越接近原图。
- `SSIM`：反映结构相似性，越高表示字符结构越稳定。
- `Entropy`：反映信息量，适当提高说明细节更丰富；过高可能表示噪声或背景也被增强。

额外补充的图像质量指标：

- `contrast_std`：灰度对比度，过低说明字符和背景不易区分。
- `laplacian_var`：清晰度/模糊度，过低说明笔画边缘模糊。
- `edge_density`：边缘密度，过低可能无有效笔画，过高可能背景纹理过强。
- `dark_ratio` / `white_ratio`：过暗或过白比例，用于筛极端背景。
- `alpha_ratio`：字符 mask 面积比例，过小或过大都不合理。
- `darkness_mean`：mask 内墨迹强度。
- `bbox_fill_ratio`：最大连通域填充率，用于识别“整块深色矩形被当作字符”的坏样本。

## 远程运行

### 只评分已有结果

对原始单字和已有增强结果统一评分：

```powershell
& $PY .\tools\score_image_quality.py `
  --data_root $DATA_ROOT `
  --clean_samples .\data_exp\label_index\clean_samples.csv `
  --experiment_dir .\data_exp\difficulty_aug_experiment `
  --out_dir .\data_exp\image_quality_scores `
  --filtered_clean_samples .\data_exp\image_quality_scores\clean_samples_quality_filtered.csv `
  --filtered_manifest_dir .\data_exp\image_quality_scores\filtered_manifests `
  --require_source_ok_for_manifest
```

输出：

```text
image_quality_scores/
  source_image_quality.csv
  source_quality_summary.csv
  generated_image_quality.csv
  generated_quality_summary.csv
  clean_samples_quality_filtered.csv
  filtered_manifests/
    simple_fusion_filtered_generated_samples.csv
    difficult_gamma_filtered_generated_samples.csv
    ...
  summary.json
```

### 不跑 CNN 的完整质量实验

先做小规模连通性测试：

```powershell
& $PY .\tools\run_quality_augmentation_experiment.py `
  --data_root $DATA_ROOT `
  --clean_samples .\data_exp\label_index\clean_samples.csv `
  --background_root .\data_exp\single_char_background_library `
  --out_dir .\data_exp\quality_aug_experiment_smoke `
  --rare_chars .\data_exp\label_index\rare_chars.csv `
  --target_count 20 `
  --limit_chars 20 `
  --pre_extract_methods gamma_usm,guided_gamma_usm `
  --filter_sources_first `
  --require_source_ok_for_manifest `
  --strict_background_sources
```

正式运行：

```powershell
& $PY .\tools\run_quality_augmentation_experiment.py `
  --data_root $DATA_ROOT `
  --clean_samples .\data_exp\label_index\clean_samples.csv `
  --background_root .\data_exp\single_char_background_library `
  --out_dir .\data_exp\quality_aug_experiment `
  --rare_chars .\data_exp\label_index\rare_chars.csv `
  --target_count 20 `
  --pre_extract_methods gamma,clahe,usm,gamma_usm,guided_gamma_usm,median_gamma_usm `
  --filter_sources_first `
  --require_source_ok_for_manifest `
  --strict_background_sources
```

这个流程不会训练 CNN，只会生成各方法的风格迁移样本并输出图像质量评分。

## 建议用法

第一步先看 `source_image_quality.csv`，确认哪些原图被判为低质量。重点查看：

- `quality_ok`
- `quality_score`
- `quality_reasons`
- `contrast_std`
- `laplacian_var`
- `bbox_fill_ratio`

第二步看 `generated_image_quality.csv`，比较不同增强方法的有效样本比例和平均质量分。

也可以直接看两个汇总表：

- `source_quality_summary.csv`：按医简来源汇总原图质量。
- `generated_quality_summary.csv`：按增强方法汇总通过率、平均质量分、平均 PSNR/SSIM/Entropy。

第三步再决定是否重跑增强：

- 若原图低质量比例较高，先用 `clean_samples_quality_filtered.csv` 作为新的输入，避免坏源图参与风格迁移。
- 若某些方法的融合坏样本比例低、SSIM/Entropy/字符 mask 指标更稳，则保留该方法。
- 若某些方法经常放大裂纹、帛纹或生成实心方块，则淘汰或调低增强强度。

## 高低分图像对比导出

为了在汇报中直观看到评分是否合理，可以导出每个来源/方法下的高分与低分样例：

```powershell
& $PY .\tools\export_quality_score_examples.py `
  --quality_dir .\data_exp\quality_aug_experiment\quality_scores `
  --out_dir .\data_exp\quality_score_examples `
  --kind both `
  --per_group 12 `
  --include_only_failed_low
```

输出结构：

```text
quality_score_examples/
  source/
    天回/
      high/
      low/
    马王堆/
      high/
      low/
  generated/
    gamma_usm/
      high/
      low/
    guided_gamma_usm/
      high/
      low/
  quality_score_examples_manifest.csv
```

文件名中会带上 `score`、`char` 和 `reason`。其中 `reason` 可用于解释低分原因，例如 `blurred`、`low_contrast`、`filled_rectangle_mask`、`weak_ink` 等。

## 当前建议

CNN 相关脚本暂时保留为备选，不作为当前主结论依据。当前例会和下一轮实验优先报告图像评分与可视化抽样结果。
