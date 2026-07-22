# 图像质量路线全量运行与导出

当前路线暂时不使用 CNN 训练结果作为困难字判断依据，而是先筛选原始单字图像质量，再生成简单背景融合和不同图像增强后的背景融合结果，最后对生成图像进行图像质量评分与可视化抽样。

生成阶段默认遵守“准入准出”原则：每张增强图的宽高与对应原始单字图片保持一致。`generated_samples.csv` 和 `generated_image_quality.csv` 中会记录 `source_width`、`source_height`、`output_width`、`output_height`，并在评分表中提供 `source_output_size_match` 方便检查。

## 1. 全量运行

远程电脑先确认变量：

```powershell
$PY = "C:\Users\Lenovo\miniconda3\envs\yijian311\python.exe"
$DATA_ROOT = "C:\yijian_project\data\raw"
cd C:\yijian_project\code\yijian
git pull
```

如果要重新跑一份完整结果：

```powershell
Remove-Item .\data_exp\quality_aug_experiment -Recurse -Force -ErrorAction SilentlyContinue

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

主要统计表：

```text
data_exp\quality_aug_experiment\source_quality_prefilter\source_image_quality.csv
data_exp\quality_aug_experiment\source_quality_prefilter\source_quality_summary.csv
data_exp\quality_aug_experiment\quality_scores\generated_image_quality.csv
data_exp\quality_aug_experiment\quality_scores\generated_quality_summary.csv
```

## 2. 高低分样例对比导出

用于报告中展示评分是否合理：

```powershell
& $PY .\tools\export_quality_score_examples.py `
  --quality_dir .\data_exp\quality_aug_experiment\quality_scores `
  --out_dir .\data_exp\quality_score_examples `
  --kind both `
  --per_group 12 `
  --include_only_failed_low
```

输出目录：

```text
data_exp\quality_score_examples
```

其中会按来源或增强方法分别导出 high / low 样例，并生成：

```text
data_exp\quality_score_examples\quality_score_examples_manifest.csv
```

## 3. 按原始单字图片溯源分组导出

用于查看“同一张原始单字图像”衍生出了哪些增强图。每个分组文件夹中会包含：

- `00_original__...png`：原始单字图片
- `*_generated__method-...__bg-...__score-...png`：由该原图衍生出的增强图片
- `source.txt`：原图路径、字符、来源等说明
- `lineage.csv`：该组内所有衍生图片的评分和路径

导出报告抽样版，默认看衍生图数量最多的 80 个原图分组：

```powershell
& $PY .\tools\export_source_lineage_examples.py `
  --quality_dir .\data_exp\quality_aug_experiment\quality_scores `
  --out_dir .\data_exp\source_lineage_examples `
  --source_limit 80 `
  --per_method 4 `
  --sort_by generated_count
```

如果想把所有原图分组和所有衍生图全部导出：

```powershell
& $PY .\tools\export_source_lineage_examples.py `
  --quality_dir .\data_exp\quality_aug_experiment\quality_scores `
  --out_dir .\data_exp\source_lineage_examples_all `
  --source_limit 0 `
  --per_method 0 `
  --sort_by generated_count
```

如果只想看某几个指定字符，例如“更”和“爱”：

```powershell
& $PY .\tools\export_source_lineage_examples.py `
  --quality_dir .\data_exp\quality_aug_experiment\quality_scores `
  --out_dir .\data_exp\source_lineage_examples_selected_chars `
  --char 更 `
  --char 爱 `
  --source_limit 0 `
  --per_method 0
```

总索引表：

```text
data_exp\source_lineage_examples\source_lineage_manifest.csv
```
