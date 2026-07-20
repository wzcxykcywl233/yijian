# 困难字二阶段增强实验

## 实验目标

本实验用于判断：对少样本字符而言，哪些图像预增强方法能比“直接做简单背景融合”更有效。

困难字的定义不是单纯“识别准确率低”，而是：

```text
少样本字符 + 简单背景融合相对真实数据 baseline 没有明显正向作用，或造成退化
```

这些字的常见表现可能包括：字形模糊导致字符分割困难、背景深色裂纹被误分为笔画、源单字像素低导致复杂笔画不可分等。脚本用逐字验证集指标来近似这种现象。

## 流程

```text
clean_samples.csv
-> 固定 train/val/test 划分
-> 仅真实数据训练 baseline_real
-> 对少样本字做 simple_fusion，补齐到 target_count
-> 用真实训练集 + simple_fusion 训练 simple_fusion 模型
-> 比较 baseline_real 与 simple_fusion 的逐字验证指标
-> 选出“简单融合无明显增益/退化”的困难字
-> 对困难字分别使用 gamma / clahe / usm / 组合方法进行预增强后再背景融合
-> 每种困难字重增强方法单独训练
-> 汇总指标，比较哪种图像增强方法最有效
```

困难字筛选默认条件：

- 真实训练集中该字数量 `< target_count`
- baseline 和 simple_fusion 的验证集中都出现过该字
- simple_fusion 相对 baseline 的准确率增益 `<= difficulty_min_improvement`
- 且满足以下任一情况：
  - simple_fusion 准确率 `<= difficulty_threshold`
  - simple_fusion 相对 baseline 变差

默认参数：

```text
target_count = 20
difficulty_threshold = 0.6
difficulty_min_improvement = 0.02
```

如果设置 `--difficulty_top_k`，会优先选择准确率增益最低、simple_fusion 准确率最低、验证支持数较多的字符。

## 本地冒烟测试

本地只验证流程连通性，不代表正式识别效果。

```powershell
$env:PYTHONPATH=".\.codex_deps"
python .\tools\smoke_test_difficulty_augmentation_experiment.py
```

冒烟测试默认使用 `nearest_centroid` 后端，不依赖 PyTorch。

## 远程正式运行

远程电脑建议先生成清洗索引和单字背景库：

```powershell
python .\tools\build_label_index.py `
  --data_root C:\yijian_project\data\raw `
  --out_dir .\data_exp\label_index `
  --min_count 20

python .\tools\build_single_char_background_library.py `
  --data_root C:\yijian_project\data\raw `
  --clean_samples .\data_exp\label_index\clean_samples.csv `
  --out_root .\data_exp\single_char_background_library `
  --patch_size 128
```

然后运行完整实验：

```powershell
& $PY .\tools\run_difficulty_augmentation_experiment.py `
  --data_root $DATA_ROOT `
  --clean_samples .\data_exp\label_index\clean_samples.csv `
  --background_root .\data_exp\single_char_background_library `
  --out_dir .\data_exp\difficulty_aug_experiment `
  --target_count 20 `
  --backend torch_cnn `
  --epochs 20 `
  --batch_size 128 `
  --pre_extract_methods gamma,clahe,usm,gamma_usm,guided_gamma_usm,median_gamma_usm `
  --difficulty_threshold 0.6 `
  --difficulty_min_improvement 0.02 `
  --difficulty_top_k 100 `
  --strict_background_sources
```

## 输出

```text
difficulty_aug_experiment/
  splits/
    train.csv
    val.csv
    test.csv
    train_rare_chars.csv
  augment/
    simple_fusion/
    difficult_gamma/
    difficult_clahe/
    ...
  models/
    baseline_real/
      metrics.json
      per_class_metrics.csv
      difficult_chars.csv
    simple_fusion/
      metrics.json
      per_class_metrics.csv
    difficult_gamma/
    ...
  difficult_chars_by_simple_gain.csv
  experiment_summary.csv
  experiment_summary.json
```

`difficult_chars_by_simple_gain.csv` 是二阶段重增强真正使用的困难字清单，包含：

- `baseline_accuracy`
- `simple_accuracy`
- `accuracy_delta`
- `support`
- `reason`

`experiment_summary.csv` 是主结果表，包含每个阶段/方法的验证准确率、训练样本数、增强样本数和困难字数量。

## 对照原则

- `splits/` 只生成一次，所有方法共享同一 train/val/test，避免划分差异影响结论。
- 第一轮 `simple_fusion` 使用 `--pre_extract_enhance none`。
- 困难字阶段只使用“真实训练集 + 当前方法重新生成的数据”，不混入第一轮 simple_fusion 数据。
- 背景默认四源轮转，正式实验建议加 `--strict_background_sources`，确保四种医简背景都参与。
- 选择最终方法时，优先看同一验证集上的整体准确率、困难字平均准确率和逐类提升情况；测试集建议只在最终候选方法确定后使用一次。
