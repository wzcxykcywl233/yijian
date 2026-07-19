# 困难字符二阶段增强实验

## 实验目标

对少样本字符先做真实数据训练和简单背景融合训练，再根据验证集逐类准确率找出困难字符。困难字符不沿用第一轮简单融合数据，而是分别采用不同的“图像预增强后再背景融合”方法重新生成数据，并比较重训指标。

实验流程：

```text
clean_samples.csv
-> 固定 train/val/test 划分
-> 真实数据 baseline 训练
-> 简单背景融合补齐到 target_count
-> 简单融合训练
-> 根据 val per-class accuracy 选择困难字符
-> 困难字符分别用 gamma / clahe / usm / 组合方法重新增强
-> 每种增强方法单独训练
-> 汇总指标，选择效果最好的增强方法
```

## 本地冒烟测试

本地只验证流程连通性，不代表正式识别效果。

```powershell
$env:PYTHONPATH=".\.codex_deps"
python .\tools\smoke_test_difficulty_augmentation_experiment.py
```

烟测使用 `nearest_centroid` 后端，不依赖 PyTorch。

## 远程正式运行

远程电脑建议先生成清洗索引和单字背景库：

```powershell
python .\tools\build_label_index.py `
  --data_root . `
  --out_dir .\data_exp\label_index `
  --min_count 20

python .\tools\build_single_char_background_library.py `
  --data_root . `
  --clean_samples .\data_exp\label_index\clean_samples.csv `
  --out_root .\data_exp\single_char_background_library `
  --patch_size 128
```

然后运行完整实验。若远程有 PyTorch，建议使用 `--backend torch_cnn`；否则可先用 `nearest_centroid` 做连通性检查。

```powershell
python .\tools\run_difficulty_augmentation_experiment.py `
  --data_root . `
  --clean_samples .\data_exp\label_index\clean_samples.csv `
  --background_root .\data_exp\single_char_background_library `
  --out_dir .\data_exp\difficulty_aug_experiment `
  --target_count 20 `
  --backend torch_cnn `
  --epochs 20 `
  --batch_size 128 `
  --pre_extract_methods gamma,clahe,usm,gamma_usm,guided_gamma_usm,median_gamma_usm `
  --difficulty_threshold 0.6 `
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
    difficult_gamma/
    ...
  experiment_summary.csv
  experiment_summary.json
```

`experiment_summary.csv` 是主结果表，包含每个阶段/方法的验证准确率、训练样本数、增强样本数和困难字符数量。每个 `models/*/per_class_metrics.csv` 可用于分析具体哪些字符提升或下降。

## 对照原则

- `splits/` 只生成一次，所有方法共享同一 train/val/test，避免划分差异影响结论。
- 第一轮 `simple_fusion` 用 `--pre_extract_enhance none`。
- 困难字符阶段只使用“真实训练集 + 当前方法重新生成的数据”，不混入第一轮简单融合数据。
- 背景默认四源轮转，正式实验建议加 `--strict_background_sources`，确保四种医简背景都参与。
- 选择最终方法时优先看同一验证集上的整体准确率、困难字符平均准确率和逐类提升情况；测试集建议只在最终候选方法确定后使用一次。
