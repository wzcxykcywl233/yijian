# 数据集标签过滤规则

老师指出：标记为 `□` 的字符暂时需要剔除，因为这类字符无法确认到底属于哪一个具体字符类别。

因此后续分类训练、低频字统计、数据增强补样都采用统一规则：

```text
char == "□" 的样本不进入训练类别
char == "□" 的样本不参与少样本补齐
char == "□" 的样本单独记录到 excluded_unknown_samples.csv
```

注意：背景库生成阶段仍然可以使用整片 XML 中的所有 bndbox 做 inpainting。因为背景库阶段的目标是去除图像上的墨迹区域，不是在判断字符类别。

## 生成清洗后的标签索引

本地或远程都可以运行：

```powershell
python .\tools\build_label_index.py `
  --data_root . `
  --out_dir .\tmp\label_index `
  --min_count 20
```

远程标准路径：

```powershell
C:\yijian_project\envs\yijian\Scripts\python.exe .\tools\build_label_index.py `
  --data_root C:\yijian_project\data\raw `
  --out_dir C:\yijian_project\data\label_index `
  --min_count 20
```

## 输出

```text
label_index/
  clean_samples.csv              # 已排除未知方框字符后的样本索引
  excluded_unknown_samples.csv   # 被排除的 □ 样本
  class_counts.csv               # 清洗后的类别计数
  rare_chars.csv                 # 少于 min_count 的类别
  summary.json                   # 汇总信息
```

后续增强脚本应读取 `clean_samples.csv` 和 `rare_chars.csv`，不要直接使用原始 `label.tsv`。
