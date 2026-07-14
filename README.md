# Ancient Chinese Medical Manuscript Dataset

## 项目简介

本项目整理并发布了四部出土医学文献的字符级与整片级数据集，用于：

* 古文字识别（OCR）
* 医简文字检测
* 单字符分类
* 版面分析
* 文本检测与识别
* 深度学习与计算机视觉研究

数据来源包括：

* 天回医简
* 马王堆汉墓帛书
* 张家山汉简
* 武威汉代医简

项目同时提供：

* MNIST 风格单字符数据集
* VOC 格式整片目标检测数据集

---

# 数据集结构

```text
dataset/
│
├── 天回医简/
│   ├── mnist/
│   └── voc/
│
├── 马王堆/
│   ├── mnist/
│   └── voc/
│
├── 张家山/
│   ├── mnist/
│   └── voc/
│
└── 武威/
    ├── mnist/
    └── voc/
```

---

# 数据集说明

## 1. MNIST 风格单字符数据集

用于字符分类任务。

### 特点

* 单张图片仅包含一个字符
* 已裁剪并标准化
* 适用于：

  * CNN
  * Vision Transformer
  * Swin Transformer
  * OCR 分类模型

### 目录示例

```text
mnist/
├── train/
├── val/
└── test/
```

## 2. VOC 格式整片数据集

用于目标检测与版面分析任务。

### 标注格式

采用 Pascal VOC 格式：

```text
label/
    *.xml

img/
    *.bmp

```

### XML 标注内容

包含：

* 字符类别
* Bounding Box
* 图像信息

适用于：

* YOLO
* Faster R-CNN
* SSD
* DETR
* MMDetection
* Detectron2

---

# 数据来源

## 天回医简

出土于四川成都天回镇汉墓，是目前已知最早的医学文献之一。

## 马王堆

长沙马王堆汉墓出土帛书医学文献。

## 张家山

湖北张家山汉简医学相关文献。

## 武威

甘肃武威汉代医简文献。

---

# 适用研究方向

本数据集适用于：

* 古文字 OCR
* 医简字符检测
* 汉简识别
* 小样本学习
* 目标检测
* 文本识别
* 文献数字化
* 数字人文研究

---

# 推荐环境

```text
Python >= 3.9
PyTorch >= 2.0
CUDA >= 11.8
```

推荐框架：

* PyTorch
* MMDetection
* PaddleOCR
* Detectron2
* Ultralytics YOLO

---

# 示例任务

## 字符分类

输入：

```text
单字符图片
```

输出：

```text
字符类别
```

---

## 字符检测

输入：

```text
整片医简图像
```

输出：

```text
字符位置 + 类别
```

---

# 注意事项

1. 数据仅用于学术研究与非商业用途
2. 请遵守相关文献与图像版权要求
3. 使用数据集时请注明来源

---

