# d2-detection

基于 [Dinomaly](https://github.com/guojiajeremy/Dinomaly) 的异常检测项目，支持 DINOv2 和 DINOv3 作为视觉编码器。

## 1. 环境配置

```bash
conda create -n env_dinomaly python=3.8 -y
conda activate env_dinomaly
pip install -r requirements.txt
pip install safetensors   # DINOv3 需要
```

**依赖说明：** PyTorch 1.12 + CUDA 11.3，如需其他 CUDA 版本请参考 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 安装对应版本。

## 2. 数据集准备

### MVTec AD

下载 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 数据集，解压到项目同级目录：

```
Project/
├── d2-detection/       # 本仓库
└── mvtec_anomaly_detection/
    ├── bottle/
    ├── cable/
    ├── ...
    └── zipper/
```

如果数据集放在其他位置，运行时通过 `--data_path` 指定路径。

### VisA

下载 [VisA](https://github.com/amazon-science/spot-diff) 数据集后，运行预处理脚本：

```bash
python prepare_data/prepare_visa.py --data_path <visa数据集路径>
```

## 3. 编码器权重

### DINOv2（默认）

首次运行时会自动从 Facebook 服务器下载权重到 `backbones/weights/`，无需手动操作。

### DINOv3

从 ModelScope 下载模型到项目根目录的 `Dinov3/` 文件夹，目录结构：

```
d2-detection/
└── Dinov3/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

如需使用不同规格（ViT-B/L），下载到不同目录后通过 `encoder_name` 指定路径：

```python
encoder_name = 'dinov3_vit_small_16'              # 默认读取 ./Dinov3
encoder_name = 'dinov3_vit_base_16:./Dinov3B'     # ViT-B
encoder_name = 'dinov3_vit_large_16:./Dinov3L'    # ViT-L
```

## 4. 运行训练

### 统一训练模式（uni）

所有类别共享一个模型：

```bash
conda activate env_dinomaly

# MVTec AD
python dinomaly_mvtec_uni.py --data_path ../mvtec_anomaly_detection

# VisA
python dinomaly_visa_uni.py --data_path ../VisA

# Real-IAD
python dinomaly_realiad_uni.py --data_path ../realiad
```

### 单类别训练模式（sep）

每个类别单独训练一个模型：

```bash
# MVTec AD
python dinomaly_mvtec_sep.py --data_path ../mvtec_anomaly_detection

# VisA
python dinomaly_visa_sep.py --data_path ../VisA

# MPDD
python dinomaly_mpdd_sep.py --data_path ../MPDD
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_path` | 数据集路径 | `../mvtec_anomaly_detection` |
| `--save_dir` | 结果保存目录 | `./saved_results` |
| `--save_name` | 实验名称 | 自动生成 |

### 切换编码器

在训练脚本的 `train()` 函数中修改 `encoder_name`：

```python
# DINOv2 系列
encoder_name = 'dinov2reg_vit_small_14'
encoder_name = 'dinov2reg_vit_base_14'     # 默认
encoder_name = 'dinov2reg_vit_large_14'

# DINOv3 系列
encoder_name = 'dinov3_vit_small_16'
encoder_name = 'dinov3_vit_base_16:./Dinov3B'
encoder_name = 'dinov3_vit_large_16:./Dinov3L'
```

`crop_size`、`embed_dim`、`num_heads`、`target_layers` 等参数会根据编码器自动适配，无需手动调整。

## 5. 输出说明

训练日志和结果保存在 `saved_results/<save_name>/log.txt`，包含：

- 训练 loss
- 每个类别的评估指标：
  - **I-Auroc / I-AP / I-F1** — 图像级异常检测
  - **P-AUROC / P-AP / P-F1 / P-AUPRO** — 像素级异常定位

## 6. 项目结构

```
d2-detection/
├── dinomaly_mvtec_uni.py    # MVTec 统一训练
├── dinomaly_mvtec_sep.py    # MVTec 单类别训练
├── dinomaly_visa_*.py       # VisA 数据集
├── dinomaly_realiad_*.py    # Real-IAD 数据集
├── dinomaly_mpdd_sep.py     # MPDD 数据集
├── models/
│   ├── uad.py               # ViTill 模型架构
│   ├── vit_encoder.py        # 编码器加载
│   └── vision_transformer.py # 解码器模块
├── dinov1/                   # DINOv1 实现
├── dinov2/                   # DINOv2 实现
├── dinov3/                   # DINOv3 实现（2D RoPE）
├── dataset.py                # 数据集加载
├── utils.py                  # 评估工具
└── optimizers/               # StableAdamW 优化器
```
