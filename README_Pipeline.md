# MorphoDiff 完整流程笔记本

本目录包含一个完整的 Jupyter 笔记本 (`MorphoDiff_Complete_Pipeline.ipynb`)，它展示了 MorphoDiff 从数据预处理到训练再到推理的完整流程。

## 🚀 快速开始

### 1. 环境准备

首先确保您有合适的Python环境（推荐Python 3.10+）：

```bash
# 创建虚拟环境
python -m venv morphodiff_env
source morphodiff_env/bin/activate  # Linux/Mac
# 或 morphodiff_env\Scripts\activate  # Windows

# 安装基础依赖
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install datasets wandb jupyter notebook
pip install numpy pandas matplotlib seaborn Pillow tqdm
```

### 2. 配置 Accelerate

```bash
accelerate config
```

按照提示配置您的训练设置（GPU、混合精度等）。

### 3. 运行笔记本

```bash
jupyter notebook MorphoDiff_Complete_Pipeline.ipynb
```

## 📚 笔记本内容

笔记本包含以下主要部分：

### 1. 环境设置和依赖安装
- 检查 CUDA 可用性
- 自动安装缺失的包
- 导入所有必要的库

### 2. 数据预处理
- 创建示例数据集结构
- 准备 metadata.jsonl 文件
- 数据格式验证

### 3. 扰动编码（Perturbation Encoding）
- 设置扰动编码器
- 创建示例化学化合物编码
- 准备推理用的扰动列表

### 4. 模型训练
- 配置训练参数
- 生成训练命令
- 演示训练流程（可选择实际运行）

### 5. 图像生成和推理
- 配置图像生成参数
- 使用预训练模型演示图像生成
- 展示实际的 MorphoDiff 推理流程

### 6. 结果评估
- 图像质量统计分析
- 可视化生成的图像
- 基础评估指标

## 🔧 配置选项

在笔记本的第二个部分，您可以调整以下重要参数：

```python
# 数据路径
DATA_ROOT = "/tmp/morphodiff_data"  # 调整为您的数据目录
DATASET_NAME = "BBBC021"  # 选择数据集

# 训练参数
TRAINING_CONFIG = {
    "resolution": 512,
    "train_batch_size": 4,  # 根据GPU内存调整
    "max_train_steps": 500,  # 实际训练时增加
    "learning_rate": 1e-5,
    # ...
}
```

## 📊 示例数据

笔记本会自动创建示例数据用于演示。对于实际使用，您需要：

1. **下载真实数据集**：
   - BBBC021: https://bbbc.broadinstitute.org/BBBC021
   - RxRx1: https://www.rxrx.ai/rxrx1
   - Rohban et al.: https://github.com/broadinstitute/cellpainting-gallery

2. **准备扰动编码**：
   - 化学化合物：使用 RDKit 生成分子描述符
   - 基因扰动：使用 scGPT 生成基因嵌入

3. **组织数据结构**：
   ```
   train_imgs/
   ├── metadata.jsonl
   ├── image001.png
   ├── image002.png
   └── ...
   ```

## 🎯 运行模式

笔记本支持两种运行模式：

### 演示模式（默认）
- `dry_run=True`
- 显示命令但不实际执行
- 使用示例数据和预训练模型
- 适合学习和理解流程

### 实际运行模式
- `dry_run=False`
- 执行实际的训练和推理
- 需要真实数据和充足的计算资源
- 适合实际项目

## ⚠️ 注意事项

1. **计算资源**：实际训练需要大量 GPU 内存和计算时间
2. **数据大小**：真实数据集通常很大，确保有足够的存储空间
3. **依赖版本**：某些包可能需要特定版本，参考 `requirements.txt`
4. **CUDA 版本**：确保 PyTorch 和 CUDA 版本兼容

## 🔍 故障排除

### 常见问题：

1. **CUDA 不可用**：
   - 检查 GPU 驱动和 CUDA 安装
   - 安装对应的 PyTorch 版本

2. **内存不足**：
   - 减少 batch_size
   - 启用 gradient_checkpointing
   - 使用混合精度训练

3. **包导入错误**：
   - 确保所有依赖都已安装
   - 检查 Python 环境是否正确

## 📖 进一步学习

- [MorphoDiff 论文](https://openreview.net/pdf?id=PstM8YfhvI)
- [Diffusers 文档](https://huggingface.co/docs/diffusers)
- [Accelerate 文档](https://huggingface.co/docs/accelerate)

## 🤝 贡献

如果您发现问题或有改进建议，欢迎提交 Issue 或 Pull Request。

---

**祝您使用愉快！** 🧬🔬✨