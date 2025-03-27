# CNN 手写数字识别项目

这个项目实现了基于卷积神经网络(CNN)的手写数字识别系统，支持MNIST和Fashion-MNIST数据集的训练和测试。本项目主要用于深入理解CNN的工作原理和实现细节，是深度学习入门的良好实践。

## 项目特点

- 详细的CNN模型实现，包括理论基础注释
- 支持简单CNN和深层CNN两种架构
- 完整的训练、验证和测试流程
- 丰富的可视化功能，包括模型预测、特征图和卷积核可视化
- 支持TensorBoard实时监控训练过程
- **支持识别用户自定义的手写数字图像**

## 项目结构

```
CNN_Digit_Recognition/
│
├── data/                   # 数据存储目录(自动创建)
├── models/                 # 模型定义
│   └── cnn_model.py        # CNN模型实现
├── utils/                  # 工具函数
│   ├── data_loader.py      # 数据加载器
│   ├── trainer.py          # 训练器实现
│   ├── visualization.py    # 可视化工具
│   └── custom_input.py     # 自定义输入处理
├── checkpoints/            # 模型保存目录(自动创建)
├── runs/                   # TensorBoard日志(自动创建)
├── main.py                 # 主程序
├── predict_custom.py       # 自定义手写数字识别脚本
├── requirements.txt        # 依赖项
└── README.md               # 项目说明
```

## CNN理论基础

卷积神经网络(CNN)是一种专门设计用于处理网格结构数据(如图像)的深度学习架构。CNN的主要组件包括：

1. **卷积层（Convolutional Layer）**：
   - 通过卷积核在输入上滑动进行特征提取
   - 特点：局部连接和权值共享，大大减少参数数量
   - 每个卷积核学习不同的特征模式

2. **激活函数（Activation Function）**：
   - 常用ReLU（Rectified Linear Unit）
   - 引入非线性特性，增强模型表达能力

3. **池化层（Pooling Layer）**：
   - 下采样操作，减小特征图尺寸
   - 提供一定程度的平移不变性
   - 减少计算量和参数数量

4. **全连接层（Fully Connected Layer）**：
   - 将特征映射转换为最终分类结果
   - 通常位于网络末端

## 安装依赖

首先安装项目所需的依赖项：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本训练

```bash
python main.py --dataset mnist --model simple --epochs 10
```

### 使用深层CNN模型

```bash
python main.py --dataset mnist --model deep --epochs 15
```

### 在Fashion-MNIST上训练

```bash
python main.py --dataset fashion-mnist --model deep --epochs 15
```

### 仅进行测试和可视化

```bash
python main.py --train False --test True --visualize True
```

## 自定义手写数字识别

训练完成后，您可以使用模型识别自己手写的数字或图像中的数字，支持三种输入方式：

### 1. 使用绘图工具手写数字（默认模式）

```bash
python predict_custom.py --model simple --dataset mnist
```

这将打开一个绘图窗口，您可以：
- 使用鼠标在窗口中绘制数字
- 按's'保存并进行预测
- 按'c'清空画布
- 按'q'退出

### 2. 使用摄像头拍摄手写数字

```bash
python predict_custom.py --model simple --mode camera
```

这将激活您的摄像头，倒计时后自动拍摄图像进行识别。

### 3. 使用现有图像文件

```bash
python predict_custom.py --model simple --mode file --input path/to/your/digit/image.jpg
```

## 命令行参数

### main.py 参数

- `--dataset`：选择数据集 (`mnist` 或 `fashion-mnist`)
- `--model`：选择模型架构 (`simple` 或 `deep`)
- `--batch-size`：训练批次大小
- `--epochs`：训练轮次
- `--lr`：学习率
- `--weight-decay`：权重衰减（L2正则化）系数
- `--val-split`：验证集比例
- `--seed`：随机种子
- `--save-dir`：模型保存目录
- `--no-cuda`：禁用CUDA（即使可用）
- `--num-workers`：数据加载的工作线程数
- `--train`：是否训练模型
- `--test`：是否测试模型
- `--visualize`：是否可视化模型预测和特征

### predict_custom.py 参数

- `--model`：选择模型架构 (`simple` 或 `deep`)
- `--dataset`：模型训练时使用的数据集 (`mnist` 或 `fashion-mnist`)
- `--save-dir`：模型保存的目录
- `--mode`：输入模式 (`file`, `draw`, `camera`)
- `--input`：图像文件路径（仅在`mode=file`时使用）
- `--no-cuda`：禁用CUDA（即使可用）

## 模型架构

### SimpleCNN

简单的CNN架构，包含：
- 2个卷积层
- 2个最大池化层
- 2个全连接层
- Dropout正则化

### DeepCNN

更深层次的CNN架构，包含：
- 4个卷积层
- 批归一化层
- 残差连接
- 更强的正则化

## 结果可视化

运行后，项目会生成以下可视化内容：

1. 训练曲线：损失和准确率随训练轮次的变化
2. 混淆矩阵：显示分类结果的详细统计
3. 模型预测可视化：随机测试样本的预测结果
4. 特征图可视化：展示CNN模型提取的特征
5. 卷积核可视化：展示CNN模型学习的滤波器
6. 自定义输入预测结果：展示原始输入和处理后的图像以及预测结果 