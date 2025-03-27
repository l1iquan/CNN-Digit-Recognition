import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络用于MNIST手写数字识别
    
    CNN的理论基础:
    1. 局部连接：每个神经元只与输入数据的一小部分连接，这部分称为感受野
    2. 权值共享：同一个特征图中的神经元共享权重，减少参数数量
    3. 池化操作：通过池化层实现参数下采样，提高模型鲁棒性
    
    网络结构:
    - 两个卷积层，每个后面跟着ReLU激活函数和MaxPooling
    - 两个全连接层用于分类
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积层：输入通道=1(灰度图)，输出通道=32，卷积核大小=3x3
        # 卷积层工作原理：使用可学习的卷积核在输入上滑动，提取局部特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # 第二个卷积层：输入通道=32，输出通道=64，卷积核大小=3x3
        # 通过增加通道数，网络可以学习更复杂的特征表示
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 池化层：使用2x2窗口进行最大池化，步长为2
        # 池化作用：1.降低特征图分辨率 2.实现平移不变性 3.降低过拟合风险
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层：线性变换将特征映射到类别空间
        # 7*7是经过两次池化后的特征图大小(28/2/2=7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个输出对应10个数字类别
        
        # Dropout层：随机丢弃神经元，防止过拟合
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # x形状: [batch_size, 1, 28, 28]
        
        # 第一个卷积块：卷积 -> ReLU激活 -> 最大池化
        # ReLU激活函数对负值截断为0，引入非线性，同时计算高效
        x = self.pool(F.relu(self.conv1(x)))  # 输出形状: [batch_size, 32, 14, 14]
        
        # 第二个卷积块
        x = self.pool(F.relu(self.conv2(x)))  # 输出形状: [batch_size, 64, 7, 7]
        
        # 展平操作，将三维特征图转为一维特征向量，以便输入全连接层
        x = x.view(-1, 64 * 7 * 7)  # 输出形状: [batch_size, 64*7*7]
        
        # 全连接层 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 最终分类层
        x = self.fc2(x)  # 输出形状: [batch_size, 10]
        
        # 使用log_softmax比直接使用softmax更稳定，防止数值溢出
        # 对数概率输出，与NLLLoss配合使用
        return F.log_softmax(x, dim=1)


class DeepCNN(nn.Module):
    """
    更深层次的CNN模型，使用更多层次提取复杂特征
    
    增加的特性:
    1. 批归一化：加速收敛并提高稳定性
    2. 残差连接：缓解深度网络的梯度消失问题
    3. 更多的卷积层：提取更丰富的特征表示
    """
    def __init__(self):
        super(DeepCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化：标准化每个批次的激活，加速训练
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 第一个卷积块
        identity1 = x  # 保存输入用于残差连接
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # 由于通道数不同，我们不能直接相加，这里忽略残差连接
        x = F.relu(x)
        x = self.pool(x)
        
        # 第二个卷积块（包含残差连接）
        identity2 = x  # 保存特征用于残差连接
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = x + identity2  # 残差连接：将输入直接加到输出上，缓解梯度消失问题
        x = F.relu(x)
        x = self.pool(x)
        
        # 全连接层
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1) 