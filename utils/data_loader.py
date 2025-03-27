import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_loaders(batch_size=64, val_split=0.1, root='./data', num_workers=2):
    """
    创建MNIST数据集的训练、验证和测试数据加载器
    
    参数:
        batch_size (int): 每批数据的大小
        val_split (float): 验证集比例 (0.0-1.0)
        root (str): 数据存储的根目录
        num_workers (int): 数据加载的工作线程数
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 数据预处理和增强
    # MNIST为灰度图像，范围为[0,1]，通过transforms.Normalize进行标准化处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor，并缩放到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    # 下载并加载训练数据
    full_train_dataset = datasets.MNIST(
        root=root, 
        train=True,  # 使用训练集
        download=True,  # 如果数据不存在则下载
        transform=transform  # 应用上面定义的变换
    )
    
    # 计算训练集和验证集大小
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    # 随机拆分训练集和验证集
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
    )
    
    # 下载并加载测试数据
    test_dataset = datasets.MNIST(
        root=root, 
        train=False,  # 使用测试集
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,  # 打乱训练数据
        num_workers=num_workers,
        pin_memory=True  # 加速数据传输到GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_fashion_mnist_loaders(batch_size=64, val_split=0.1, root='./data', num_workers=2):
    """
    创建Fashion-MNIST数据集的训练、验证和测试数据加载器
    Fashion-MNIST是比MNIST稍微复杂一些的数据集，包含10类服装图像
    
    参数:
        batch_size (int): 每批数据的大小
        val_split (float): 验证集比例 (0.0-1.0)
        root (str): 数据存储的根目录
        num_workers (int): 数据加载的工作线程数
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 数据预处理和增强，添加一些数据增强以提高模型泛化能力
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),      # 随机旋转±10度
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST的均值和标准差
    ])
    
    # 测试时不需要数据增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 下载并加载训练数据
    full_train_dataset = datasets.FashionMNIST(
        root=root, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # 计算训练集和验证集大小
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    # 随机拆分训练集和验证集
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 为验证集重新应用transform（不包含数据增强）
    val_dataset.dataset.transform = test_transform
    
    # 下载并加载测试数据
    test_dataset = datasets.FashionMNIST(
        root=root, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 