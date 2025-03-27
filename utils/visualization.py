import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import os

def show_batch(batch, predictions=None, true_labels=None, class_names=None, num_samples=16):
    """
    显示一批图像及其预测结果
    
    参数:
        batch: 图像批次，形状为 [batch_size, channels, height, width]
        predictions: 预测标签
        true_labels: 真实标签
        class_names: 类别名称列表
        num_samples: 要显示的样本数量
    """
    # 确保要显示的样本数量不超过批次大小
    batch_size = batch.size(0)
    num_samples = min(num_samples, batch_size)
    
    # 如果预测或真实标签不为None，确保长度匹配
    if predictions is not None:
        assert len(predictions) >= num_samples, "预测数量应至少等于要显示的样本数量"
    if true_labels is not None:
        assert len(true_labels) >= num_samples, "真实标签数量应至少等于要显示的样本数量"
    
    # 如果未提供类别名称，使用数字
    if class_names is None:
        max_label = 9  # 假设MNIST数据集
        class_names = [str(i) for i in range(max_label + 1)]
    
    # 计算网格尺寸
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # 创建图形
    fig = plt.figure(figsize=(15, 15))
    
    for i in range(num_samples):
        # 创建子图
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        
        # 获取图像并转换为[height, width, channels]格式
        img = batch[i].cpu().numpy().transpose((1, 2, 0))
        
        # 如果是单通道图像，去掉通道维度
        if img.shape[2] == 1:
            img = img.squeeze(2)
        
        # 反归一化处理(如果需要)
        mean = np.array([0.1307])
        std = np.array([0.3081])
        img = img * std + mean
        
        # 显示图像
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        
        # 设置标题
        title = ""
        if true_labels is not None and predictions is not None:
            pred_class = predictions[i]
            true_class = true_labels[i]
            correct = pred_class == true_class
            title = f"预测: {class_names[pred_class]}\n真实: {class_names[true_class]}"
            title_color = "green" if correct else "red"
            ax.set_title(title, color=title_color)
        elif true_labels is not None:
            true_class = true_labels[i]
            title = f"标签: {class_names[true_class]}"
            ax.set_title(title)
        
        # 隐藏坐标轴
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_model_predictions(model, data_loader, device, num_samples=16, class_names=None, save_path=None):
    """
    可视化模型对一批数据的预测结果
    
    参数:
        model: PyTorch模型
        data_loader: 数据加载器
        device: 计算设备(CPU/GPU)
        num_samples: 要可视化的样本数量
        class_names: 类别名称列表
        save_path: 保存图像的路径，如果为None则显示图像而不保存
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 获取一批数据
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # 限制样本数量
    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 进行预测
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predictions = outputs.max(1)
    
    # 将标签和预测结果转移到CPU
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # 显示结果
    fig = show_batch(images, predictions, labels, class_names, num_samples)
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"可视化结果已保存到 {save_path}")
    else:
        plt.show()

def visualize_feature_maps(model, image, layer_name=None, device='cpu', save_dir='feature_maps'):
    """
    可视化CNN模型的特征图
    
    参数:
        model: PyTorch模型
        image: 输入图像，形状为[1, channels, height, width]
        layer_name: 要可视化的层名称，如果为None则使用第一个卷积层
        device: 计算设备(CPU/GPU)
        save_dir: 保存特征图的目录
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 移动图像到指定设备
    image = image.to(device)
    
    # 如果未指定层名称，使用第一个卷积层
    if layer_name is None:
        # 查找第一个卷积层
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                layer_name = name
                break
    
    # 定义钩子函数来保存特征图
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(name))
    
    # 前向传播
    with torch.no_grad():
        output = model(image)
    
    # 获取特征图
    feature_maps = activation[layer_name].squeeze(0)
    
    # 可视化特征图
    num_features = feature_maps.size(0)
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    fig = plt.figure(figsize=(20, 20))
    for i in range(num_features):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        
        # 获取特征图
        feature_map = feature_maps[i].cpu().numpy()
        
        # 显示特征图
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Feature {i+1}')
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps of Layer: {layer_name}', fontsize=16)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'{layer_name}_feature_maps.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"特征图已保存到 {save_path}")
    
    return feature_maps

def visualize_filters(model, layer_name=None, save_dir='filters'):
    """
    可视化CNN模型的卷积核(滤波器)
    
    参数:
        model: PyTorch模型
        layer_name: 要可视化的卷积层名称，如果为None则使用第一个卷积层
        save_dir: 保存滤波器可视化的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果未指定层名称，使用第一个卷积层
    if layer_name is None:
        # 查找第一个卷积层
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                layer_name = name
                break
    
    # 获取卷积核权重
    filters = None
    for name, module in model.named_modules():
        if name == layer_name:
            filters = module.weight.data.cpu().clone()
            break
    
    if filters is None:
        print(f"找不到层 {layer_name}")
        return
    
    # 归一化滤波器以便可视化
    n_filters, n_channels, h, w = filters.shape
    
    # 只取前64个滤波器进行可视化(如果有)
    n_filters = min(n_filters, 64)
    
    # 如果是单通道卷积核，直接可视化
    if n_channels == 1:
        filters = filters.view(n_filters, h, w).unsqueeze(1)
        grid = torchvision.utils.make_grid(filters, nrow=8, normalize=True, padding=1)
        plt.figure(figsize=(15, 15))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title(f'Filters of Layer: {layer_name}')
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'{layer_name}_filters.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"滤波器可视化已保存到 {save_path}")
    else:
        # 多通道卷积核，为每个输出通道绘制一张图
        for i in range(n_filters):
            filt = filters[i]
            filt = filt.unsqueeze(1)
            grid = torchvision.utils.make_grid(filt, nrow=8, normalize=True, padding=1)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.title(f'Filter {i+1} of Layer: {layer_name}')
            plt.axis('off')
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(save_dir, f'{layer_name}_filter_{i+1}.png')
            plt.savefig(save_path)
            plt.close()
        
        print(f"滤波器可视化已保存到 {save_dir}")
    
    return filters 