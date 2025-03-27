import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models.cnn_model import SimpleCNN, DeepCNN
from utils.data_loader import get_mnist_loaders, get_fashion_mnist_loaders
from utils.trainer import Trainer
from utils.visualization import visualize_model_predictions, visualize_feature_maps, visualize_filters

def main(args):
    """
    主函数，执行CNN模型的训练和评估
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子以便结果可复现
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # 加载数据
    print(f"正在加载{args.dataset}数据集...")
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers
        )
        # MNIST的类别名称
        class_names = [str(i) for i in range(10)]
    elif args.dataset == 'fashion-mnist':
        train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers
        )
        # Fashion-MNIST的类别名称
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 创建模型
    print(f"创建{args.model}模型...")
    if args.model == 'simple':
        model = SimpleCNN()
    elif args.model == 'deep':
        model = DeepCNN()
    else:
        raise ValueError(f"不支持的模型: {args.model}")
    
    # 打印模型结构
    print(model)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device
    )
    
    # 训练模型
    if args.train:
        print(f"开始训练模型，共{args.epochs}个epochs...")
        trainer.train(
            num_epochs=args.epochs,
            save_dir=args.save_dir
        )
        print("训练完成！")
    
    # 测试模型
    if args.test:
        print("在测试集上评估模型...")
        best_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            test_loss, test_acc = trainer.test(model_path=best_model_path)
        else:
            print(f"找不到最佳模型权重文件: {best_model_path}，使用当前模型进行测试")
            test_loss, test_acc = trainer.test()
    
    # 可视化模型预测
    if args.visualize:
        print("可视化模型预测结果...")
        # 确保模型加载了最佳权重
        best_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path) and not args.train:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # 可视化模型预测
        visualize_model_predictions(
            model=model,
            data_loader=test_loader,
            device=device,
            num_samples=16,
            class_names=class_names,
            save_path='model_predictions.png'
        )
        
        # 获取一个样本图像用于可视化特征图
        data_iter = iter(test_loader)
        images, _ = next(data_iter)
        sample_image = images[0:1]  # 选择第一张图像并保持批次维度
        
        # 可视化特征图
        visualize_feature_maps(
            model=model,
            image=sample_image,
            layer_name='conv1',  # 第一个卷积层的名称
            device=device,
            save_dir='feature_maps'
        )
        
        # 可视化卷积核
        visualize_filters(
            model=model,
            layer_name='conv1',  # 第一个卷积层的名称
            save_dir='filters'
        )

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CNN模型用于手写数字识别')
    
    # 数据集和模型参数
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fashion-mnist'],
                        help='要使用的数据集 (默认: mnist)')
    parser.add_argument('--model', type=str, default='simple', 
                        choices=['simple', 'deep'],
                        help='要使用的CNN模型 (默认: simple)')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='训练的批次大小 (默认: 64)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='训练的轮次 (默认: 10)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, 
                        help='权重衰减 (L2正则化) (默认: 1e-5)')
    parser.add_argument('--val-split', type=float, default=0.1, 
                        help='验证集比例 (默认: 0.1)')
    
    # 杂项参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子 (默认: 42)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', 
                        help='保存模型的目录 (默认: checkpoints)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA训练')
    parser.add_argument('--num-workers', type=int, default=2, 
                        help='数据加载的工作线程数 (默认: 2)')
    
    # 功能标志
    parser.add_argument('--train', action='store_true', default=True,
                        help='训练模型')
    parser.add_argument('--test', action='store_true', default=True,
                        help='测试模型')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='可视化模型预测和特征')
    
    args = parser.parse_args()
    
    main(args) 