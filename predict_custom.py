import argparse
import os
import torch
from models.cnn_model import SimpleCNN, DeepCNN
from utils.custom_input import create_drawing_canvas, predict_digit, capture_from_webcam

def main(args):
    """
    主函数，用于预测自定义手写数字图像
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # 创建模型
    print(f"加载{args.model}模型...")
    if args.model == 'simple':
        model = SimpleCNN()
    elif args.model == 'deep':
        model = DeepCNN()
    else:
        raise ValueError(f"不支持的模型: {args.model}")
    
    # 加载模型权重
    model_path = os.path.join(args.save_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, 'last_model.pth')
        if not os.path.exists(model_path):
            raise ValueError(f"找不到模型权重文件: {args.save_dir}/best_model.pth 或 {args.save_dir}/last_model.pth")
    
    print(f"从 {model_path} 加载模型权重...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # 设置类别名称
    if args.dataset == 'mnist':
        class_names = [str(i) for i in range(10)]
    elif args.dataset == 'fashion-mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        class_names = None
    
    # 根据输入模式获取图像
    if args.mode == 'file':
        if not os.path.exists(args.input):
            raise ValueError(f"找不到输入图像: {args.input}")
        image_path = args.input
        print(f"使用文件: {image_path}")
    elif args.mode == 'draw':
        print("打开绘图界面. 请使用鼠标绘制一个数字...")
        image_path = create_drawing_canvas('custom_digit.jpg')
        if image_path is None:
            print("用户取消绘制")
            return
    elif args.mode == 'camera':
        print("准备使用摄像头拍摄...")
        image_path = capture_from_webcam('custom_digit.jpg')
        if image_path is None:
            print("摄像头捕获失败")
            return
    else:
        raise ValueError(f"不支持的输入模式: {args.mode}")
    
    # 预测数字
    print(f"预测图像 {image_path} 中的数字...")
    predicted_class, confidence = predict_digit(model, image_path, device, class_names)
    
    # 输出结果
    if args.dataset == 'mnist':
        print(f"预测结果: 数字 {predicted_class}，置信度: {confidence:.2f}%")
    else:
        print(f"预测结果: {class_names[predicted_class]}，置信度: {confidence:.2f}%")
    
    print(f"预测可视化结果已保存到 custom_prediction.png")

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用CNN模型预测自定义手写数字')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='simple', 
                        choices=['simple', 'deep'],
                        help='要使用的CNN模型 (默认: simple)')
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fashion-mnist'],
                        help='模型训练的数据集 (默认: mnist)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', 
                        help='模型保存的目录 (默认: checkpoints)')
    
    # 输入参数
    parser.add_argument('--mode', type=str, default='draw', 
                        choices=['file', 'draw', 'camera'],
                        help='输入模式: file(使用文件), draw(绘制), camera(摄像头) (默认: draw)')
    parser.add_argument('--input', type=str, default=None,
                        help='输入图像的路径，仅在mode=file时使用')
    
    # 杂项参数
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA')
    
    args = parser.parse_args()
    
    main(args) 