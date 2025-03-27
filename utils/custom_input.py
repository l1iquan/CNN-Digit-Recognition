import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import os
import traceback

def preprocess_image(image_path, target_size=(28, 28)):
    """
    使用PIL替代OpenCV预处理用户提供的手写数字图像
    
    参数:
        image_path: 图像文件路径
        target_size: 目标大小，MNIST为28x28
        
    返回:
        处理后的图像张量，形状为 [1, 1, 28, 28]
    """
    try:
        # 读取图像并转换为灰度
        image = Image.open(image_path).convert('L')
        
        # 保存原始尺寸用于调试
        original_size = image.size
        print(f"原始图像尺寸: {original_size}")
        
        # 提取图像直方图用于分析
        hist = image.histogram()
        
        # 判断图像背景颜色（暗背景上的亮数字，或亮背景上的暗数字）
        # 计算图像平均亮度来决定是否需要反转
        avg_pixel = sum(i * w for i, w in enumerate(hist)) / sum(hist)
        print(f"图像平均亮度: {avg_pixel}")
        
        # 如果平均亮度大于128，认为是暗数字在亮背景上，需要反转
        if avg_pixel > 128:
            image = ImageOps.invert(image)
            print("图像已反转（亮背景暗数字）")
        else:
            print("图像未反转（暗背景亮数字）")
        
        # 增强对比度，提高数字与背景的区分度
        image = ImageOps.autocontrast(image, cutoff=5)
        
        # 应用阈值化处理（二值化）- 根据图像分布动态调整阈值
        # 计算自适应阈值 - 使用Otsu方法的简化版本
        pixels = list(image.getdata())
        non_empty_pixels = [p for p in pixels if p > 0]
        if non_empty_pixels:
            # 使用平均值作为阈值
            adaptive_threshold = sum(non_empty_pixels) / len(non_empty_pixels)
            # 确保阈值在合理范围内
            adaptive_threshold = max(70, min(adaptive_threshold, 180))
        else:
            adaptive_threshold = 128  # 默认值
        
        print(f"使用自适应阈值: {adaptive_threshold}")
        
        # 二值化
        image = image.point(lambda p: p > adaptive_threshold and 255)
        
        # 尝试找到并居中数字
        # 将图像转换为numpy数组以便处理
        img_array = np.array(image)
        
        # 找到非零像素的区域（这应该是数字区域）
        rows = np.any(img_array, axis=1)
        cols = np.any(img_array, axis=0)
        
        if np.any(rows) and np.any(cols):
            # 找到数字区域的边界
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # 确保边界是有效的
            if rmin < rmax and cmin < cmax:
                # 裁剪数字区域
                digit_region = img_array[rmin:rmax+1, cmin:cmax+1]
                
                # 转回PIL图像
                digit_image = Image.fromarray(digit_region)
                
                # 添加边距
                padding_ratio = 0.2  # 边距比例
                padding_x = int(digit_image.width * padding_ratio)
                padding_y = int(digit_image.height * padding_ratio)
                padded_image = ImageOps.expand(digit_image, border=(padding_x, padding_y, padding_x, padding_y), fill=0)
                
                # 更新图像为处理后的图像
                image = padded_image
                print(f"已提取并居中数字，大小: {image.size}")
        
        # 调整大小到目标尺寸（保持纵横比）
        ratio = min(target_size[0] / image.width, target_size[1] / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.LANCZOS)
        
        # 创建新的空白图像并将调整大小后的图像居中放置
        final_image = Image.new("L", target_size, 0)
        paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
        final_image.paste(resized_image, paste_position)
        
        # 转换为numpy数组
        img_array = np.array(final_image).astype(np.float32) / 255.0
        
        # 应用MNIST数据集的均值和标准差进行规范化
        img_array = (img_array - 0.1307) / 0.3081
        
        # 转换为PyTorch张量，并添加批次和通道维度
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return tensor
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        traceback.print_exc()
        raise

def predict_digit(model, image_path, device='cpu', class_names=None):
    """
    使用训练好的模型预测手写数字
    
    参数:
        model: 训练好的模型
        image_path: 图像文件路径
        device: 计算设备(CPU/GPU)
        class_names: 类别名称列表，用于Fashion-MNIST
        
    返回:
        预测的数字和概率
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 预处理图像
    processed_image = preprocess_image(image_path)
    processed_image = processed_image.to(device)
    
    # 使用模型进行预测
    with torch.no_grad():
        output = model(processed_image)
        probabilities = torch.exp(output)  # 将对数概率转换为概率
        top_prob, top_class = probabilities.topk(1, dim=1)
    
    # 获取预测结果
    predicted_class = top_class.item()
    confidence = top_prob.item() * 100
    
    # 使用类别名称(如果提供)
    if class_names is not None:
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = str(predicted_class)
    
    # 可视化预测结果
    plt.figure(figsize=(6, 6))
    
    # 读取原始图像
    original_image = Image.open(image_path)
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("原始图像")
    plt.axis('off')
    
    # 显示预处理后的图像
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"预处理后\n预测: {predicted_label}\n置信度: {confidence:.2f}%")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('custom_prediction.png')
    plt.close()
    
    return predicted_class, confidence

def capture_from_webcam(output_path='custom_digit.jpg', countdown=3):
    """
    从网络摄像头捕获图像 - 此功能需要cv2，在没有cv2情况下返回错误
    """
    print("警告：摄像头功能需要OpenCV支持")
    print("当前环境不支持OpenCV，无法使用摄像头功能")
    print("请使用文件模式(--mode file)代替")
    return None

def create_drawing_canvas(output_path='custom_digit.jpg', canvas_size=(500, 500)):
    """
    创建绘图界面 - 此功能需要cv2，在没有cv2的情况下返回错误
    """
    print("警告：绘图功能需要OpenCV支持")
    print("当前环境不支持OpenCV，无法使用绘图功能")
    print("请使用文件模式(--mode file)代替")
    return None 