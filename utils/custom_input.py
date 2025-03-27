import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path, target_size=(28, 28)):
    """
    预处理用户提供的手写数字图像，转换为模型可接受的格式
    
    参数:
        image_path: 图像文件路径
        target_size: 目标大小，MNIST为28x28
        
    返回:
        处理后的图像张量，形状为 [1, 1, 28, 28]
    """
    # 读取图像并转换为灰度
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 反转颜色（假设手写数字是深色背景上的浅色笔画）
    gray = cv2.bitwise_not(gray)
    
    # 二值化处理
    _, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓以定位数字
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大轮廓，这应该是数字
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 从原图中裁剪数字区域
        digit_roi = binary[y:y+h, x:x+w]
        
        # 添加边距
        padding = int(min(w, h) * 0.2)
        padded_roi = cv2.copyMakeBorder(digit_roi, padding, padding, padding, padding, 
                                         cv2.BORDER_CONSTANT, value=0)
        
        # 调整大小到目标尺寸
        resized = cv2.resize(padded_roi, target_size, interpolation=cv2.INTER_AREA)
    else:
        # 如果没有找到轮廓，直接调整整个图像大小
        resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    # 标准化到[0, 1]区间
    normalized = resized.astype(np.float32) / 255.0
    
    # 应用MNIST数据集的均值和标准差进行规范化
    normalized = (normalized - 0.1307) / 0.3081
    
    # 转换为PyTorch张量，并添加批次和通道维度
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor

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
    从网络摄像头捕获图像
    
    参数:
        output_path: 输出图像的路径
        countdown: 倒计时秒数
    
    返回:
        保存图像的路径
    """
    try:
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return None
        
        # 倒计时
        for i in range(countdown, 0, -1):
            ret, frame = cap.read()
            if not ret:
                print("无法获取画面")
                cap.release()
                return None
            
            # 添加倒计时文本
            cv2.putText(frame, f"拍摄倒计时: {i}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示画面
            cv2.imshow('Camera', frame)
            cv2.waitKey(1000)  # 等待1秒
        
        # 捕获图像
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_path, frame)
            print(f"图像已保存到 {output_path}")
        
        # 释放摄像头
        cap.release()
        cv2.destroyAllWindows()
        
        return output_path if ret else None
    
    except Exception as e:
        print(f"捕获图像时出错: {e}")
        return None

def create_drawing_canvas(output_path='custom_digit.jpg', canvas_size=(500, 500)):
    """
    创建一个简单的绘图界面，让用户手写数字
    
    参数:
        output_path: 保存绘制图像的路径
        canvas_size: 画布大小
        
    返回:
        保存图像的路径
    """
    try:
        import cv2
        import numpy as np
        
        # 创建空白画布
        canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
        draw = False
        last_x, last_y = -1, -1
        
        # 鼠标回调函数
        def draw_circle(event, x, y, flags, param):
            nonlocal draw, last_x, last_y, canvas
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # 开始绘制
                draw = True
                last_x, last_y = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if draw:
                    # 绘制线条
                    cv2.line(canvas, (last_x, last_y), (x, y), (0, 0, 0), 20)
                    last_x, last_y = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                # 停止绘制
                draw = False
        
        # 创建窗口和回调
        cv2.namedWindow('Drawing Canvas')
        cv2.setMouseCallback('Drawing Canvas', draw_circle)
        
        while True:
            # 显示画布
            display_canvas = canvas.copy()
            cv2.putText(display_canvas, "绘制数字后按 's' 保存, 按 'c' 清空, 按 'q' 退出", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Drawing Canvas', display_canvas)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # 保存图像
                cv2.imwrite(output_path, canvas)
                print(f"绘制的数字已保存到 {output_path}")
                break
            elif key == ord('c'):
                # 清空画布
                canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
            elif key == ord('q'):
                # 退出
                return None
        
        cv2.destroyAllWindows()
        return output_path
    
    except Exception as e:
        print(f"创建绘图界面时出错: {e}")
        return None 