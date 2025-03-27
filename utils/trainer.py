import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    """
    CNN模型训练器类，负责模型的训练、验证和测试过程管理
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 learning_rate=0.001, weight_decay=1e-5, device=None):
        """
        初始化训练器
        
        参数:
            model: PyTorch模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            learning_rate: 学习率
            weight_decay: L2正则化系数
            device: 训练设备 (CPU/GPU)
        """
        # 如果未指定设备，自动选择可用GPU或CPU
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 定义损失函数为负对数似然损失(与log_softmax输出匹配)
        self.criterion = nn.NLLLoss()
        
        # 定义优化器为Adam，包含权重衰减(L2正则化)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 学习率调度器，随着训练进行降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 创建TensorBoard日志目录
        self.log_dir = os.path.join('runs', time.strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(self.log_dir)
        
        # 保存训练过程中的指标
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"模型将在 {self.device} 上训练")
        
    def train_epoch(self, epoch):
        """
        训练一个完整的epoch
        
        参数:
            epoch: 当前的训练轮次
            
        返回:
            average_loss: 平均训练损失
            accuracy: 训练精度
        """
        self.model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            # 将数据移到指定设备
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(data)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            self.optimizer.step()
            
            # 累计损失
            running_loss += loss.item()
            
            # 计算精度
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条显示的损失和精度
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        # 计算平均损失和精度
        average_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # 记录到TensorBoard
        self.writer.add_scalar('Loss/train', average_loss, epoch)
        self.writer.add_scalar('Accuracy/train', accuracy, epoch)
        
        # 保存指标
        self.train_losses.append(average_loss)
        self.train_accuracies.append(accuracy)
        
        return average_loss, accuracy
        
    def validate(self, epoch):
        """
        在验证集上评估模型
        
        参数:
            epoch: 当前的训练轮次
            
        返回:
            average_loss: 平均验证损失
            accuracy: 验证精度
        """
        self.model.eval()  # 设置为评估模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 禁用梯度计算
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 累计损失
                running_loss += loss.item()
                
                # 计算精度
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 计算平均损失和精度
        average_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # 记录到TensorBoard
        self.writer.add_scalar('Loss/validation', average_loss, epoch)
        self.writer.add_scalar('Accuracy/validation', accuracy, epoch)
        
        # 保存指标
        self.val_losses.append(average_loss)
        self.val_accuracies.append(accuracy)
        
        # 更新学习率调度器
        self.scheduler.step(average_loss)
        
        return average_loss, accuracy
        
    def train(self, num_epochs, save_dir='checkpoints'):
        """
        训练模型指定轮次
        
        参数:
            num_epochs: 训练轮次
            save_dir: 模型保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 追踪最佳验证准确率
        best_val_acc = 0.0
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 在验证集上评估
            val_loss, val_acc = self.validate(epoch)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"Best model saved with validation accuracy: {val_acc:.2f}%")
            
            # 保存最后一个epoch的模型
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        print("训练完成!")
        
    def test(self, model_path=None):
        """
        在测试集上评估模型
        
        参数:
            model_path: 要加载的模型路径，如果为None则使用当前模型
            
        返回:
            test_loss: 测试损失
            test_accuracy: 测试精度
        """
        # 加载模型权重(如果指定)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()  # 设置为评估模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        # 禁用梯度计算
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Testing"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 累计损失
                running_loss += loss.item()
                
                # 计算精度
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 收集预测和目标用于混淆矩阵
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算平均损失和精度
        test_loss = running_loss / len(self.test_loader)
        test_accuracy = 100. * correct / total
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # 创建并保存混淆矩阵
        self._plot_confusion_matrix(np.array(all_preds), np.array(all_targets))
        
        return test_loss, test_accuracy
    
    def _plot_training_curves(self):
        """绘制训练和验证损失及准确率曲线"""
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率曲线
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()
        
    def _plot_confusion_matrix(self, y_pred, y_true, class_names=None):
        """
        绘制混淆矩阵
        
        参数:
            y_pred: 预测标签
            y_true: 真实标签
            class_names: 类别名称列表
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 如果未提供类别名称，使用数字
        if class_names is None:
            if np.max(y_true) == 9:  # MNIST数据集
                class_names = [str(i) for i in range(10)]
            else:
                class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('confusion_matrix.png')
        plt.close() 