import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from models.transformer import TransformerModel  
from models.DenseNet import DenseNetModel       
from models.CNN import CNNModel               
from train_env_test import train_model, plot_losses, test_weighted_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from sklearn.model_selection import train_test_split



# 加载数据
data_path = '/content/drive/My Drive/data.npz'
data = np.load(data_path)
X = data['X']
y = data['Y']

# 设置数据划分的比例
test_size = 0.1   # 测试集占10%
val_size = 0.15   # 验证集占训练集的15%

# 先分离出统一的测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 对 Transformer 的数据进行预处理
# 计算 IQR
label_sums_train_val = np.sum(y_train_val, axis=1)
q25, q75 = np.percentile(label_sums_train_val, 25), np.percentile(label_sums_train_val, 75)
iqr = q75 - q25

# 定义异常值的范围
cut_off = iqr * 2
lower, upper = q25 - cut_off, q75 + cut_off

# 识别异常值
outliers = (label_sums_train_val < lower) | (label_sums_train_val > upper)

# 过滤掉异常值
X_filtered_train_val = X_train_val[~outliers]
y_filtered_train_val = y_train_val[~outliers]

# 分离 Transformer 的训练集和验证集
X_train_transformer, X_val_transformer, y_train_transformer, y_val_transformer = train_test_split(X_filtered_train_val, y_filtered_train_val, test_size=val_size, random_state=42)

# 分离 DenseNet 和 CNN 的训练集和验证集
X_train_densenet, X_val_densenet, y_train_densenet, y_val_densenet = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)
X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)


# 初始化 KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化保存最佳模型的变量
best_val_loss_transformer = float('inf')
best_model_transformer = None

best_val_loss_densenet = float('inf')
best_model_densenet = None

best_val_loss_cnn = float('inf')
best_model_cnn = None

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 在KFold交叉验证之前初始化存储损失的列表
all_train_losses_transformer = []
all_val_losses_transformer = []
all_train_losses_densenet = []
all_val_losses_densenet = []
all_train_losses_cnn = []
all_val_losses_cnn = []

# 损失函数
criterion = nn.SmoothL1Loss()

# KFold交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    # 数据加载
    train_loader_transformer = DataLoader(TensorDataset(torch.tensor(X_train_transformer, dtype=torch.float32), torch.tensor(y_train_transformer, dtype=torch.float32)), batch_size=32, shuffle=True)
    val_loader_transformer = DataLoader(TensorDataset(torch.tensor(X_val_transformer, dtype=torch.float32), torch.tensor(y_val_transformer, dtype=torch.float32)), batch_size=32, shuffle=False)
    train_loader_densenet = DataLoader(TensorDataset(torch.tensor(X_train_densenet, dtype=torch.float32), torch.tensor(y_train_densenet, dtype=torch.float32)), batch_size=32, shuffle=True)
    val_loader_densenet = DataLoader(TensorDataset(torch.tensor(X_val_densenet, dtype=torch.float32), torch.tensor(y_val_densenet, dtype=torch.float32)), batch_size=32, shuffle=False)
    train_loader_cnn = DataLoader(TensorDataset(torch.tensor(X_train_transformer, dtype=torch.float32), torch.tensor(y_train_transformer, dtype=torch.float32)), batch_size=32, shuffle=True)
    val_loader_cnn = DataLoader(TensorDataset(torch.tensor(X_val_transformer, dtype=torch.float32), torch.tensor(y_val_transformer, dtype=torch.float32)), batch_size=32, shuffle=False)

    # Transformer模型
    transformer_model = TransformerModel(...).to(device)
    transformer_optimizer = optim.RMSprop(transformer_model.parameters(), lr=0.0001, weight_decay=1e-5)
    transformer_criterion = nn.MSELoss()
    transformer_scheduler = ReduceLROnPlateau(transformer_optimizer, mode='min', factor=0.8, patience=10, verbose=True)
    train_losses_transformer, val_losses_transformer = train_model(transformer_model, train_loader_transformer, val_loader_transformer, transformer_criterion, transformer_optimizer, 50, transformer_scheduler, device)
    if np.mean(val_losses_cnn) < best_val_loss_transformer:
        best_val_loss_transformer = np.mean(val_losses_cnn)
        best_model_transformer = transformer_model.state_dict()

    # DenseNet模型
    densenet_model = DenseNetModel(...).to(device)
    densenet_optimizer = optim.Adam(densenet_model.parameters(), lr=0.00001, weight_decay=1e-7)
    densenet_criterion = nn.MSELoss()
    densenet_scheduler = ReduceLROnPlateau(densenet_optimizer, mode='min', factor=0.8, patience=10, verbose=True)
    train_losses_densenet, val_losses_densenet = train_model(densenet_model, train_loader_densenet, val_loader_densenet, densenet_criterion, densenet_optimizer, 50, densenet_scheduler, device)
    if np.mean(val_losses_densenet) < best_val_loss_densenet:
        best_val_loss_densenet = np.mean(val_losses_densenet)
        best_model_densenet = densenet_model.state_dict()

    # CNN模型
    cnn_model = CNNModel(...).to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001, weight_decay=1e-4)
    cnn_criterion = nn.MSELoss()
    cnn_scheduler = ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.9999, patience=500, verbose=True)
    train_losses_cnn, val_losses_cnn = train_model(cnn_model, train_loader_cnn, val_loader_cnn, cnn_criterion, cnn_optimizer, 50, cnn_scheduler, device)
    if np.mean(val_losses_cnn) < best_val_loss_cnn:
        best_val_loss_cnn = np.mean(val_losses_cnn)
        best_model_cnn = cnn_model.state_dict()

    # 收集每个模型的损失
    all_train_losses_transformer.append(train_losses_transformer)
    all_val_losses_transformer.append(val_losses_transformer)
    all_train_losses_densenet.append(train_losses_densenet)
    all_val_losses_densenet.append(val_losses_densenet)
    all_train_losses_cnn.append(train_losses_cnn)
    all_val_losses_cnn.append(val_losses_cnn)

# 保存每个模型的最佳版本
models_save_directory = '/content/drive/My Drive/saved_models'
os.makedirs(models_save_directory, exist_ok=True)

transformer_save_path = os.path.join(models_save_directory, 'best_transformer_model.pth')
torch.save(best_model_transformer, transformer_save_path)
print(f"Best Transformer model saved to: {transformer_save_path}")

densenet_save_path = os.path.join(models_save_directory, 'best_densenet_model.pth')
torch.save(best_model_densenet, densenet_save_path)
print(f"Best DenseNet model saved to: {densenet_save_path}")

cnn_save_path = os.path.join(models_save_directory, 'best_cnn_model.pth')
torch.save(best_model_cnn, cnn_save_path)
print(f"Best CNN model saved to: {cnn_save_path}")

# 画出每个模型的损失曲线
for i in range(5):  # 假设有5个折叠
    plot_losses(all_train_losses_transformer[i], all_val_losses_transformer[i], f"Transformer Fold {i+1}")
    plot_losses(all_train_losses_densenet[i], all_val_losses_densenet[i], f"DenseNet Fold {i+1}")
    plot_losses(all_train_losses_cnn[i], all_val_losses_cnn[i], f"CNN Fold {i+1}")


# 计算归一化系数
coeff_transformer = 1 / np.mean(all_val_losses_transformer[-1][-10:])
coeff_densenet = 1 / np.mean(all_val_losses_densenet[-1][-10:])
coeff_cnn = 1 / np.mean(all_val_losses_cnn[-1][-10:])
total_coeff = coeff_transformer + coeff_densenet + coeff_cnn
coeff_transformer /= total_coeff
coeff_densenet /= total_coeff
coeff_cnn /= total_coeff

test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=32, shuffle=False)

# 对加权模型进行测试
test_loss = test_weighted_model(
    [transformer_model, densenet_model, cnn_model],
    [coeff_transformer, coeff_densenet, coeff_cnn],
    test_loader,
    criterion,
    device
)

print("Test Loss of Weighted Model:", test_loss)