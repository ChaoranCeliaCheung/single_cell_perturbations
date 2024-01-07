import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_filters, kernel_size, stride, pool_kernel_size, pool_stride, dropout_rate):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.dropout = nn.Dropout(dropout_rate)

        # 动态计算卷积和池化后的长度
        conv_out_length = (606 - kernel_size) // stride + 1
        pooled_out_length = (conv_out_length - pool_kernel_size) // pool_stride + 1
        self.fc = nn.Linear(num_filters * pooled_out_length, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 数据和模型参数    
input_dim = 1  # 输入数据的通道数
output_dim = 18211  # 输出数据的维度
num_filters = 64  # 卷积层的过滤器数量
kernel_size = 3  # 卷积核大小
stride = 1  # 卷积步长
pool_kernel_size = 2  # 池化核大小
pool_stride = 2  # 池化步长
dropout_rate = 0.3  # Dropout比率
batch_size = 32
num_epochs = 50
learning_rate = 0.01
