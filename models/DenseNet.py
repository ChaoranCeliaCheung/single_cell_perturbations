import torch.nn as nn
import torch.nn.functional as F

# 定义 DenseNet 模型
class DenseNetModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# 数据和模型参数
input_dim = 606
output_dim = 18211