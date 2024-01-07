import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, combined_input_dim, data_vector_dim, transformer_dim, nhead, num_transformer_layers, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Linear(combined_input_dim, transformer_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer = nn.Transformer(d_model=transformer_dim, nhead=nhead, num_encoder_layers=num_transformer_layers, batch_first=True)
        self.out = nn.Linear(transformer_dim, data_vector_dim)

    def forward(self, src):
        src = self.src_embedding(src)
        src = F.gelu(src)
        src = self.dropout(src)
        transformer_output = self.transformer(src, src)
        output = self.out(F.gelu(transformer_output))
        return output
    

# 数据和模型参数
combined_input_dim = 606
data_vector_dim = 18211
transformer_dim = 512
nhead = 8
num_transformer_layers = 3
dropout_rate = 0.1
num_epochs = 50
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



