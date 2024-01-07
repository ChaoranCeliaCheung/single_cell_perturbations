import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import ast

df = pd.read_csv('/content/drive/My Drive/df.csv')
df_compounds = df['sm_name']

# 对 'SMILES' 列进行去重
unique_smiles = df['SMILES'].drop_duplicates()
print(unique_smiles)

# 提取从第六列开始的所有列名
column_names = df.columns[5:]  # 列索引从 0 开始，所以第六列的索引是 5

# 创建一个新的 DataFrame
gene_name = pd.DataFrame(column_names, columns=['GeneNames'])

#=============================================================================================================================================
# 导入LINCS L100的数据，并对数据进行筛选

sig_info = pd.read_csv("/content/drive/My Drive/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt", sep="\t")
gene_info = pd.read_csv("/content/drive/My Drive/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt", sep="\t", dtype=str)
pert_info = pd.read_csv("/content/drive/My Drive/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt", sep="\t", dtype=str)

# 检查哪些 GeneNames 存在于 gene_info 中
gene_name['ExistsInGeneInfo'] = gene_name['GeneNames'].isin(gene_info.index)

# 分离可以找到和无法找到的基因列表
found_genes = gene_name[gene_name['ExistsInGeneInfo']]
not_found_genes = gene_name[~gene_name['ExistsInGeneInfo']]

# 从 gene_info 中获取 pr_gene_id
found_genes = found_genes.join(gene_info['pr_gene_id'], on='GeneNames')

not_found_gene_ids = gene_info[~gene_info['pr_gene_id'].isin(found_genes['pr_gene_id'])]

# 提取这些行的 pr_gene_id 列
not_train_found_genes = not_found_gene_ids['pr_gene_id'].reset_index(drop=True)
print(not_train_found_genes)

# 显示结果
print("可以找到的基因")
print(found_genes)

print("\n database中无法找到对应的基因列表:")
print(not_found_genes)

print("\n 训练数据中不包含的基因列表:")
print(not_train_found_genes)

# 使用 'pert_ids' 在 'sig_info' 中查找 'sig_id'
sig_ids = sig_info[(sig_info['pert_itime'] == '24 h')]['sig_id']

# 导入gctx文件
gctoo_1 = parse("/content/drive/My Drive/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx", cid=sig_ids)

# 创建映射字典
sig_id_to_iname = pd.Series(sig_info.pert_id.values, index=sig_info.sig_id).to_dict()
gene_id_to_symbol = pd.Series(gene_info.pr_gene_symbol.values, index=gene_info.pr_gene_id).to_dict()
pert_iname_to_smiles = pd.Series(pert_info.canonical_smiles.values, index=pert_info.pert_id).to_dict()

# 替换索引
gctoo_1.data_df.columns = gctoo_1.data_df.columns.map(sig_id_to_iname)
gctoo_1.data_df.index = gctoo_1.data_df.index.map(gene_id_to_symbol)

# 应用映射
gctoo_1.data_df.columns = gctoo_1.data_df.columns.map(pert_iname_to_smiles)

# 交换 gctoo.data_df 的行和列
data_df_1 = gctoo_1.data_df.transpose()

# 打印更新后的 DataFrame
print(data_df_1)

#===================================================================================================================================
# SVD分解

data = data_df_1

# 保存原始行名（索引）
original_index = data.index

# 将 DataFrame 转换为 NumPy 数组进行 SVD
data_array = data.values

# 选择您想要保留的成分数量
n_components = 200  # 例如，保留前200个成分

# 创建 TruncatedSVD 实例并对数据进行降维
svd = TruncatedSVD(n_components=n_components)
reduced_data_array = svd.fit_transform(data_array)

# 将降维后的数据转换回 DataFrame，并重新赋予原始行名
reduced_data = pd.DataFrame(reduced_data_array, index=original_index)

# 输出降维后的数据
print("Reduced data: ", reduced_data)

#==============================================================================================================================================================
# 训练ChemBERTa

# texts = reduced_data.index.tolist()  # Extract texts (input)
texts = reduced_data.iloc[:, 0].tolist()
targets = reduced_data.iloc[:, 1:].values  # Extract targets (regression values)
print(texts[:5])
print(targets.size)


# Load pre-trained model tokenizer and model
tokenizer = None
if not tokenizer:
  tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
  model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

# Freeze all layers in the model except the embedding layer
for name, param in model.named_parameters():
    # print(name)
    if 'embedding' not in name:  # Freeze layers that are not the embeddings
        param.requires_grad = False

class SimpleRegressor(nn.Module):
    def __init__(self, bert_model):
        super(SimpleRegressor, self).__init__()
        self.bert = bert_model
        self.regressor = nn.Linear(600, 200)  # Adjust for your target size

    def forward(self, input_ids):
        outputs = self.bert(input_ids)[0][::,0,::]
        return self.regressor(outputs)

# Instantiate the regressor
regressor = SimpleRegressor(model)

# Use a regression-appropriate loss function like Mean Squared Error
criterion = nn.MSELoss()

# Update optimizer to include parameters of the regressor
optimizer = optim.Adam(regressor.parameters(), lr=1e-4)


# Convert targets to tensor
targets = torch.tensor(targets, dtype=torch.float)

# Tokenize input texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]

# Fine-tuning loop
for epoch in range(5):  # Adjust number of epochs as needed
    regressor.train()
    optimizer.zero_grad()
    outputs = regressor(input_ids)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")

#============================================================================================================================================================================================================
# Embedding 训练数据 

train_df = pd.read_csv('/content/drive/MyDrive/train_df.csv')
# print(train_df.loc[:,:])

def build_ChemBERTa_features(smiles_list):
    model.eval()
    embeddings = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)
    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt", padding=False, truncation=True)
            model_output = model(**encoded_input)
            embedding = model_output[0][::,0,::]
            embeddings[i] = embedding
            embedding = torch.mean(model_output[0], 1)
            embeddings_mean[i] = embedding
    return embeddings.numpy(), embeddings_mean.numpy()


def save_ChemBERTa_features(smiles_list, out_dir, on_train_data=False):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    emb, emb_mean = build_ChemBERTa_features(smiles_list)
    if on_train_data:
        np.save(f"{out_dir}/chemberta_train_original.npy", emb)
        np.save(f"{out_dir}/chemberta_train_original_mean.npy", emb_mean)
    else:
        np.save(f"{out_dir}/chemberta_test.npy", emb)
        np.save(f"{out_dir}/chemberta_test_mean.npy", emb_mean) 

smiles_list = train_df.loc[:,"SMILES"]
save_ChemBERTa_features(smiles_list, '/content/drive/MyDrive', True)
emb, mean = build_ChemBERTa_features(smiles_list)

# emb

combined_features = []
for i in range(614):
    ct_feature = ast.literal_eval(train_df.loc[i,"CellTypeVector"])
    tensor_feature = torch.tensor(ct_feature)
    embedding_tensor = torch.from_numpy(emb[i])
    combined_feature = torch.cat([embedding_tensor,tensor_feature]).tolist()

    label = ast.literal_eval(train_df.loc[i,"DataVector"])
    # tensor_label = torch.tensor(label)

    combined_features.append([combined_feature,label])

# for i, (tensor, vector) in enumerate(zip(tensors_list, vectors_list)):
#     torch.save(tensor, f'tensor_{i}.pt')
#     torch.save(vector, f'vector_{i}.pt')

# data = {'TensorFile': [f'tensor_{i}.pt' for i in range(614)],
#         'VectorFile': [f'vector_{i}.pt' for i in range(614)]}

# df = pd.DataFrame(data)
# df.to_csv('tensor_vector_table.csv', index=True)
# # combined_features = torch.stack(combined_features)

# 将数据转换为 pandas DataFrame
df = pd.DataFrame(combined_features)

# 将 DataFrame 保存为 CSV 文件
df.to_csv('/content/drive/My Drive/combined_features.csv', index=False)



