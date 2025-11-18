import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

# 复用预测模型类
class STIMPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_len):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, out_len * input_dim)
        self.out_len = out_len
        self.input_dim = input_dim

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.reshape(-1, self.out_len, self.input_dim)


# 主逻辑
from dataset.dataset_prediction import PRE8dPredDataset
from utils import check_dir

parser = argparse.ArgumentParser(description='STIMP生成预测数据')

# 路径与参数
parser.add_argument('--area', type=str, default='Himawari_SST', help='数据区域')
parser.add_argument('--imputed_data_path', type=str, required=True, help='插补数据路径')
parser.add_argument('--checkpoint_path', type=str, required=True, help='预测模型权重')
parser.add_argument('--output_path', type=str, required=True, help='输出路径')
parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
parser.add_argument('--in_len', type=int, default=46, help='输入时间步')
parser.add_argument('--out_len', type=int, default=46, help='预测时间步')

# 模型参数
parser.add_argument('--hidden_dim', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=2)

config = parser.parse_args()

# 设备与目录
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
check_dir(config.output_path)

# 加载测试集
dataset = PRE8dPredDataset(config, mode='test')
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0
)

# 加载模型
model = STIMPPredictor(
    input_dim=1,
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    out_len=config.out_len
).to(device)
model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
model.eval()

# 生成预测数据
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc='生成预测数据'):
        inputs, labels, _ = [x.to(device) for x in batch]
        pred = model(inputs)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# 保存
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
np.save(os.path.join(config.output_path, 'sst_predicted.npy'), all_preds)
np.save(os.path.join(config.output_path, 'sst_ground_truth.npy'), all_labels)
print(f"预测数据保存至 {config.output_path}，形状: {all_preds.shape}")