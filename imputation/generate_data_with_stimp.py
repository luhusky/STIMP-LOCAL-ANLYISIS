import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.stimp import STIMP
from dataset.dataset_imputation import PRE8dDataset
from utils import check_dir

parser = argparse.ArgumentParser(description='STIMP生成插补数据')

# 路径与参数
parser.add_argument('--area', type=str, default='Himawari_SST', help='数据区域')
parser.add_argument('--missing_ratio', type=float, default=0.1, help='缺失率')
parser.add_argument('--checkpoint_path', type=str, required=True, help='模型权重路径')
parser.add_argument('--output_path', type=str, required=True, help='输出路径')
parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
parser.add_argument('--num_samples', type=int, default=5, help='生成样本数')
parser.add_argument('--in_len', type=int, default=46, help='输入时间步')
parser.add_argument('--out_len', type=int, default=46, help='输出时间步')

# 模型参数（与训练一致）
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--diffusion_embedding_size', type=int, default=32)
parser.add_argument('--num_steps', type=int, default=25)
parser.add_argument('--schedule', type=str, default='quad')
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.02)

config = parser.parse_args()

# 设备与目录
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
check_dir(config.output_path)

# 加载全量数据集
dataset = PRE8dDataset(config, mode='all')  # 加载所有数据
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0
)

# 加载辅助数据
data_dir = f"./data/{config.area}"
low_bound = torch.from_numpy(np.load(os.path.join(data_dir, 'min.npy'))).float().to(device)
high_bound = torch.from_numpy(np.load(os.path.join(data_dir, 'max.npy'))).float().to(device)
adj = torch.from_numpy(np.load(os.path.join(data_dir, 'adj.npy'))).float().to(device)

# 加载模型
model = STIMP(config, low_bound, high_bound).to(device)
model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
model.eval()

# 生成插补数据
all_imputed = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc='生成插补数据'):
        datas, masks, _, _, _ = [x.to(device) for x in batch]
        imputed = model.impute(datas, masks, adj, n_samples=config.num_samples)
        all_imputed.append(imputed.cpu().numpy())

# 保存
all_imputed = np.concatenate(all_imputed, axis=0)
np.save(os.path.join(config.output_path, 'sst_imputed.npy'), all_imputed)
print(f"插补数据保存至 {config.output_path}，形状: {all_imputed.shape}")