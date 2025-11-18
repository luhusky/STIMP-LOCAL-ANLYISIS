
import argparse

import torch
import os

from torch._C.cpp import nn

from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
import numpy as np


# 预测模型类（基于STIMP插补数据）
class STIMPPredictor(nn.Module):
    """STIMP预测模型（用于基于插补数据的预测训练）"""

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
        # x形状: [B, T_in, D]，其中D=1（SST单变量）
        _, (hn, _) = self.lstm(x)  # 取最后一层的隐藏状态
        out = self.fc(hn[-1])  # 用最后一层输出预测
        return out.reshape(-1, self.out_len, self.input_dim)  # [B, T_out, D]


# 主训练逻辑
import argparse
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
import numpy as np

from dataset.dataset_prediction import PRE8dPredDataset  # 预测数据集
from utils import check_dir, masked_mae, seed_everything

parser = argparse.ArgumentParser(description='STIMP预测训练（基于插补数据）')

# 数据参数
parser.add_argument('--area', type=str, default='Himawari_SST', help='数据区域')
parser.add_argument('--imputed_data_path', type=str, required=True, help='插补数据路径')
parser.add_argument('--in_len', type=int, default=46, help='输入时间步')
parser.add_argument('--out_len', type=int, default=46, help='预测时间步')

# 训练参数
parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
parser.add_argument('--epochs', type=int, default=150, help='总轮次')
parser.add_argument('--test_freq', type=int, default=50, help='每50轮保存权重')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')

# 模型参数
parser.add_argument('--hidden_dim', type=int, default=8, help='隐藏层维度')
parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')

config = parser.parse_args()

# 目录与设备
base_dir = f"./log/prediction/{config.area}/STIMP/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
check_dir(base_dir)
seed_everything(1234)

# 日志配置
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(base_dir, f'pred_train_{timestamp}.log'),
    format='%(asctime)s - %(message)s'
)
logging.info(f"配置: {config}")

# 加载插补数据集
train_dataset = PRE8dPredDataset(config, mode='train')
test_dataset = PRE8dPredDataset(config, mode='test')
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0
)

# 初始化预测模型
model = STIMPPredictor(
    input_dim=1,  # SST单变量
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    out_len=config.out_len
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# 训练主循环
best_mae = float('inf')
for epoch in tqdm(range(1, config.epochs + 1), desc='STIMP预测训练'):
    model.train()
    train_loss = AverageMeter()

    for batch in train_loader:
        inputs, labels, masks = [x.to(device) for x in batch]  # [B, T_in, 1], [B, T_out, 1], [B, T_out, 1]
        pred = model(inputs)  # [B, T_out, 1]
        loss = masked_mae(pred, labels, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), inputs.size(0))

    scheduler.step()
    logging.info(f"Epoch {epoch} | 训练损失: {train_loss.avg:.4f}")

    # 每50轮保存最优权重
    if epoch % config.test_freq == 0:
        model.eval()
        test_mae = AverageMeter()
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels, masks = [x.to(device) for x in batch]
                pred = model(inputs)
                mae = masked_mae(pred, labels, masks)
                test_mae.update(mae.item(), inputs.size(0))

        logging.info(f"Epoch {epoch} | 测试MAE: {test_mae.avg:.4f}")
        if test_mae.avg < best_mae:
            best_mae = test_mae.avg
            torch.save(
                model.state_dict(),
                os.path.join(base_dir, f"best_pred_model_epoch{epoch}.pth")
            )
            logging.info(f"保存最优预测模型（Epoch {epoch}，MAE: {best_mae:.4f}）")

logging.info(f"预测训练结束，最优MAE: {best_mae:.4f}")