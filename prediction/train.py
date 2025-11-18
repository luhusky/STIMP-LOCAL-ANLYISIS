import torch
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import numpy as np

sys.path.insert(0, os.getcwd())
from dataset.dataset_prediction import PRE8dDataset  # 用修改后的数据集类
from utils import check_dir, masked_mse, seed_everything
from utils import AverageMeter
from model.graphtransformer import GraphTransformer
from model.mtgnn import MTGNN

parser = argparse.ArgumentParser(description='SST预测训练')
parser.add_argument('--area', type=str, default='Himawari', help='区域')
parser.add_argument('--method', type=str, default='GraphTransformer', help='模型')
parser.add_argument('--index', type=int, default=0, help='数据集索引')
parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
parser.add_argument('--wd', type=float, default=1e-6, help='权重衰减')
parser.add_argument('--test_freq', type=int, default=20, help='测试频率')
parser.add_argument('--hidden_dim', type=int, default=8, help='隐藏维度')
parser.add_argument('--in_len', type=int, default=46, help='输入长度')
parser.add_argument('--out_len', type=int, default=46, help='输出长度')
parser.add_argument('--missing_ratio', type=float, default=0.1, help='缺失率')
parser.add_argument('--use_imputed', action='store_true', help='是否使用插值数据')  # 新增：控制是否用插值数据

config = parser.parse_args()

# 区域维度配置（新增Himawari）
if config.area=="MEXICO":
    config.height, config.width = 36, 120
elif config.area=="PRE":
    config.height, config.width = 60, 96
elif config.area=="Chesapeake":
    config.height, config.width = 60, 48
elif config.area=="Yangtze":
    config.height, config.width = 96, 72
elif config.area=="Himawari":
    config.height, config.width = 128, 128  # Himawari的128x128
else:
    print("Not Implement")

# 路径与设备
flag = "with_imputation" if config.use_imputed else "without_imputation"
base_dir = f"./log/prediction/{config.area}/{config.method}/{flag}/"
check_dir(base_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_everything(1234)

# 日志配置
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(base_dir, f'{timestamp}_dataset_{config.index}.log'),
    format='%(asctime)s - %(message)s'
)
print(config)
logging.info(config)

# 数据加载（使用修改后的数据集类，支持use_imputed参数）
train_dataset = PRE8dDataset(config, mode='train', use_imputed=config.use_imputed)
train_dloader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=2)
test_dloader = DataLoader(PRE8dDataset(config, mode='test', use_imputed=config.use_imputed), config.batch_size, shuffle=False)

# 辅助数据
adj = np.load(f"./data/{config.area}/adj.npy")
adj = torch.from_numpy(adj).float().to(device)
is_sea = torch.from_numpy(train_dataset.area).bool().to(device)
mean = torch.from_numpy(train_dataset.mean).float().to(device)
std = torch.from_numpy(train_dataset.std).float().to(device)

# 模型初始化（不变）
assert config.method in ["GraphTransformer", "MTGNN"]
if config.method == "GraphTransformer":
    model = GraphTransformer(config, adj, is_sea, mean, std).to(device)
elif config.method == "MTGNN":
    model = MTGNN(
        adj=adj, gcn_true=True, build_adj=False, kernel_set=[7,7], kernel_size=7,
        gcn_depth=1, num_nodes=adj.shape[0], dropout=0.3, subgraph_size=20, node_dim=8,
        dilation_exponential=2, conv_channels=2, residual_channels=2, skip_channels=4,
        end_channels=8, seq_length=config.in_len, in_dim=1, out_dim=config.out_len,
        layers=2, propalpha=0.5, tanhalpha=3, layer_norm_affline=False
    ).to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
scheduler = CosineAnnealingLR(optimizer, config.epochs)
best_mae_sst = 100  # 变量名改为sst

# 训练循环（仅修改变量名）
train_process = tqdm(range(config.epochs))
for epoch in train_process:
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    scheduler.step(epoch)
    end = time.time()

    for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
        datas, labels, label_masks = (
            datas.float().to(device),
            labels.to(device),
            label_masks.to(device)
        )

        # 标准化
        means = datas.mean(1, keepdim=True).detach()
        datas = datas - means
        stdev = torch.sqrt(torch.var(datas, dim=1, keepdim=True, unbiased=False) + 1e-5)
        datas /= stdev

        # 预测与损失
        prediction = model(datas)
        prediction = prediction * stdev + means
        loss = masked_mse(prediction, labels, label_masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录指标
        losses_m.update(loss.item(), datas.size(0))
        data_time_m.update(time.time() - end)
        end = time.time()
        torch.cuda.synchronize()

    # 训练日志
    log_buffer = f"Epoch {epoch} | 训练损失: {losses_m.avg:.4f} | 时间: {data_time_m.avg:.4f}"
    train_process.write(log_buffer)
    logging.info(log_buffer)
    losses_m.reset()
    data_time_m.reset()

    # 测试（变量名改为sst）
    if epoch % config.test_freq == 0 and epoch != 0:
        model.eval()
        predictions = []
        labels_list = []
        label_masks_list = []

        with torch.no_grad():
            for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
                datas, labels, label_masks = (
                    datas.float().to(device),
                    labels.to(device),
                    label_masks.to(device)
                )

                means = datas.mean(1, keepdim=True).detach()
                datas = datas - means
                stdev = torch.sqrt(torch.var(datas, dim=1, keepdim=True, unbiased=False) + 1e-5)
                datas /= stdev

                prediction = model(datas)
                prediction = prediction * stdev + means

                predictions.append(prediction[:, :, 0].cpu())
                labels_list.append(labels[:, :, 0].cpu())
                label_masks_list.append(label_masks[:, :, 0].cpu())

        # 计算指标（变量名改为sst）
        predictions = torch.cat(predictions, 0)
        labels = torch.cat(labels_list, 0)
        label_masks = torch.cat(label_masks_list, 0)

        sst_mae = (torch.abs(predictions - labels) * label_masks).sum([1]) / (label_masks.sum([1]) + 1e-5)
        sst_mse = (((predictions - labels) * label_masks) **2).sum([1]) / (label_masks.sum([1]) + 1e-5)
        sst_mae_mean = sst_mae.mean()
        sst_mse_mean = sst_mse.mean()

        log_buffer = f"测试结果 | MAE: {sst_mae_mean:.4f} | MSE: {sst_mse_mean:.4f}"
        train_process.write(log_buffer)
        logging.info(log_buffer)

        # 保存最优模型
        if sst_mae_mean < best_mae_sst:
            best_mae_sst = sst_mae_mean
            torch.save(model, os.path.join(base_dir, f"best_model_{config.method}_{config.index}.pt"))
            np.save(os.path.join(base_dir, f"prediction_{config.index}.npy"), predictions.numpy())
            logging.info(f"保存最优模型，MAE: {best_mae_sst:.4f}")

# 保存真实标签
if not os.path.exists(f"./data/{config.area}/trues.npy"):
    np.save(f"./data/{config.area}/trues.npy", labels.cpu().numpy())
    np.save(f"./data/{config.area}/true_masks.npy", label_masks.cpu().numpy())

