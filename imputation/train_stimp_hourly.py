import torch
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
import numpy as np
import argparse
import sys

# 强制添加项目根目录到路径（确保导入无错）
sys.path.insert(0, os.getcwd())
from dataset import HimawariHourlyDataset  # 从dataset包导入小时数据集类
from utils import check_dir, masked_mae, masked_mse, seed_everything
from model.graphdiffusion import IAP_base  # 导入STIMP核心模型

# -------------------------- 命令行参数（与预处理脚本一致） --------------------------
parser = argparse.ArgumentParser(description='STIMP小时数据插值模型训练（严格遵循：训练插值→生成插值数据→训练预测→生成预测）')

# 必需参数：预处理后.npy数据根目录
parser.add_argument('--raw_data_path', type=str, required=True, help='预处理后.npy数据根目录（如E:\\1workinANHUA\\4\\model_training\\hourly_samples）')
# 区域与尺寸参数
parser.add_argument('--area', type=str, default='himawari', help='固定为himawari')
parser.add_argument('--height', type=int, default=128, help='空间高度（与预处理下采样尺寸一致）')
parser.add_argument('--width', type=int, default=128, help='空间宽度（与预处理下采样尺寸一致）')
# 序列长度参数（必须与预处理脚本一致）
parser.add_argument('--in_len', type=int, default=12, help='输入序列长度（=预处理SEQ_LEN）')
parser.add_argument('--out_len', type=int, default=1, help='输出序列长度（=预处理PRED_LEN）')
# 训练参数
parser.add_argument('--missing_ratio', type=float, default=0.1, help='缺失率（0-1）')
parser.add_argument('--epochs', type=int, default=500, help='训练轮次')
parser.add_argument('--batch_size', type=int, default=1, help='批次大小（高分辨率数据建议=1）')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--wd', type=float, default=1e-4, help='权重衰减')
parser.add_argument('--test_freq', type=int, default=50, help='每n轮测试一次并保存模型')
# 模型参数（与原STIMP一致）
parser.add_argument('--embedding_size', type=int, default=32, help='嵌入维度')
parser.add_argument('--hidden_channels', type=int, default=32, help='隐藏层维度')
parser.add_argument('--diffusion_embedding_size', type=int, default=64, help='扩散模型嵌入维度')
parser.add_argument('--side_channels', type=int, default=1, help='辅助特征通道数（SST单通道）')
parser.add_argument('--beta_start', type=float, default=0.0001, help='扩散beta起始值')
parser.add_argument('--beta_end', type=float, default=0.2, help='扩散beta结束值')
parser.add_argument('--num_steps', type=float, default=50, help='去噪步数')
parser.add_argument('--num_samples', type=int, default=10, help='采样数量')
parser.add_argument('--schedule', type=str, default='quad', help='噪声调度类型')
parser.add_argument('--target_strategy', type=str, default='random', help='掩码策略（random/block）')
parser.add_argument('--num_heads', type=int, default=8, help='自注意力头数')

if __name__ == '__main__':
    config = parser.parse_args()

    # 训练输出目录（自动创建，按参数区分）
    base_dir = f"./tmp/imputation/{config.in_len}/{config.area}/STIMP_hourly_missing_{config.missing_ratio}/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    check_dir(base_dir)  # 确保目录存在
    seed_everything(1234)  # 固定随机种子，保证可复现

    # 日志配置（保存训练过程）
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(base_dir, f'train_log_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(message)s'
    )
    print("="*80)
    print("STIMP小时数据插值模型训练（严格遵循你的流程：训练插值→生成插值数据→训练预测→生成预测）")
    print("="*80)
    print("配置参数：")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    logging.info(f"配置参数：{vars(config)}")

    # 加载小时数据数据集（从dataset包导入，无报错）
    print("\n" + "="*80)
    print("加载数据集...")
    train_dataset = HimawariHourlyDataset(config, mode="train")
    test_dataset = HimawariHourlyDataset(config, mode="test")
    print(f"数据集加载完成：训练集{len(train_dataset)}样本，测试集{len(test_dataset)}样本")

    # 数据加载器（单进程，避免多进程冲突）
    train_dloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 单进程
        pin_memory=True  # 加速GPU数据传输
    )
    test_dloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 加载空间邻接矩阵（预处理脚本生成）
    print("\n" + "="*80)
    print("加载空间邻接矩阵...")
    adj_path = os.path.join(config.raw_data_path, "spatial_graph.npy")
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"未找到邻接矩阵！请先运行预处理脚本：{adj_path}")
    adj = np.load(adj_path)
    adj = torch.from_numpy(adj).float().to(device)
    # 邻接矩阵下采样（与数据空间尺寸匹配）
    n_nodes = config.height * config.width
    if adj.shape[0] != n_nodes:
        print(f"邻接矩阵下采样：{adj.shape[0]} → {n_nodes}")
        adj = torch.nn.functional.interpolate(
            adj.unsqueeze(0).unsqueeze(0),
            size=(n_nodes, n_nodes),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0)
        adj = (adj > 0.5).float()  # 二值化，确保是邻接矩阵
    print(f"邻接矩阵加载完成（形状：{adj.shape}）")

    # 数据边界（与预处理标准化一致）
    low_bound = torch.from_numpy(train_dataset.min).float().to(device)
    high_bound = torch.from_numpy(train_dataset.max).float().to(device)
    print(f"数据边界：min={low_bound.item():.2f}℃，max={high_bound.item():.2f}℃")

    # 初始化模型和优化器
    print("\n" + "="*80)
    print("初始化模型...")
    model = IAP_base(config, low_bound, high_bound).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    # 学习率调度器（75%/90%轮次衰减）
    p1 = int(0.75 * config.epochs)
    p2 = int(0.9 * config.epochs)
    optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[p1, p2],
        gamma=0.1
    )
    print(f"模型初始化完成（参数总数：{sum(p.numel() for p in model.parameters()):,}）")

    # 训练主循环（插值模型训练，不改变你的流程）
    best_mae_sst = 100.0  # 记录最佳SST插值MAE
    print("\n" + "="*80)
    train_process = tqdm(range(1, config.epochs + 1), desc="训练进度")
    for epoch in train_process:
        model.train()  # 训练模式
        optimizer_scheduler.step(epoch)  # 更新学习率
        data_time_m = AverageMeter()  # 数据加载时间统计
        losses_m = AverageMeter()  # 训练损失统计
        end = time.time()

        # 遍历训练数据
        for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
            # 数据转GPU/CPU
            datas = datas.float().to(device)
            data_ob_masks = data_ob_masks.to(device)
            data_gt_masks = data_gt_masks.to(device)
            labels = labels.to(device)
            label_masks = label_masks.to(device)

            # 计算训练损失（调用STIMP模型的trainstep方法）
            loss = model.trainstep(datas, data_ob_masks, adj, is_train=1)
            losses_m.update(loss.item(), datas.size(0))  # 累计损失
            data_time_m.update(time.time() - end)  # 累计数据加载时间

            # 反向传播+梯度裁剪
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()
            torch.cuda.synchronize()  # 同步GPU
            end = time.time()

        # 打印训练日志
        log_str = f"Epoch {epoch:3d} | 训练损失: {losses_m.avg:.4f} | 数据加载时间: {data_time_m.avg:.4f}s"
        train_process.set_description(log_str)
        logging.info(log_str)

        # 测试与保存最佳模型（每test_freq轮）
        if epoch % config.test_freq == 0 and epoch != 0:
            model.eval()  # 评估模式
            sst_mae_list, sst_mse_list = [], []
            with torch.no_grad():  # 关闭梯度计算，节省内存
                for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
                    # 数据转设备
                    datas = datas.float().to(device)
                    data_ob_masks = data_ob_masks.to(device)
                    data_gt_masks = data_gt_masks.to(device)
                    labels = labels.to(device)
                    label_masks = label_masks.to(device)

                    # 模型插补（生成完整数据）
                    imputed_data = model.impute(datas, data_gt_masks, adj, config.num_samples)
                    imputed_data = imputed_data.median(dim=1).values  # 取中位数，稳定结果

                    # 计算评估指标（仅在缺失区域）
                    mask = (data_ob_masks - data_gt_masks).cpu()  # 1=缺失区域（需要评估）
                    sst_mae = masked_mae(imputed_data[:, :, 0].cpu(), datas[:, :, 0].cpu(), mask[:, :, 0])
                    sst_mse = masked_mse(imputed_data[:, :, 0].cpu(), datas[:, :, 0].cpu(), mask[:, :, 0])
                    sst_mae_list.append(sst_mae)
                    sst_mse_list.append(sst_mse)

            # 计算平均指标
            sst_mae = torch.stack(sst_mae_list).mean()
            sst_mse = torch.stack(sst_mse_list).mean()
            test_log = f"测试结果 | Epoch {epoch:3d} | SST MAE: {sst_mae:.4f} | SST MSE: {sst_mse:.4f}"
            print("\n" + "="*80)
            print(test_log)
            logging.info(test_log)

            # 保存最佳模型（按MAE排序）
            if sst_mae < best_mae_sst:
                best_mae_sst = sst_mae
                model_path = os.path.join(base_dir, f'best_model_epoch_{epoch}_mae_{sst_mae:.4f}.pt')
                torch.save(model, model_path)
                logging.info(f"保存最佳模型：{model_path}（MAE: {best_mae_sst:.4f}）")
                print(f"保存最佳模型到：{model_path}")
            print("="*80 + "\n")

    # 训练完成
    print("\n" + "="*80)
    print("🎉 STIMP小时数据插值模型训练完成！")
    print(f"最佳插值MAE：{best_mae_sst:.4f}")
    print(f"最佳模型位置：{os.path.join(base_dir, 'best_model_*.pt')}")
    print("下一步：使用该模型生成插值数据（完整无缺失），进入你的'生成插值数据→训练预测→生成预测'流程")
    print("="*80)
    logging.info(f"训练完成！最佳MAE：{best_mae_sst:.4f}")