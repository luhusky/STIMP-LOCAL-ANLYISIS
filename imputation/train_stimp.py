import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import os
import platform
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
import numpy as np
import argparse
import sys

# 检查操作系统
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    print("🔧 检测到 Windows 系统，进行兼容性调整")

# 导入 SwanLab
try:
    import swanlab

    SWANLAB_AVAILABLE = True
except ImportError:
    print(" SwanLab 未安装，使用 pip install swanlab 安装以获得更好的实验跟踪")
    SWANLAB_AVAILABLE = False

sys.path.insert(0, os.getcwd())
from dataset.dataset_imputation import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, seed_everything


class EarlyStopping:
    """早停止类，用于在验证损失不再改善时停止训练"""

    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 验证损失不再改善的等待轮数
            verbose (bool): 是否打印早停止信息
            delta (float): 认为有改善的最小变化量
            path (str): 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'🚨 早停止计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型当验证损失减少时'''
        if self.verbose:
            print(f'🎯 验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class TrainConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(description='STIMP插补训练（修复版本）')
        # 区域和数据路径 - 使用修复后的数据路径
        parser.add_argument('--area', type=str, default='Bohai', help='目标区域（如Bohai）')
        parser.add_argument('--raw_data_path', type=str, default=r'E:\1workinANHUA\4\data\Himawari-bohaidata-fixed',
                            help='修复后的原始数据路径')
        # 训练参数
        parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
        parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
        parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
        parser.add_argument('--wd', type=float, default=1e-4, help='权重衰减')
        parser.add_argument('--test_freq', type=int, default=25, help='测试频率')
        # 早停止参数
        parser.add_argument('--early_stopping_patience', type=int, default=10, help='早停止耐心值')
        # 模型参数
        parser.add_argument('--embedding_size', type=int, default=4, help='嵌入维度')
        parser.add_argument('--hidden_channels', type=int, default=4, help='隐藏层维度')
        parser.add_argument('--diffusion_embedding_size', type=int, default=64, help='扩散嵌入维度')
        parser.add_argument('--side_channels', type=int, default=1, help='辅助通道数')
        # 任务参数
        parser.add_argument('--in_len', type=int, default=24, help='输入序列长度')
        parser.add_argument('--out_len', type=int, default=24, help='输出序列长度')
        parser.add_argument('--missing_ratio', type=float, default=0.1, help='缺失率')
        # 扩散参数
        parser.add_argument('--beta_start', type=float, default=0.0001, help='beta起始值')
        parser.add_argument('--beta_end', type=float, default=0.2, help='beta结束值')
        parser.add_argument('--num_steps', type=int, default=50, help='扩散步数')
        parser.add_argument('--num_samples', type=int, default=1, help='插补样本数')
        parser.add_argument('--schedule', type=str, default='quad', help='扩散调度')
        parser.add_argument('--target_strategy', type=str, default='random', help='掩码策略')
        # 注意力参数
        parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
        # 生成插补数据参数
        parser.add_argument('--generate_imputation', action='store_true', help='训练后生成插补数据')
        parser.add_argument('--num_generate_samples', type=int, default=10, help='生成样本数')
        # SwanLab 配置
        parser.add_argument('--swanlab_project', type=str, default='STIMP-Bohai-Fixed', help='SwanLab项目名称')
        parser.add_argument('--swanlab_experiment', type=str, default=None, help='SwanLab实验名称')

        self.args = parser.parse_args()

        # 设置实验名称
        if self.args.swanlab_experiment is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            self.args.swanlab_experiment = f"{self.args.area}_missing_{self.args.missing_ratio}_{timestamp}"

        # 设置空间维度
        if self.args.area == "MEXICO":
            self.args.height, self.args.width = 36, 120
        elif self.args.area == "PRE":
            self.args.height, self.args.width = 60, 96
        elif self.args.area == "Chesapeake":
            self.args.height, self.args.width = 60, 48
        elif self.args.area == "Yangtze":
            self.args.height, self.args.width = 96, 72
        elif self.args.area in ["Himawari", "Bohai"]:
            self.args.height, self.args.width = 128, 128
        else:
            raise ValueError(f"未支持的区域: {self.args.area}")

    def __getattr__(self, name):
        return getattr(self.args, name)


def setup_swanlab(config):
    """设置 SwanLab 实验跟踪"""
    if not SWANLAB_AVAILABLE:
        return None

    # SwanLab 配置
    swanlab_config = {
        # 训练参数
        "area": config.area,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "weight_decay": config.wd,
        "test_frequency": config.test_freq,
        "early_stopping_patience": config.early_stopping_patience,
        # 模型参数
        "embedding_size": config.embedding_size,
        "hidden_channels": config.hidden_channels,
        "diffusion_embedding_size": config.diffusion_embedding_size,
        # 任务参数
        "input_length": config.in_len,
        "output_length": config.out_len,
        "missing_ratio": config.missing_ratio,
        # 扩散参数
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "num_steps": config.num_steps,
        "num_samples": config.num_samples,
        "schedule": config.schedule,
    }

    try:
        # 初始化 SwanLab
        run = swanlab.init(
            project=config.swanlab_project,
            experiment_name=config.swanlab_experiment,
            config=swanlab_config,
            description=f"STIMP model training with FIXED data for {config.area} with missing ratio {config.missing_ratio}"
        )

        print(f"🔬 SwanLab 实验跟踪已启动")
        return run
    except Exception as e:
        print(f"⚠️ SwanLab 初始化失败: {e}")
        print("将继续训练，但不进行实验跟踪")
        return None


def calculate_metrics_fixed(imputed, original, mask):
    """计算多种评估指标 - 修复版本"""
    imputed_flat = imputed.cpu().squeeze()
    original_flat = original.cpu().squeeze()
    mask_flat = mask.cpu().squeeze()

    # 调试信息
    print(f"调试信息 - 输入形状: imputed{imputed_flat.shape}, original{original_flat.shape}, mask{mask_flat.shape}")
    print(f"调试信息 - 数据范围: imputed[{imputed_flat.min():.4f}, {imputed_flat.max():.4f}], "
          f"original[{original_flat.min():.4f}, {original_flat.max():.4f}]")

    # 只计算缺失位置 (mask=0表示缺失，1表示观测)
    valid_mask = (1 - mask_flat) > 0.5



    if valid_mask.sum() == 0:
        print("⚠️ 警告：没有找到需要插补的缺失位置")
        # 如果没有缺失位置，计算所有位置的指标作为参考
        valid_mask = torch.ones_like(mask_flat).bool()

    mae = masked_mae(imputed_flat, original_flat, valid_mask)
    mse = masked_mse(imputed_flat, original_flat, valid_mask)
    rmse = torch.sqrt(mse)

    print(f"调试信息 - 有效位置数: {valid_mask.sum()}, MAE: {mae:.4f}")

    return mae, mse, rmse


def validate_data_before_training(config):
    """在训练前验证数据质量"""
    print("🔍 训练前数据验证...")

    try:
        # 加载数据集进行验证
        os.environ['RAW_DATA_PATH'] = config.raw_data_path
        train_dataset = PRE8dDataset(config.args, mode='train')

        # 检查一个样本
        sample = train_dataset[0]
        input_seq, input_ob_mask, input_gt_mask, output_seq, output_ob_mask = sample

        print("样本数据检查:")
        print(f"  输入序列形状: {input_seq.shape}")
        print(f"  输入序列范围: [{input_seq.min():.4f}, {input_seq.max():.4f}]")
        print(f"  观测掩码中1的比例: {input_ob_mask.mean():.4f}")
        print(f"  真实掩码中1的比例: {input_gt_mask.mean():.4f}")

        # 检查是否有异常值
        if input_seq.max() > 100 or input_seq.min() < -100:
            print("❌ 警告: 输入数据范围异常，可能存在归一化问题")
            return False
        else:
            print("✅ 数据范围正常")
            return True

    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
        return False


def generate_imputation_data(config, model, test_loader, adj, device):
    """生成插补数据"""
    print("🚀 开始生成插补数据...")
    model.eval()

    all_imputed_data = []
    all_original_data = []
    all_masks = []
    all_input_ob_masks = []

    with torch.no_grad():
        for step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(tqdm(test_loader)):
            # 数据移至设备
            datas = datas.float().to(device)
            data_ob_masks = data_ob_masks.float().to(device)
            data_gt_masks = data_gt_masks.float().to(device)
            adj_batch = adj.repeat(datas.shape[0], 1, 1).to(device)

            # 生成插补数据
            imputed = model.impute(datas, data_gt_masks, adj_batch, config.num_generate_samples)

            # 取中位数作为最终结果
            imputed_median = imputed.median(dim=1).values

            # 保存结果
            all_imputed_data.append(imputed_median.cpu().numpy())
            all_original_data.append(datas.cpu().numpy())
            all_masks.append(data_gt_masks.cpu().numpy())
            all_input_ob_masks.append(data_ob_masks.cpu().numpy())

            # 限制处理批次数量（可选）
            if step >= 50:  # 处理前50个批次
                break

    # 合并所有批次
    imputed_data = np.concatenate(all_imputed_data, axis=0)
    original_data = np.concatenate(all_original_data, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    input_ob_masks = np.concatenate(all_input_ob_masks, axis=0)

    # 保存结果
    output_dir = f"./imputation_results/{config.area}"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_path = os.path.join(
        output_dir,
        f"stimp_imputation_{config.area}_missing_{config.missing_ratio}_{timestamp}.npz"
    )

    np.savez_compressed(
        output_path,
        imputed_data=imputed_data,
        original_data=original_data,
        masks=masks,
        input_ob_masks=input_ob_masks,
        config=vars(config)
    )

    print(f"✅ 插补数据已保存到: {output_path}")
    print(f"📊 数据形状:")
    print(f"  - 插补数据: {imputed_data.shape}")
    print(f"  - 原始数据: {original_data.shape}")
    print(f"  - 掩码: {masks.shape}")
    print(f"  - 输入观测掩码: {input_ob_masks.shape}")

    return imputed_data, original_data, masks


def main():
    # 清理内存
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = TrainConfig()
    print("训练配置（修复版本）:")
    for arg, value in vars(config.args).items():
        print(f"  {arg}: {value}")

    # 设置 SwanLab
    swanlab_run = setup_swanlab(config)

    # 环境配置
    base_dir = f"./tmp/imputation/{config.in_len}/{config.area}/STIMP-fixed/"
    check_dir(base_dir)
    seed_everything(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 日志配置
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_path = os.path.join(base_dir, f'{timestamp}_missing_{config.missing_ratio}.log')
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(message)s',
        encoding='utf-8'
    )

    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info("STIMP Training Configuration (FIXED VERSION):")
    for arg, value in vars(config.args).items():
        logging.info(f"  {arg}: {value}")

    # 训练前数据验证
    if not validate_data_before_training(config):
        print("❌ 数据验证失败，停止训练")
        return

    # 加载数据集
    os.environ['RAW_DATA_PATH'] = config.raw_data_path
    train_dataset = PRE8dDataset(config.args, mode='train')
    val_dataset = PRE8dDataset(config.args, mode='val')
    test_dataset = PRE8dDataset(config.args, mode='test')

    # 记录数据统计信息
    data_stats = {
        "data/total_samples": len(train_dataset.all_samples),
        "data/train_samples": len(train_dataset.samples),
        "data/val_samples": len(val_dataset.samples),
        "data/test_samples": len(test_dataset.samples),
        "data/nodes": train_dataset.total_nodes,
    }

    if hasattr(train_dataset, 'missing_rate'):
        data_stats["data/missing_rate"] = train_dataset.missing_rate

    logging.info("📊 数据统计信息:")
    for key, value in data_stats.items():
        logging.info(f"  {key}: {value}")

    if swanlab_run:
        swanlab.log(data_stats)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 加载邻接矩阵
    if config.area in ["Himawari", "Bohai"]:
        adj_path = os.path.join(config.raw_data_path, "adj.npy")
    else:
        adj_path = f"./data/{config.area}/adj.npy"

    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"邻接矩阵文件不存在: {adj_path}")

    adj = np.load(adj_path).astype(np.float32)
    adj = torch.from_numpy(adj).float().to(device)

    # 验证邻接矩阵形状
    expected_nodes = 4443
    if adj.shape[0] != expected_nodes or adj.shape[1] != expected_nodes:
        print(f"⚠️ 警告: 邻接矩阵形状 {adj.shape} 与预期节点数 {expected_nodes} 不匹配")

    print(f"✅ 加载邻接矩阵: {adj.shape}")

    # 加载统计量
    low_bound = torch.from_numpy(train_dataset.mean).float().to(device)
    high_bound = torch.from_numpy(train_dataset.std).float().to(device)

    # 初始化模型
    from model.graphdiffusion import IAP_base
    model = IAP_base(config.args, low_bound, high_bound).to(device)

    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    p1 = int(0.75 * config.epochs)
    p2 = int(0.9 * config.epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 初始化早停止
    early_stopping_path = os.path.join(base_dir, f"early_stopping_best_{config.missing_ratio}.pth")
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        verbose=True,
        delta=0.001,
        path=early_stopping_path
    )

    print(f"🎯 早停止已启用，耐心值: {config.early_stopping_patience}")

    # 训练循环
    best_mae = float('inf')
    best_mse = float('inf')
    train_pbar = tqdm(range(1, config.epochs + 1), desc="训练进度")

    for epoch in train_pbar:
        model.train()
        loss_meter = AverageMeter()
        data_time_meter = AverageMeter()
        end = time.time()

        for step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_loader):
            # 数据移至设备
            datas = datas.float().to(device)
            data_ob_masks = data_ob_masks.float().to(device)
            data_gt_masks = data_gt_masks.float().to(device)
            adj_batch = adj.repeat(datas.shape[0], 1, 1).to(device)

            # 混合精度训练
            with torch.cuda.amp.autocast():
                loss = model.trainstep(datas, data_ob_masks, adj_batch, 1)

            # 更新指标
            loss_meter.update(loss.item(), datas.shape[0])
            data_time_meter.update(time.time() - end)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 清理内存
            torch.cuda.empty_cache()
            end = time.time()

        # 学习率调度
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 记录训练指标
        train_metrics = {
            "train/loss": loss_meter.avg,
            "train/data_time": data_time_meter.avg,
            "train/learning_rate": current_lr,
        }

        # 日志输出
        log_msg = f"Epoch {epoch} | 训练损失: {loss_meter.avg:.4f} | 耗时: {data_time_meter.avg:.2f}s | LR: {current_lr:.6f}"
        print(log_msg)
        logging.info(log_msg)
        train_pbar.set_description(f"Epoch {epoch} | 损失: {loss_meter.avg:.4f}")

        # 上传指标到 SwanLab
        if swanlab_run:
            swanlab.log(train_metrics, step=epoch)

        # 验证集和测试集评估（用于早停止和监控）
        if epoch % config.test_freq == 0 or epoch == config.epochs:
            model.eval()
            val_mae_list, val_mse_list, val_rmse_list = [], [], []
            test_mae_list, test_mse_list, test_rmse_list = [], [], []

            # 1. 验证集评估（用于早停止）
            with torch.no_grad():
                for step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(val_loader):
                    if step >= 5:  # 限制验证样本数量以加快速度
                        break

                    # 数据移至设备
                    datas = datas.float().to(device)
                    data_ob_masks = data_ob_masks.float().to(device)
                    data_gt_masks = data_gt_masks.float().to(device)
                    adj_batch = adj.repeat(datas.shape[0], 1, 1).to(device)

                    # 插补
                    imputed = model.impute(datas, data_gt_masks, adj_batch, config.num_samples)

                    # 计算多种指标
                    mae, mse, rmse = calculate_metrics_fixed(imputed, datas, data_gt_masks)
                    val_mae_list.append(mae)
                    val_mse_list.append(mse)
                    val_rmse_list.append(rmse)

            # 2. 测试集评估（仅用于监控，不用于早停止）
            with torch.no_grad():
                for step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_loader):
                    if step >= 5:  # 限制测试样本数量以加快速度
                        break

                    # 数据移至设备
                    datas = datas.float().to(device)
                    data_ob_masks = data_ob_masks.float().to(device)
                    data_gt_masks = data_gt_masks.float().to(device)
                    adj_batch = adj.repeat(datas.shape[0], 1, 1).to(device)

                    # 插补
                    imputed = model.impute(datas, data_gt_masks, adj_batch, config.num_samples)

                    # 计算多种指标
                    mae, mse, rmse = calculate_metrics_fixed(imputed, datas, data_gt_masks)
                    test_mae_list.append(mae)
                    test_mse_list.append(mse)
                    test_rmse_list.append(rmse)

            # 平均验证集指标
            if val_mae_list:
                avg_val_mae = torch.stack(val_mae_list).mean().item()
                avg_val_mse = torch.stack(val_mse_list).mean().item()
                avg_val_rmse = torch.stack(val_rmse_list).mean().item()

                val_metrics = {
                    "val/mae": avg_val_mae,
                    "val/mse": avg_val_mse,
                    "val/rmse": avg_val_rmse,
                }

                val_msg = f"验证 | MAE: {avg_val_mae:.4f} | MSE: {avg_val_mse:.4f} | RMSE: {avg_val_rmse:.4f}"
                print(val_msg)
                logging.info(val_msg)

            # 平均测试集指标
            if test_mae_list:
                avg_test_mae = torch.stack(test_mae_list).mean().item()
                avg_test_mse = torch.stack(test_mse_list).mean().item()
                avg_test_rmse = torch.stack(test_rmse_list).mean().item()

                test_metrics = {
                    "test/mae": avg_test_mae,
                    "test/mse": avg_test_mse,
                    "test/rmse": avg_test_rmse,
                }

                test_msg = f"测试 | MAE: {avg_test_mae:.4f} | MSE: {avg_test_mse:.4f} | RMSE: {avg_test_rmse:.4f}"
                print(test_msg)
                logging.info(test_msg)

            # 上传指标到 SwanLab
            if swanlab_run:
                swanlab.log(val_metrics, step=epoch)
                swanlab.log(test_metrics, step=epoch)

            # 保存最佳模型（基于验证集指标）
            if val_mae_list and avg_val_mae < best_mae:
                best_mae = avg_val_mae
                best_mse = avg_val_mse
                best_model_path = os.path.join(base_dir, f"best_{config.missing_ratio}.pth")
                torch.save(model.state_dict(), best_model_path)
                best_msg = f" 保存最佳模型 | 验证MAE: {best_mae:.4f} | 验证MSE: {best_mse:.4f}"
                print(best_msg)
                logging.info(best_msg)

                if swanlab_run:
                    swanlab.log({"best/val_mae": best_mae, "best/val_mse": best_mse}, step=epoch)

            # 早停止检查 - 使用验证集MAE作为早停止指标
            if val_mae_list:
                early_stopping(avg_val_mae, model)

                if early_stopping.early_stop:
                    print(f"🛑 早停止触发！在 epoch {epoch} 停止训练")
                    logging.info(f"早停止触发于 epoch {epoch}")
                    break

    # 训练完成
    final_msg = f"训练结束 | 最佳验证MAE: {best_mae:.4f} | 最佳验证MSE: {best_mse:.4f}"
    print(final_msg)
    logging.info(final_msg)

    # 加载早停止保存的最佳模型
    if os.path.exists(early_stopping_path):
        print(f"📥 加载早停止保存的最佳模型: {early_stopping_path}")
        model.load_state_dict(torch.load(early_stopping_path))

    # 最终测试集评估
    print("🧪 开始最终测试集评估...")
    model.eval()
    test_mae_list = []
    test_mse_list = []
    test_rmse_list = []

    with torch.no_grad():
        for step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_loader):
            if step >= 10:
                break

            # 数据移至设备
            datas = datas.float().to(device)
            data_ob_masks = data_ob_masks.float().to(device)
            data_gt_masks = data_gt_masks.float().to(device)
            adj_batch = adj.repeat(datas.shape[0], 1, 1).to(device)

            # 插补
            imputed = model.impute(datas, data_gt_masks, adj_batch, config.num_samples)

            # 计算多种指标
            mae, mse, rmse = calculate_metrics_fixed(imputed, datas, data_gt_masks)
            test_mae_list.append(mae)
            test_mse_list.append(mse)
            test_rmse_list.append(rmse)

    # 计算最终测试指标
    if test_mae_list:
        final_test_mae = torch.stack(test_mae_list).mean().item()
        final_test_mse = torch.stack(test_mse_list).mean().item()
        final_test_rmse = torch.stack(test_rmse_list).mean().item()

        final_test_msg = f"最终测试 | MAE: {final_test_mae:.4f} | MSE: {final_test_mse:.4f} | RMSE: {final_test_rmse:.4f}"
        print(final_test_msg)
        logging.info(final_test_msg)

        if swanlab_run:
            swanlab.log({
                "final_test/mae": final_test_mae,
                "final_test/mse": final_test_mse,
                "final_test/rmse": final_test_rmse
            })

    # 生成插补数据（如果启用）
    if config.generate_imputation:
        print("🎯 开始生成插补数据...")
        imputed_data, original_data, masks = generate_imputation_data(config.args, model, test_loader, adj, device)

        # 记录生成结果
        if swanlab_run:
            swanlab.log({
                "generation/samples_generated": imputed_data.shape[0],
                "generation/timestamp": timestamp
            })

    if swanlab_run:
        swanlab.log({"final/best_mae": best_mae, "final/best_mse": best_mse})
        swanlab.finish()

    return best_mae, best_mse


if __name__ == '__main__':
    main()