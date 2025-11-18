import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys

# 强制添加项目根目录到路径（确保能导入utils.py）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_random_mask  # 从根目录utils.py导入


class HimawariHourlyDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.data_root = config.raw_data_path  # 预处理后.npy数据根目录
        self.in_len = config.in_len
        self.out_len = config.out_len
        self.missing_ratio = config.missing_ratio
        self.height, self.width = config.height, config.width
        self.n_nodes = self.height * self.width  # 总节点数=高×宽

        # 加载预处理后的小时数据
        self.load_preprocessed_data()
        # 数据边界（与预处理标准化一致）
        self.min = self.norm_params[0]
        self.max = self.norm_params[1]

    def load_preprocessed_data(self):
        print(f"[HimawariHourlyDataset] 加载{self.mode}集：{self.data_root}")
        # 加载样本和标签
        if self.mode == "train":
            self.samples = np.load(os.path.join(self.data_root, "train_samples.npy"))
            self.labels = np.load(os.path.join(self.data_root, "train_labels.npy"))
        else:  # test模式
            self.samples = np.load(os.path.join(self.data_root, "test_samples.npy"))
            self.labels = np.load(os.path.join(self.data_root, "test_labels.npy"))

        # 加载标准化参数（min, max）
        self.norm_params = np.load(os.path.join(self.data_root, "norm_params.npy"))

        # 校验序列长度（与训练脚本参数一致）
        assert self.samples.shape[
                   1] == self.in_len, f"输入序列长度不匹配：样本{self.samples.shape[1]} vs 配置{self.in_len}"
        assert self.labels.shape[
                   1] == self.out_len, f"输出序列长度不匹配：标签{self.labels.shape[1]} vs 配置{self.out_len}"

        # 校验空间尺寸（不匹配则下采样）
        sample_h, sample_w = self.samples.shape[2], self.samples.shape[3]
        if sample_h != self.height or sample_w != self.width:
            print(f"[HimawariHourlyDataset] 下采样空间尺寸：({sample_h},{sample_w}) → ({self.height},{self.width})")
            self.samples = self.downsample(self.samples, self.height, self.width)
            self.labels = self.downsample(self.labels, self.height, self.width)

        print(f"[HimawariHourlyDataset] {self.mode}集加载完成：样本数{len(self.samples)}，形状{self.samples.shape}")

    def downsample(self, data, target_h, target_w):
        """下采样空间维度（适配模型输入尺寸）"""
        import torch.nn.functional as F
        data_tensor = torch.from_numpy(data).float().unsqueeze(1)  # (N, 1, T, H, W)
        downsampled = F.interpolate(
            data_tensor,
            size=(data.shape[1], target_h, target_w),  # 保持时间维度，下采样空间
            mode="bilinear",
            align_corners=False
        )
        return downsampled.squeeze(1).numpy()  # 还原为(N, T, H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """生成模型要求的输出格式：(datas, data_ob_masks, data_gt_masks, labels, label_masks)"""
        # 1. 提取单样本（N=1, T, H, W）
        data = self.samples[idx:idx + 1]  # (1, in_len, H, W)
        label = self.labels[idx:idx + 1]  # (1, out_len, H, W)

        # 2. 转换为模型输入格式：(1, T, N_nodes, 1)（N_nodes=H*W）
        data = data.reshape(1, self.in_len, self.n_nodes, 1)
        label = label.reshape(1, self.out_len, self.n_nodes, 1)

        # 3. 生成掩码（陆地掩码+随机缺失掩码）
        # 陆地掩码：预处理时陆地填充为0，标记为未观测（1=未观测）
        land_mask = (data[0, :, :, 0] == 0).astype(np.float32)  # (in_len, N_nodes)
        land_mask = land_mask.reshape(1, self.in_len, self.n_nodes, 1)

        # 随机缺失掩码：按missing_ratio生成
        random_mask = create_random_mask(
            shape=(1, self.in_len, self.n_nodes, 1),
            missing_ratio=self.missing_ratio,
            target_strategy=self.config.target_strategy
        )

        # 合并掩码：data_ob_masks=陆地掩码∪随机缺失掩码（1=未观测）
        data_ob_masks = np.maximum(land_mask, random_mask.numpy())
        data_gt_masks = random_mask.numpy()  # 仅随机缺失部分需要插补（1=缺失）

        # 4. 标签掩码（全1，标签无缺失）
        label_masks = np.ones_like(label)

        # 转换为torch张量

        data = torch.from_numpy(data).float()
        data_ob_masks = torch.from_numpy(data_ob_masks).float()
        data_gt_masks = torch.from_numpy(data_gt_masks).float()
        label = torch.from_numpy(label).float()
        label_masks = torch.from_numpy(label_masks).float()

        return data, data_ob_masks, data_gt_masks, label, label_masks