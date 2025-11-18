import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from local_attention import LocalAttention

class GCN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class DiffusionEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class IAP_base(nn.Module):
    def __init__(self, config, low_bound, high_bound):
        super(IAP_base, self).__init__()
        self.config = config
        self.low_bound = low_bound.float()
        self.high_bound = high_bound.float()
        self.diffusion_model = SpatialTemporalEncoding(
            config=config,
            low_bound=self.low_bound,
            high_bound=self.high_bound
        ).float()
        self.float()

    def forward(self, x, mask, adj):
        return self.diffusion_model(x.float(), mask.float(), adj.float())

    def trainstep(self, x, mask, adj, is_train=1):
        return self.diffusion_model.trainstep(x.float(), mask.float(), adj.float(), is_train)

    def impute(self, x, mask, adj, num_samples=10):
        return self.diffusion_model.impute(x.float(), mask.float(), adj.float(), num_samples)


class SpatialTemporalEncoding(nn.Module):
    def __init__(self, config, low_bound, high_bound):
        super(SpatialTemporalEncoding, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.low_bound = low_bound.to(self.device).float()  # [N]
        self.high_bound = high_bound.to(self.device).float()  # [N]

        # 加载海洋掩码和节点信息 - 支持渤海数据
        if config.area in ["Himawari", "Bohai"]:
            import os
            sea_mask_path = os.path.join(os.environ.get('RAW_DATA_PATH', f'./data/{config.area}/'), 'is_sea.npy')
            node_info_path = os.path.join(os.environ.get('RAW_DATA_PATH', f'./data/{config.area}/'), 'node_info.npy')
        else:
            sea_mask_path = f'./data/{config.area}/is_sea.npy'
            node_info_path = None

        # 渤海数据特殊处理
        if config.area in ["Himawari", "Bohai"] and os.path.exists(node_info_path):
            print("🌊 加载渤海节点信息...")
            node_info = np.load(node_info_path, allow_pickle=True).item()
            self.n_sea = node_info['total_nodes']  # 4443
            self.node_indices = node_info['node_indices']  # 节点坐标

            # 创建海洋掩码（基于节点信息）
            sea_mask_2d = np.load(sea_mask_path)
            sea_mask_values = []
            for row, col in self.node_indices:
                if 0 <= row < sea_mask_2d.shape[0] and 0 <= col < sea_mask_2d.shape[1]:
                    sea_mask_values.append(sea_mask_2d[row, col])
                else:
                    sea_mask_values.append(0.0)

            self.is_sea = np.array(sea_mask_values) > 0.5
            self.sea_indices = torch.from_numpy(self.is_sea).bool().to(self.device)
        else:
            # 原始逻辑
            self.is_sea = np.load(sea_mask_path).astype(bool)
            self.sea_indices = torch.from_numpy(self.is_sea).bool().to(self.device)
            self.n_sea = self.sea_indices.sum().item()

        print(f" 区域 {config.area}: 使用 {self.n_sea} 个节点")

        # 模型参数
        self.embedding_size = config.embedding_size
        self.hidden_channels = config.hidden_channels
        self.in_len = config.in_len
        self.out_len = config.out_len
        self.C = 1

        # 时间嵌入层
        self.time_embedding = nn.Embedding(self.in_len, self.embedding_size).to(self.device).float()

        # 空间位置嵌入层
        self.position_embedding = nn.Sequential(
            nn.Linear(3, self.embedding_size).to(self.device).float(),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size).to(self.device).float()
        ).to(self.device).float()

        # 扩散过程参数
        self.num_steps = config.num_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.schedule = config.schedule
        self.betas = self.get_betas().float().to(self.device)
        self.alphas = (1. - self.betas).float().to(self.device)
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).float().to(self.device)

        # 核心配置：根据节点数量调整窗口大小
        self.spatial_window_size = 64
        self.target_seq_len = ((
                                       self.n_sea + self.spatial_window_size - 1) // self.spatial_window_size) * self.spatial_window_size
        self.pad_len = self.target_seq_len - self.n_sea
        print(f" 序列填充：原始长度{self.n_sea} → 目标长度{self.target_seq_len}（填充{self.pad_len}个0）")

        # 注意力层
        self.spatial_attention = LocalAttention(
            window_size=self.spatial_window_size,
            dim=self.embedding_size
        ).to(self.device).float()
        self.temporal_attention = LocalAttention(
            window_size=2,
            dim=self.embedding_size
        ).to(self.device).float()

        # 解码层
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_channels).to(self.device).float(),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, 1).to(self.device).float()
        ).to(self.device).float()

        self.apply(self._init_weights)
        self.float().to(self.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, LocalAttention):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_betas(self):
        if self.schedule == 'quad':
            betas = torch.linspace(
                self.beta_start ** 0.5,
                self.beta_end ** 0.5,
                self.num_steps,
                device=self.device
            ) ** 2
        elif self.schedule == 'linear':
            betas = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_steps,
                device=self.device
            )
        else:
            raise ValueError(f"未知的扩散调度: {self.schedule}")
        return betas

    def get_position_embeding(self):
        """位置编码 - 适配渤海数据"""
        # 渤海数据：使用节点坐标创建位置编码
        if self.config.area in ["Himawari", "Bohai"] and hasattr(self, 'node_indices'):
            # 从节点坐标创建位置信息
            rows = np.array([idx[0] for idx in self.node_indices])  # 行坐标
            cols = np.array([idx[1] for idx in self.node_indices])  # 列坐标

            # 归一化坐标到 [0, 1]
            pos_h = rows / (self.config.height - 1) if self.config.height > 1 else rows
            pos_w = cols / (self.config.width - 1) if self.config.width > 1 else cols

            # 转换为tensor并移动到设备
            pos_h_tensor = torch.from_numpy(pos_h).float().to(self.device)
            pos_w_tensor = torch.from_numpy(pos_w).float().to(self.device)

            # 给位置编码填充0，匹配目标长度
            pos_h_padded = F.pad(pos_h_tensor, (0, self.pad_len), mode='constant', value=0.0)
            pos_w_padded = F.pad(pos_w_tensor, (0, self.pad_len), mode='constant', value=0.0)

            # 给mean/std填充0
            mean_padded = F.pad(self.low_bound, (0, self.pad_len), mode='constant', value=0.0)
            std_padded = F.pad(self.high_bound, (0, self.pad_len), mode='constant', value=0.0)

            pe = torch.stack([mean_padded, std_padded, pos_h_padded], dim=0).float()
        else:
            # 原始逻辑
            pos_h = torch.arange(self.config.height, device=self.device).float()
            pos_h = pos_h[:, None].repeat(1, self.config.width)
            pos_h_sea = pos_h[self.sea_indices.cpu()]  # 注意设备转换

            pos_h_padded = F.pad(pos_h_sea.to(self.device), (0, self.pad_len), mode='constant', value=0.0)
            mean_padded = F.pad(self.low_bound, (0, self.pad_len), mode='constant', value=0.0)
            std_padded = F.pad(self.high_bound, (0, self.pad_len), mode='constant', value=0.0)

            pe = torch.stack([mean_padded, std_padded, pos_h_padded], dim=0).float()

        return pe

    def spatial_temporal_encoding(self, x, mask):
        B, T, C, N, _ = x.shape  # N=节点数
        E = self.embedding_size

        # 强制通道数为1
        if C != 1:
            x = x[:, :, :1, :, :]
            C = 1

        # 确保输入张量在正确的设备上
        x = x.float().to(self.device)
        mask = mask.float().to(self.device)

        # 步骤1：调整原始数据形状并填充 → [B, T, target_seq_len, C=1]
        x_flat = rearrange(x.float(), 'b t c n 1 -> b t n c').float()  # [B, T, N, 1]
        x_padded = F.pad(x_flat, (0, 0, 0, self.pad_len), mode='constant', value=0.0).to(
            self.device)  # [B, T, target_seq_len, 1]

        # 步骤2：掩码填充（填充部分掩码设为0，忽略其影响）- 确保在相同设备上
        mask_flat = rearrange(mask.float(), 'b t c n 1 -> b t n c').float()  # [B, T, N, 1]
        mask_padded = F.pad(mask_flat, (0, 0, 0, self.pad_len), mode='constant', value=0.0).to(
            self.device)  # [B, T, target_seq_len, 1]

        # 步骤3：时间嵌入
        time_ids = torch.arange(T, device=self.device).long()
        time_emb = self.time_embedding(time_ids).float()  # [T, E]
        time_emb = time_emb[None, :, None, :].repeat(B, 1, self.target_seq_len, 1).float()  # [B, T, target_seq_len, E]

        # 步骤4：空间位置嵌入（已填充）
        pos_emb = self.get_position_embeding().float()  # [3, target_seq_len]
        pos_emb = rearrange(pos_emb, 'c n -> n c').float()  # [target_seq_len, 3]
        pos_emb = self.position_embedding(pos_emb).float()  # [target_seq_len, E]
        pos_emb = pos_emb[None, None, :, :].repeat(B, T, 1, 1).float()  # [B, T, target_seq_len, E]

        # 步骤5：融合特征（填充后维度匹配）
        x_data = x_padded.repeat(1, 1, 1, E // 2).float().to(self.device)  # [B, T, target_seq_len, E//2]
        time_feature = time_emb[..., :E // 4].float()  # [B, T, target_seq_len, E//4]
        pos_feature = pos_emb[..., :E // 4].float()  # [B, T, target_seq_len, E//4]
        x_emb = torch.cat([x_data, time_feature, pos_feature], dim=-1).float()  # [B, T, target_seq_len, E]

        # 步骤6：空间注意力（填充后长度可整除窗口大小）
        x_spatial = rearrange(x_emb, 'b t n e -> t (b n) e').float()  # [T, B*target_seq_len, E]
        spatial_attn_out = self.spatial_attention(x_spatial, x_spatial, x_spatial).float()
        spatial_attn_out = rearrange(spatial_attn_out, 't (b n) e -> b t n e', b=B,
                                     n=self.target_seq_len).float()  # [B, T, target_seq_len, E]

        # 确保 mask_padded 在相同设备上
        mask_padded_device = mask_padded[..., 0:1].to(self.device)
        spatial_attn_out = (spatial_attn_out * mask_padded_device).float()  # 忽略填充部分

        # 步骤7：裁剪回原始长度（去掉填充的部分）
        spatial_attn_out = spatial_attn_out[:, :, :self.n_sea, :]  # [B, T, N, E]

        # 步骤8：时间注意力（原始长度，不影响）
        x_temporal = rearrange(spatial_attn_out, 'b t n e -> (b n) t e').float()  # [B*N, T, E]
        temporal_attn_out = self.temporal_attention(x_temporal, x_temporal, x_temporal).float()
        temporal_attn_out = rearrange(temporal_attn_out, '(b n) t e -> b t n e', b=B,
                                      n=self.n_sea).float()  # [B, T, N, E]

        # 步骤9：解码输出（恢复原始形状）
        out = self.decoder(temporal_attn_out).float()  # [B, T, N, 1]
        out = rearrange(out, 'b t n c -> b t c n 1').float()  # [B, T, 1, N, 1]

        return out

    def forward_diffusion(self, x, t):
        x = x.float().clone().to(self.device)
        eps = torch.randn_like(x, device=self.device).float()
        alpha_hat = self.alpha_hats[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        x_t = (torch.sqrt(alpha_hat) * x + torch.sqrt(1. - alpha_hat) * eps).float()
        return x_t, eps

    def reverse_diffusion(self, x_t, t, mask, adj):
        x_t = x_t.float().clone().to(self.device)
        mask = mask.float().clone().to(self.device)
        adj = adj.float().clone().to(self.device)
        x_emb = self.spatial_temporal_encoding(x_t, mask).float()
        eps_pred = x_emb.float()
        return eps_pred

    def trainstep(self, x, mask, adj, is_train=1):
        x = x.float().clone().to(self.device)
        mask = mask.float().clone().to(self.device)
        adj = adj.float().clone().to(self.device)

        B = x.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        x_t, eps = self.forward_diffusion(x, t)
        eps_pred = self.reverse_diffusion(x_t, t, mask, adj)

        loss = F.mse_loss(eps_pred.float() * mask.float(), eps.float() * mask.float())
        return loss.float()

    def impute(self, x, mask, adj, num_samples=10):
        x = x.float().clone().to(self.device)
        mask = mask.float().clone().to(self.device)
        adj = adj.float().clone().to(self.device)

        B, T, C, N, _ = x.shape
        x_t = torch.randn_like(x, device=self.device).float()
        samples = []

        for _ in range(num_samples):
            x_sample = x_t.clone().float()
            for t in reversed(range(self.num_steps)):
                eps_pred = self.reverse_diffusion(
                    x_sample.float(),
                    torch.tensor([t] * B, device=self.device).long(),
                    mask.float(),
                    adj.float()
                )
                alpha = self.alphas[t].float()
                alpha_hat = self.alpha_hats[t].float()
                beta = self.betas[t].float()

                mean = (1. / torch.sqrt(alpha) * (
                        x_sample - (1. - alpha) / torch.sqrt(1. - alpha_hat) * eps_pred
                )).float()

                if t > 0:
                    var = ((1. - self.alpha_hats[t - 1].float()) / (1. - alpha_hat) * beta).float()
                else:
                    var = torch.zeros_like(mean).float()

                if t > 0:
                    z = torch.randn_like(mean).float()
                    x_sample = (mean + torch.sqrt(var) * z).float()
                else:
                    x_sample = mean.float()

            x_imputed = (x.float() * mask.float() + x_sample.float() * (1. - mask.float())).float()
            samples.append(x_imputed)

        samples = torch.stack(samples, dim=1).float()
        median_sample = samples.median(dim=1).values.float()
        return median_sample

    def forward(self, x, mask, adj):
        # 设备一致性检查
        if x.device != self.device:
            x = x.to(self.device)
        if mask.device != self.device:
            mask = mask.to(self.device)
        if adj.device != self.device:
            adj = adj.to(self.device)

        return self.impute(x, mask, adj)