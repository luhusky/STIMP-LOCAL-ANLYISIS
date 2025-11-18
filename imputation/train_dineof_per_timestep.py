import numpy as np
from scipy import linalg


class DINEOF:
    def __init__(self, R, tensor_shape, mask=None, nitemax=300, toliter=1e-5):
        self.R = R  # 矩阵秩（可调整，影响插值效果）
        self.tensor_shape = tensor_shape  # (height, width, time)
        self.nitemax = nitemax  # 最大迭代次数
        self.toliter = toliter  # 收敛阈值

        # 适配Himawari掩码（2D→3D）
        if mask is not None:
            self.mask = np.load(mask).astype(bool)  # 加载is_sea.npy（2D）
            # 扩展到时间维度：[height, width] → [height, width, time]
            if self.mask.ndim == 2:
                self.mask = np.tile(self.mask[..., np.newaxis], (1, 1, tensor_shape[-1]))
        else:
            self.mask = np.ones(tensor_shape, dtype=bool)

        assert self.mask.shape == tensor_shape, f"掩码维度不匹配: {self.mask.shape} vs {tensor_shape}"
        self.reconstructed_tensor = None
        self.rel_error = None

    def fit(self, X):
        """
        X: 输入数据，形状为(height, width, time)
        输出：插值后的张量（同形状）
        """
        n, m, t = self.tensor_shape
        X_flat = X.reshape(n * m, t)  # 展平空间维度 [n*m, time]
        mask_flat = self.mask.reshape(n * m, t)  # 掩码同步展平

        # 初始化：缺失值填0
        Xa = X_flat.copy()
        Xa[~mask_flat] = 0.0
        # 初始SVD分解
        U, s, Vh = linalg.svd(Xa, full_matrices=False)
        U = U[:, :self.R]
        s = s[:self.R]
        Vh = Vh[:self.R, :]
        X_hat = U @ np.diag(s) @ Vh

        # 迭代优化
        for i in range(self.nitemax):
            # 已知值保留，缺失值用预测填充
            Xa[mask_flat] = X_flat[mask_flat]
            Xa[~mask_flat] = X_hat[~mask_flat]

            # 更新SVD
            U, s, Vh = linalg.svd(Xa, full_matrices=False)
            U = U[:, :self.R]
            s = s[:self.R]
            Vh = Vh[:self.R, :]
            X_hat_new = U @ np.diag(s) @ Vh

            # 检查收敛
            error = linalg.norm(X_hat_new - X_hat) / linalg.norm(X_hat)
            if error < self.toliter:
                break
            X_hat = X_hat_new

        self.rel_error = error
        self.reconstructed_tensor = X_hat.reshape(n, m, t)  # 恢复3D形状
        print(f"DINEOF迭代结束：{i + 1}次迭代，相对误差{error:.6f}")
