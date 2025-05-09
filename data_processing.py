import numpy as np


def compute_lda(X1, X2):
    """
    根据两类样本数据计算LDA投影向量。
    X1, X2: shape (n_samples, n_features) or empty if no samples
    """
    d = 2  # 默认维度，如果数据存在则会被覆盖

    has_X1_data = X1.ndim == 2 and X1.shape[0] > 0
    has_X2_data = X2.ndim == 2 and X2.shape[0] > 0

    if has_X1_data:
        n1 = X1.shape[0]
        d = X1.shape[1]
    else:
        n1 = 0

    if has_X2_data:
        n2 = X2.shape[0]
        if not has_X1_data:  # 如果 X1 为空，则从 X2 获取维度
            d = X2.shape[1]
        elif X1.ndim == 2 and d != X2.shape[1]:  # 如果都有数据但维度不匹配
            raise ValueError("X1 和 X2 的维度不匹配")
    else:
        n2 = 0

    # 为存在的类计算实际均值，空类则为零向量
    mu1_hat = np.mean(X1, axis=0) if has_X1_data else np.zeros(d)
    mu2_hat = np.mean(X2, axis=0) if has_X2_data else np.zeros(d)

    # 初始化 Sw 和 Sb
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))

    # 如果任一类别没有数据点，或者只有一个类别有数据点
    if not has_X1_data or not has_X2_data:
        if has_X1_data:  # 只有类别1
            centered_X1 = X1 - mu1_hat
            if n1 > 0:
                Sw = (centered_X1.T @ centered_X1) / n1  # Sw 是 S1_hat
        elif has_X2_data:  # 只有类别2
            centered_X2 = X2 - mu2_hat
            if n2 > 0:
                Sw = (centered_X2.T @ centered_X2) / n2  # Sw 是 S2_hat
        # Sb 保持为零，w 也为零
        return np.zeros(d), mu1_hat, mu2_hat, Sw, Sb

    # --- 标准LDA计算 (n1 > 0 且 n2 > 0) ---
    n_total = n1 + n2

    # 计算 S1_hat, S2_hat
    centered_X1 = X1 - mu1_hat
    S1_hat = (centered_X1.T @ centered_X1) / n1

    centered_X2 = X2 - mu2_hat
    S2_hat = (centered_X2.T @ centered_X2) / n2

    # 计算 Sw
    Sw = (n1 / n_total) * S1_hat + (n2 / n_total) * S2_hat

    # 计算 Sb (根据幻灯片 Basic Concepts 定义: sum (nk/N) (muk - mu_total)(muk - mu_total)^T)
    mu_total = (n1 * mu1_hat + n2 * mu2_hat) / n_total
    diff1 = mu1_hat - mu_total
    diff2 = mu2_hat - mu_total
    Sb = (n1 / n_total) * np.outer(diff1, diff1) + \
        (n2 / n_total) * np.outer(diff2, diff2)

    # 计算投影向量 w (propto Sw^-1 * (mu1_hat - mu2_hat))
    try:
        # 检查 Sw 的条件数，以决定使用 inv 还是 pinv
        if np.linalg.cond(Sw) < 1 / np.finfo(Sw.dtype).eps:
            Sw_inv = np.linalg.inv(Sw)
        else:
            Sw_inv = np.linalg.pinv(Sw)  # 如果 Sw 奇异或接近奇异，使用伪逆
    except np.linalg.LinAlgError:
        Sw_inv = np.eye(d)  # 极端情况下，如果(伪)逆计算失败，使用单位矩阵

    w = Sw_inv @ (mu1_hat - mu2_hat)

    # 标准化 w
    norm_w = np.linalg.norm(w)
    if norm_w > 1e-6:  # 避免除以零
        w = w / norm_w
    else:  # 如果 w 是零向量 (例如均值相同或 Sw_inv 映射导致零向量)
        w = np.zeros(d)
        if d > 0:
            w[0] = 1.0  # 设置一个任意的默认方向以便可视化

    return w, mu1_hat, mu2_hat, Sw, Sb


def generate_data(n1, n2, mean1, mean2, cov1, cov2):
    """生成两类二维高斯分布数据"""
    X1 = np.array([])
    if n1 > 0:
        X1 = np.random.multivariate_normal(mean1, cov1, n1)

    X2 = np.array([])
    if n2 > 0:
        X2 = np.random.multivariate_normal(mean2, cov2, n2)

    return X1, X2
