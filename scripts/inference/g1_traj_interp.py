"""G1 轨迹流插值器 — 将低频（30Hz）行目标插值为高频（250Hz）SDK 指令帧。

移植自同事项目 /home/galbot/infer_policy/joint_interpolator.py（真机在役实现），
算法逐行保留：linear=np.interp / cubic=natural C2 三次样条（Thomas 三对角）/
quintic=分段五次 Hermite（中心差分估 v/a，Cramer 法则解系数）。
本仓库裁剪：去 Passthrough，工厂改显式类名构造；dim 默认 14（双臂）。

用法:
    from g1_traj_interp import create_interpolator
    interp = create_interpolator(method="cubic", dim=14, input_hz=30, output_hz=250)
    frames = interp.interpolate(waypoints)  # (N,dim) -> (M,dim)
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 工具函数
# ============================================================================

def _output_time_grid(total_time: float, output_dt: float) -> np.ndarray:
    """Build a near-output_dt grid that always reaches the final waypoint."""
    if total_time <= 0:
        return np.array([0.0], dtype=np.float64)
    intervals = max(int(np.ceil(total_time / output_dt)), 1)
    return np.linspace(0.0, total_time, intervals + 1, dtype=np.float64)

def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas 算法求解三对角线性方程组 Ax=d, 其中:
    a = sub-diagonal (n-1 个)
    b = main diagonal (n 个)
    c = super-diagonal (n-1 个)
    d = right-hand side (n 个)
    """
    n = len(d)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([d[0] / b[0]], dtype=np.float64)
    cp = np.empty(n - 1, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])

    x = np.empty(n, dtype=np.float64)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ============================================================================
# 插值基类
# ============================================================================

class BaseInterpolator:
    dim: int

    def __init__(self, dim: int = 14):
        self.dim = dim

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        """actions: (N, dim), 返回插值后 (M, dim)"""
        return actions

    def reset(self):
        pass


# ============================================================================
# 线性插值
# ============================================================================

class LinearInterpolator(BaseInterpolator):
    """逐关节线性插值 (np.interp)，速度在控制点处不连续。"""

    def __init__(self, dim: int = 14, input_dt: float = 1.0 / 30, output_dt: float = 1.0 / 250):
        super().__init__(dim)
        self.input_dt = float(input_dt)
        self.output_dt = float(output_dt)

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        N = len(actions)
        if N < 2:
            return actions.astype(np.float32)

        t_in = np.arange(N, dtype=np.float64) * self.input_dt
        total_time = (N - 1) * self.input_dt
        t_out = _output_time_grid(total_time, self.output_dt)
        M = len(t_out)

        result = np.empty((M, self.dim), dtype=np.float32)
        for j in range(self.dim):
            result[:, j] = np.interp(t_out, t_in, actions[:, j].astype(np.float64))
        return result


# ============================================================================
# 三次样条插值 (Natural Cubic Spline)
# ============================================================================

class CubicInterpolator(BaseInterpolator):
    """Natural cubic spline — C2 连续（位置、速度、加速度）。无冲击。"""

    def __init__(self, dim: int = 14, input_dt: float = 1.0 / 30, output_dt: float = 1.0 / 250):
        super().__init__(dim)
        self.input_dt = float(input_dt)
        self.output_dt = float(output_dt)

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        N = len(actions)
        if N < 2:
            return actions.astype(np.float32)
        if N == 2:
            # 退化为线性
            lin = LinearInterpolator(self.dim, self.input_dt, self.output_dt)
            return lin.interpolate(actions)

        t = np.arange(N, dtype=np.float64) * self.input_dt
        h = np.diff(t)  # (N-1,)

        total_time = (N - 1) * self.input_dt
        t_out = _output_time_grid(total_time, self.output_dt)
        M = len(t_out)

        # --- 构建三对角系统 A·M = B ---
        n_inner = N - 2
        a_sub = h[:-1].astype(np.float64).copy()     # sub
        b_diag = (2.0 * (h[:-1] + h[1:])).astype(np.float64)  # main
        c_sup = h[1:].astype(np.float64).copy()      # super

        result = np.empty((M, self.dim), dtype=np.float32)

        for j in range(self.dim):
            y = actions[:, j].astype(np.float64)
            B = np.empty(n_inner, dtype=np.float64)
            for i in range(n_inner):
                B[i] = 6.0 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])

            M_inner = _tridiag_solve(a_sub, b_diag, c_sup, B)
            M_vec = np.zeros(N, dtype=np.float64)
            M_vec[1:N - 1] = M_inner

            # --- 多项式系数 (power form, per segment) ---
            # P(t) = a + b*(t-t_i) + c*(t-t_i)^2 + d*(t-t_i)^3
            coeffs_a = np.empty(N - 1, dtype=np.float64)
            coeffs_b = np.empty(N - 1, dtype=np.float64)
            coeffs_c = np.empty(N - 1, dtype=np.float64)
            coeffs_d = np.empty(N - 1, dtype=np.float64)
            for i in range(N - 1):
                hi = h[i]
                coeffs_a[i] = y[i]
                coeffs_c[i] = M_vec[i] / 2.0
                coeffs_d[i] = (M_vec[i + 1] - M_vec[i]) / (6.0 * hi)
                coeffs_b[i] = (y[i + 1] - y[i]) / hi - hi * (2.0 * M_vec[i] + M_vec[i + 1]) / 6.0

            # --- 求值 ---
            seg_indices = np.clip(np.searchsorted(t, t_out, side='right') - 1, 0, N - 2)
            for k in range(M):
                si = seg_indices[k]
                dt = t_out[k] - t[si]
                result[k, j] = float(coeffs_a[si] + dt * (coeffs_b[si] + dt * (coeffs_c[si] + dt * coeffs_d[si])))

        return result


# ============================================================================
# 五次 Hermite 样条插值
# ============================================================================

class QuinticInterpolator(BaseInterpolator):
    """Piecewise quintic Hermite — C2 连续，匹配位置/速度/加速度。

    速度与加速度由中心差分估计，端点用前/后向差分。
    """

    def __init__(self, dim: int = 14, input_dt: float = 1.0 / 30, output_dt: float = 1.0 / 250):
        super().__init__(dim)
        self.input_dt = float(input_dt)
        self.output_dt = float(output_dt)

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        N = len(actions)
        if N < 3:
            # 点数太少，用线性
            lin = LinearInterpolator(self.dim, self.input_dt, self.output_dt)
            return lin.interpolate(actions)

        t = np.arange(N, dtype=np.float64) * self.input_dt
        h = np.diff(t)
        total_time = (N - 1) * self.input_dt
        t_out = _output_time_grid(total_time, self.output_dt)
        M = len(t_out)

        result = np.empty((M, self.dim), dtype=np.float32)

        for j in range(self.dim):
            y = actions[:, j].astype(np.float64)

            # 估计速度 (中心差分)
            v = np.zeros(N, dtype=np.float64)
            v[0] = (y[1] - y[0]) / h[0]  # forward
            for i in range(1, N - 1):
                v[i] = (y[i + 1] - y[i - 1]) / (h[i - 1] + h[i])
            v[N - 1] = (y[N - 1] - y[N - 2]) / h[N - 2]  # backward

            # 估计加速度 (中心差分 of velocities)
            a = np.zeros(N, dtype=np.float64)
            a[0] = (v[1] - v[0]) / h[0]
            for i in range(1, N - 1):
                a[i] = (v[i + 1] - v[i - 1]) / (h[i - 1] + h[i])
            a[N - 1] = (v[N - 1] - v[N - 2]) / h[N - 2]

            # --- 逐段求系数 ---
            # P(t)=c0 + c1*τ + c2*τ² + c3*τ³ + c4*τ⁴ + c5*τ⁵,  τ = t - t_i
            coeffs = np.zeros((N - 1, 6), dtype=np.float64)
            for i in range(N - 1):
                hi = h[i]
                hi2 = hi * hi
                hi3 = hi2 * hi
                hi4 = hi3 * hi
                hi5 = hi4 * hi

                # 已知: P(0)=y_i, P'(0)=v_i, P''(0)=a_i, P(hi)=y_{i+1}, P'(hi)=v_{i+1}, P''(hi)=a_{i+1}
                c0 = y[i]
                c1 = v[i]
                c2 = a[i] / 2.0

                # 3 变量线性方程组求 c3, c4, c5:
                #  c3*hi3 + c4*hi4 + c5*hi5 = y[i+1] - (c0 + c1*hi + c2*hi2)
                #  3*c3*hi2 + 4*c4*hi3 + 5*c5*hi4 = v[i+1] - (c1 + 2*c2*hi)
                #  6*c3*hi + 12*c4*hi2 + 20*c5*hi3 = a[i+1] - 2*c2
                rhs1 = y[i + 1] - (c0 + c1 * hi + c2 * hi2)
                rhs2 = v[i + 1] - (c1 + 2.0 * c2 * hi)
                rhs3 = a[i + 1] - 2.0 * c2

                # 矩阵 [hi3 hi4 hi5; 3hi2 4hi3 5hi4; 6hi 12hi2 20hi3] * [c3 c4 c5]^T = [rhs1 rhs2 rhs3]^T
                # Cramer's rule
                det = (hi3 * (4 * hi3 * 20 * hi3 - 5 * hi4 * 12 * hi2)
                       - hi4 * (3 * hi2 * 20 * hi3 - 5 * hi4 * 6 * hi)
                       + hi5 * (3 * hi2 * 12 * hi2 - 4 * hi3 * 6 * hi))

                if abs(det) > 1e-30:
                    inv_det = 1.0 / det
                    # cofactor matrix for c3
                    c3 = ((rhs1 * (4 * hi3 * 20 * hi3 - 5 * hi4 * 12 * hi2)
                           + hi4 * (rhs3 * 5 * hi4 - rhs2 * 20 * hi3)
                           + hi5 * (rhs2 * 12 * hi2 - rhs3 * 4 * hi3)) * inv_det)
                    # cofactor matrix for c4
                    c4 = ((hi3 * (rhs2 * 20 * hi3 - rhs3 * 5 * hi4)
                           + rhs1 * (5 * hi4 * 6 * hi - 3 * hi2 * 20 * hi3)
                           + hi5 * (rhs3 * 3 * hi2 - rhs2 * 6 * hi)) * inv_det)
                    # cofactor matrix for c5
                    c5 = ((hi3 * (4 * hi3 * rhs3 - 12 * hi2 * rhs2)
                           + hi4 * (6 * hi * rhs2 - 3 * hi2 * rhs3)
                           + rhs1 * (3 * hi2 * 12 * hi2 - 4 * hi3 * 6 * hi)) * inv_det)
                else:
                    c3 = c4 = c5 = 0.0

                coeffs[i] = [c0, c1, c2, c3, c4, c5]

            # --- 求值 ---
            seg_indices = np.clip(np.searchsorted(t, t_out, side='right') - 1, 0, N - 2)
            for k in range(M):
                si = seg_indices[k]
                dt = t_out[k] - t[si]
                c = coeffs[si]
                dt2 = dt * dt
                result[k, j] = float(c[0] + dt * (c[1] + dt * (c[2] + dt * (c[3] + dt * (c[4] + dt * c[5])))))

        return result


# ============================================================================
# 工厂函数
# ============================================================================

def create_interpolator(method: str = "cubic", dim: int = 14,
                        input_hz: float = 30.0, output_hz: float = 250.0) -> BaseInterpolator:
    """按方法名创建插值器。method: linear / cubic / quintic。"""
    input_dt = 1.0 / float(input_hz)
    output_dt = 1.0 / float(output_hz)
    if method == "linear":
        return LinearInterpolator(dim, input_dt, output_dt)
    elif method == "cubic":
        return CubicInterpolator(dim, input_dt, output_dt)
    elif method == "quintic":
        return QuinticInterpolator(dim, input_dt, output_dt)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
