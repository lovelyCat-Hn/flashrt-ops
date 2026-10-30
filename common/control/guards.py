"""安全护栏纯件：chunk 绝对构型换算 / 关节包络检查（common 收编版 10-30 步2b-1）。

原文=run_g1_loop_traj.py 逐行（S1 移植版同文逐位一致），搬运不改逻辑。
跨模块只读状态走模块属性：入口初始化后回填本模块同名属性（一次即可，
其后两机入口均不再重绑）——

    import common.control.guards as GD
    GD.BASE_ARM = ...   # 首块 delta 基准（换块随 job 更新后须回填）
    GD.ENV_LO, GD.ENV_HI = ...   # 关节包络盒（数据集 action 列 [min,max]，含余量）
"""
import numpy as np

BASE_ARM = None   # 入口回填：首块 delta 基准（换块随 job 更新后须回填）
ENV_LO = None     # 入口回填：关节包络盒下界 (16,)（None=护栏关）
ENV_HI = None     # 入口回填：关节包络盒上界 (16,)（None=护栏关）


def chunk_abs_block(chunk):
    """chunk delta 块 → 绝对构型块（包络护栏的比对对象，2026-10-07 修范畴错误）。

    训练管线 relative_actions_processor：臂维=delta（相对本块预测时刻臂位
    BASE_ARM）、夹爪维=绝对 0-100。
    包络盒来自数据集 action 列逐维 [min,max]=绝对构型——必须用 BASE_ARM+delta
    比对；直接拿 delta 比对=必拦停（delta≈0 恒不在盒内，10-07 首跑实录）。
    """
    a = np.empty_like(chunk)
    a[:, :7] = chunk[:, :7] + BASE_ARM[None, :7]
    a[:, 7] = chunk[:, 7]            # 夹爪界=±inf，原值恒过
    a[:, 8:15] = chunk[:, 8:15] + BASE_ARM[None, 7:14]
    a[:, 15] = chunk[:, 15]
    return a


def env_first_violation(ch):
    """关节包络检查：返回首个越界行 (行号, 维, 值, 界)；全干净返回 None。

    ch: (n,16) 数据集维序动作块。臂维 0-6=右臂、8-15=左臂（ENV 已含余量），
    夹爪维界=±inf 恒过。逐行向量化，n≤chunk_size(50)，微秒级。
    """
    bad = (ch > ENV_HI[None, :]) | (ch < ENV_LO[None, :])
    rows = np.flatnonzero(bad.any(axis=1))
    if rows.size == 0:
        return None
    r = int(rows[0])
    j = int(np.flatnonzero(bad[r])[0])
    upper = bool(ch[r, j] > ENV_HI[j])
    return r, j, float(ch[r, j]), float(ENV_HI[j] if upper else ENV_LO[j])
