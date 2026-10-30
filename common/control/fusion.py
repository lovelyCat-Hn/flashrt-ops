"""换块融合纯件：链式限幅+EMA 行目标 / 样条分帧（common 收编版 10-30 步2b-1）。

原文=run_g1_loop_traj.py 逐行（S1 移植版同文逐位一致），搬运不改逻辑。
跨模块状态走模块属性：`args`/插值器/包络盒/爪标定等 init 后只读、一次回填；
`chunk`/`BASE_ARM`/`_EMA_STATE` 是活状态——入口每处重绑后必须回填同名属性
（build_rows 内部经 `global _EMA_STATE` 改写即写在本模块，入口侧复位/重播
也直写 `FU._EMA_STATE`，全链只有这一份状态）——

    import common.control.fusion as FU
    FU.args, FU.chunk, FU.BASE_ARM = args, chunk, BASE_ARM
    FU.INTERP, FU.GRIP_INTERP, FU.GRIP = INTERP, GRIP_INTERP, GRIP
    FU.ENV_LO, FU.ENV_LO14, FU.ENV_HI14 = ENV_LO, ENV_LO14, ENV_HI14
    ...
    FU.chunk = chunk            # 每处重绑后（job.result/截断/跳帧安装/入口匹配）
    FU._EMA_STATE = None        # 循环起点复位 / 换块 last_tgt 重播
"""
import numpy as np

args = None        # 入口回填：argparse Namespace（delta_max/ema_alpha/seam 族…）
chunk = None       # 活状态：当前块 (n,16) delta 计划（入口重绑后回填）
BASE_ARM = None    # 活状态：本块 delta 基准臂位 (14,)（换块随 job 更新后回填）
INTERP = None      # 入口回填：臂轨迹插值器（create_interpolator 产物）
GRIP_INTERP = None  # 入口回填：爪轨迹插值器（dim=2）
GRIP = None        # 入口回填：爪标定/状态 dict（None=无爪）；send_grip* 原地改写
ENV_LO = None      # 入口回填：包络盒下界 (16,)（None=帧级钳位关）
ENV_LO14 = None    # 入口回填：包络盒下界臂 14 维（帧级钳位用）
ENV_HI14 = None    # 入口回填：包络盒上界臂 14 维（帧级钳位用）
_EMA_STATE = None  # 活状态：跨块连续 EMA（上块末行滤波输出；None=无历史）


def build_rows(cur, prev0=None, offset=None, seam=False):
    """chunk delta 块 → 链式限幅行目标（14 维，右臂在前）。

    ⚠ 2026-09-23 语义修正（同母本 plan_step）：训练管线
    relative_actions_processor 把臂维动作转成 delta（相对本块预测时刻臂位
    BASE_ARM）——行目标不是 chunk 绝对值，须加回 BASE_ARM（换块时随 job 更新）。
    链式：rows[0]=cur 向行 0 绝对位限幅挪一步，rows[k]=rows[k−1] 再向行 k 挪——
    每行位移 ≤ --max-joint-delta-per-step，且逐行夹在 cur 与 chunk 绝对位之间（包络盒凸性 →
    行目标自动在包络内；样条过冲由 build_block_stream 的帧级钳位兜底）。
    EMA：--filter-alpha>0 时状态跨块连续（infer_policy action_filter 同款语义，
    2026-10-12 对照判明块内重启版对接缝无效）：新块 rows[0] 先与上块末行滤波
    输出混合再起链，接缝首行同样被滤；首块/循环起点无历史即以 rows[0] 播种。
    滤波对象=链钳后的绝对行目标（与同事"钳位→滤波"同序），凸组合仍在包络内。
    seam=True：接缝窗 α 阶梯（10-17 回放台移植）——换块后前 --seam-beats 拍
    α 从 --seam-alpha 线性爬回主 α，削块首衰减尾峰值、稳态零税；仅跨块
    有历史时生效（首块无接缝）。缺省 False=原口径逐位不变。
    """
    global _EMA_STATE
    rows = np.empty((len(chunk), 14), dtype=np.float32)
    # prev0=链起点（skip-install 参考连续用）：缺省从 cur 出发（原口径）；
    # 传入 last_tgt 时链从上一拍下发目标接续——首行=last_tgt+clip(新首行−
    # last_tgt, ±delta_max)，参考不重锚（110411 判明：换块重锚=脉动残留源）。
    prev = (np.asarray(cur, dtype=np.float32) if prev0 is None
            else np.asarray(prev0, dtype=np.float32))
    for kk in range(len(chunk)):
        arm_tgt = BASE_ARM + np.concatenate([chunk[kk][:7], chunk[kk][8:15]])
        if offset is not None:
            arm_tgt = arm_tgt + offset      # --ref-offset：整条计划平移（块首零回走）
        prev = prev + np.clip(arm_tgt - prev, -args.delta_max, args.delta_max)
        rows[kk] = prev
    if args.ema_alpha > 0:
        a = float(args.ema_alpha)
        # 接缝窗 α 阶梯（seam=True 且有跨块历史时启用）：α 从 seam_alpha
        # 线性爬回主 α；硬切会把积压滞后一次放出成新凸点（回放台离线实证）
        _ramp = seam and args.seam_alpha > 0 and _EMA_STATE is not None
        a_seam = float(args.seam_alpha)
        n_sb = max(1, int(args.seam_beats))
        # 接缝 α 助推（--seam-boost-alpha，2026-10-26 块间停顿案 160323）：
        # 与 seam-ramp 反向。换块 EMA 重播回 last_tgt（防瞬移修正）清零了
        # 速度状态 → 命令速度块首=α×模型速度、τ=1/α≈整块，水位0 短块(11拍)
        # 永在瞬态=目视停顿；块首 n 拍用大 α 快速重建速度再线性落回主 α，
        # 稳态纹理零税。α 全程 0.3 对照=纹理粗 50%（161429 残差 p50
        # 11.6→20.7），勿再走。与 _ramp 互斥（守卫把关，此处 boost 优先）。
        _boost = args.seam_boost_alpha > 0 and _EMA_STATE is not None
        a_boost = float(args.seam_boost_alpha)
        n_bb = max(1, int(args.seam_boost_beats))
        acc = None
        for kk in range(len(rows)):
            if _boost:
                a_k = a_boost + (a - a_boost) * min(kk, n_bb) / n_bb
            elif _ramp:
                a_k = a_seam + (a - a_seam) * min(kk, n_sb) / n_sb
            else:
                a_k = a
            if kk == 0:
                acc = rows[0] if _EMA_STATE is None else \
                    a_k * rows[0] + (1.0 - a_k) * _EMA_STATE
            else:
                acc = a_k * rows[kk] + (1.0 - a_k) * acc
            rows[kk] = acc
        _EMA_STATE = acc.copy()
    return rows


def build_block_stream(cur, prev0=None, offset=None, seam=False):
    """换块时一次性：行目标 → [链起点]+行目标 样条插值 → 逐行分桶帧。

    返回 (rows_abs, frames_all, bursts)：frames_all (M,14) float32；
    bursts[k]=第 k 拍下发的帧号列表（恰好 --frames-per-row 帧/行）。
    帧级包络钳位同 infer_policy robot_controller._clip_to_joint_limits
    （样条可能过冲）。帧时间网格 = 行节奏 1/args.fps → 帧率
    args.fps×frames_per_row：帧按 wall-clock 匀速铺满整行（--fps<30 时整块
    放慢播放，节奏语义同 native 的慢速彩排）。prev0 透传 build_rows
    （skip-install 参考连续：链起点=last_tgt，首帧恰=last_tgt=零缝步）。
    """
    rows = build_rows(cur, prev0, offset, seam=seam)
    _start = (np.asarray(cur, dtype=np.float32) if prev0 is None
              else np.asarray(prev0, dtype=np.float32))
    per = args.frames_per_row
    if args.merge > 1:
        # 合步（等时粗化，replay 台 3b40d0e 同构）：滤波后每 N 行取 1 个宏步
        #（末行必保=块尾/链状态不动），宏步段 P 帧按拍均分 N 组——同轨迹同速
        # 同墙钟只稀疏化目标，下发 ≈fps×per/N Hz。样条跑在宏步序列上（声明
        # input_hz 仍 30=replay 同款时间折法），行级 EMA/链钳已在 build_rows
        # 完成。返回行=重复宏步行（k 逐拍推进与水位/星饿/RTC 语义不变）。
        idx = list(range(0, len(rows), args.merge))
        if idx[-1] != len(rows) - 1:
            idx[-1] = len(rows) - 1
        rows = np.repeat(rows[idx], args.merge, axis=0)
        n_macro = len(rows) // args.merge
        waypoints = np.vstack([_start[None, :], rows[::args.merge]])
        frames_all = INTERP.interpolate(waypoints)
        if ENV_LO is not None:
            np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :],
                    out=frames_all)
        bursts = []
        for _m in range(n_macro):
            for _grp in np.array_split(np.arange(_m * per, (_m + 1) * per),
                                       args.merge):
                bursts.append([int(i) for i in _grp])   # 空组=静默拍（per<N 时）
        # ── 爪维随臂同管线（2026-10-26 用户定案；--grip-rawrows 恢复旧行率
        # 流）：dim7/15 与臂同 idx 宏步粗化+同款样条逐帧插帧。EMA 上移到行
        # 准备（镜像臂 build_rows：滤波→粗化→插帧，勿在 send 端二次滤波），
        # 换块从 GRIP['sent']（最后实发意图）重播=臂 _EMA_STATE=last_tgt 的
        # 防瞬移镜像；首块/无历史以首行播种。起点=GRIP['sent']（prev0 镜像，
        # 块首帧恰=上块末发值=零缝步）。帧值钳 [0,100]（样条过冲兜底）。
        grip_frames = None
        if GRIP is not None and not args.grip_rawrows:
            def _sent_seed():
                _v = [GRIP["sent"].get(n) for n, _d in GRIP["names"]]
                return (None if any(x is None for x in _v)
                        else np.asarray(_v, dtype=np.float32))
            g = np.stack([chunk[:, d] for _, d in GRIP["names"]],
                         axis=1).astype(np.float32)
            if args.grip_ema_alpha > 0:
                _ab = float(args.grip_ema_alpha)
                _seed = _sent_seed()
                acc = g[0].copy() if _seed is None else _seed.copy()
                for _k in range(len(g)):
                    acc = _ab * g[_k] + (1.0 - _ab) * acc
                    g[_k] = acc
            _seed = _sent_seed()
            _g_start = g[0].copy() if (prev0 is None or _seed is None) else _seed
            _g_wp = np.vstack([_g_start[None, :], g[idx]])
            grip_frames = np.clip(GRIP_INTERP.interpolate(_g_wp), 0.0, 100.0)
        return rows, frames_all, bursts, grip_frames
    waypoints = np.vstack([_start[None, :], rows])
    frames_all = INTERP.interpolate(waypoints)
    if ENV_LO is not None:
        np.clip(frames_all, ENV_LO14[None, :], ENV_HI14[None, :], out=frames_all)
    bursts = [list(range(kk * per, (kk + 1) * per)) for kk in range(len(rows))]
    return rows, frames_all, bursts, None
