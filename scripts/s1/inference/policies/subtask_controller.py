"""子任务控制器——fineart System 2 自动子任务的上层调度（11-10 v1 干跑架）。

分层（11-10 定案）：
  引擎（FlashRT）  = 两种推理原语：predict（S1 动作块）/ generate_subtask（S2 文本）
  本模块（策略层） = 拥有 S1 prompt 槽的字符串：goal → S2 生成 → hold/切换判定
  traj 引擎        = 引擎零改。SubtaskProxy 代理 model.predict 回填 TE.model，
                     S2 串行落进预取 worker 线程——worker 本就异步于 tick，
                     且与下一次 S1 infer 同线程天然串行（GPU 无并发竞争对象），
                     ~0.9s(S2)+~0.4s(S1) < 块长 1.67s → 完全藏进块窗口。
                     控制器关（不给 --subtask-goal）= 在役路径逐位不变。

语义分工（lerobot recipe 双流，configuration_fineart_vla.py 默认 recipe）：
  goal（高层目标）  只喂 S2 的 ${task} 槽（high_level_subtask 流）
  S2 产出（子任务） 才进 S1 的 ${subtask} 槽（low_level_execution 流，权重 0.70）
  两串不可混用；S2 首查落地前 S1 吃 --prompt 原值作种子串。

安全律：
  - S2 任何失败（异常/空文本）→ 保当前串、连败计数，绝不让 S1 循环死；
    连败 3 次→休眠（停查，只打日志）——坏掉的 S2 不允许按节拍持续烧 0.9s。
  - 启动 warm 首查（SubtaskController.warm）：吸收 S2 解码器 ~7.4GB 懒建
    +首调成本进 init 计时区外（同入口「预热推理 ×3」惯技）——否则首次在飞
    S2 可能顶穿 starve-limit 2s；顺带在启动打印处实锤 050000 ckpt 的 S2 头
    在本域的生成质量（挂账验证项①）。失败不致命（保种子串继续）。
  - hold v1 = 文本 diff（strip 后相等→不切换）；completed/validator 判定后置
    （lerobot autosteer 的 hold/completed 三态在 steerable_pipeline，另起一轮）。

线程模型：控制器无自有线程。refresh() 由预取 worker 在 predict 前同步调用；
  同一时刻只有一个 predict 在飞（引擎既有约束）→ 状态无锁。

已知 v1 缺口：引擎 _dump_chunk 的 evidence 记录仍记 args.prompt（CLI 原值），
  切换史只在本模块日志；summary() 接进引擎退出汇总待后置（TE.run() 内
  os._exit 不可挂钩）。

自测：``python3 policies/subtask_controller.py``（FakeModel 状态机五案，
零 GPU 零重依赖）。
"""
from __future__ import annotations

import os
import time

# safetensors 只在 ckpt 预检时懒 import——本模块须可被任意 python3 自测。

# S2 文本头在 lerobot safetensors 的键尾（引擎侧 FineARTVLATorchFrontendRtx.
# TEXT_LM_HEAD_KEY 同源；带 model. 包装前缀时 endswith 命中）。
_TEXT_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"

_FAIL_LIMIT = 3   # 连败休眠阈值


def _ckpt_has_text_head(ckpt_dir) -> bool:
    """ckpt safetensors 头部预检（只读 header，零张量加载）：有无 S2 文本头。"""
    from safetensors import safe_open

    path = os.path.join(str(ckpt_dir), "model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        return any(k.endswith(_TEXT_HEAD_KEY) for k in f.keys())


class SubtaskController:
    """S1 prompt 槽的持有者：goal 固定，current（当前子任务串）按节拍复查。

    节拍语义：warm 首查在启动（init 计时区外）落地初始子任务；循环内每
    ``every`` 次 re-plan 复查一次。绝不上 tick 路径。
    """

    def __init__(self, model, goal, every=30, max_new_tokens=64,
                 ckpt_dir=None, seed=None, log=print):
        self._model = model                 # 裸 flash_rt wrapper（.pipeline 面）
        self.goal = str(goal)
        self.every = max(1, int(every))
        self.max_new_tokens = int(max_new_tokens)
        self._log = log
        self.current = str(seed) if seed is not None else self.goal

        # 计数面（summary/日志用）
        self.n_replan = 0                   # refresh() 调用数 = re-plan 块数
        self.n_s2 = 0
        self.n_swaps = 0
        self.n_holds = 0
        self.s2_ms_last = None
        self._fails = 0                     # 连败计数（成功归零）
        self._since = 0                     # 距上次 S2 的 re-plan 数
        self._dormant = False

        if ckpt_dir is not None and not _ckpt_has_text_head(ckpt_dir):
            raise SystemExit(
                f"[subtask] ckpt 无 System 2 文本头（*{_TEXT_HEAD_KEY}）："
                f"{ckpt_dir} —— --subtask-goal 需带 S2 头的 fineart ckpt")

    def warm(self, obs=None):
        """启动首查：懒建吸收 + 初始子任务落地。失败不致命（保种子串）。

        obs 三件套（grab_views 产物）给真图；None = 纯文本前缀（引擎支持）。
        """
        text = self._s2_safe(obs)
        if text is not None:
            old = self.current
            self.current = text
            self.n_swaps += 1               # 种子 → S2 产出也算一次切换
            self._log(f"[subtask] warm 首查 {self.s2_ms_last / 1000:.2f}s "
                      f"swap {old!r} → {text!r}")
        else:
            self._log(f"[subtask] warm 首查失败 → S1 吃种子串 {self.current!r}"
                      "（循环内按节拍再试）")

    def refresh(self, obs, fallback=None):
        """预取 worker 每次 re-plan（predict 前）同步调用：到节拍→S2 复查。

        其余调用零成本（计数+返回当前串）。``fallback`` =引擎原 prompt
        （--prompt 原值），仅当控制器无串时兜底（正常路径永远有串）。
        绝不抛——S1 循环的性命不押在 S2 上。
        """
        self.n_replan += 1
        self._since += 1
        if self._dormant or self._since < self.every or obs is None:
            return self.current
        self._since = 0
        text = self._s2_safe(obs)
        if text is None:
            return self.current
        if text == self.current:
            self.n_holds += 1
            self._log(f"[subtask] #{self.n_replan} "
                      f"{self.s2_ms_last / 1000:.2f}s hold {text!r}")
            return self.current
        old = self.current
        self.current = text
        self.n_swaps += 1
        self._log(f"[subtask] #{self.n_replan} "
                  f"{self.s2_ms_last / 1000:.2f}s swap {old!r} → {text!r}")
        return self.current

    def _s2_safe(self, obs):
        """跑一次 S2：异常/空文本→None+连败计数（3 次休眠），绝不抛。"""
        t = time.perf_counter()
        try:
            raw = self._model.pipeline.generate_subtask(
                self.goal, obs, max_new_tokens=self.max_new_tokens)
            dt = (time.perf_counter() - t) * 1000
        except Exception as e:   # noqa: BLE001  安全律：S2 死活不影响 S1
            self._fails += 1
            self._log(f"[subtask] S2 失败（{self._fails}/{_FAIL_LIMIT}）"
                      f"→ 保 {self.current!r}: {type(e).__name__}: {e}")
            if self._fails >= _FAIL_LIMIT and not self._dormant:
                self._dormant = True
                self._log("[subtask] ⚠ 连败 3 次 → 休眠（停查 S2，S1 继续吃"
                          "当前串；排查 S2 头/显存后重启进程）")
            return None
        self.s2_ms_last = dt
        self.n_s2 += 1
        self._fails = 0
        text = str(raw).strip()
        if not text:
            self._log(f"[subtask] S2 空文本（{dt:.0f}ms）→ 保 "
                      f"{self.current!r}")
            return None
        return text

    def summary(self):
        return (f"[subtask] goal={self.goal!r} every={self.every} "
                f"replan={self.n_replan} s2={self.n_s2} swap={self.n_swaps} "
                f"hold={self.n_holds} fail={self._fails} "
                f"dormant={self._dormant} current={self.current!r}")


class SubtaskProxy:
    """model.predict 代理——引擎 seam 零改的唯一机关。

    回填 TE.model 后，预取 worker 的 predict 自动先过控制器（S2 藏进块
    窗口）再委托底层。除 predict 外全属性透传（pipeline 等原面不动）。
    """

    def __init__(self, model, ctrl):
        self._model = model
        self._ctrl = ctrl

    def predict(self, obs, prompt=None, state=None, rtc_prefix=None):
        prompt = self._ctrl.refresh(obs, fallback=prompt)
        return self._model.predict(obs, prompt=prompt, state=state,
                                   rtc_prefix=rtc_prefix)

    def __getattr__(self, name):
        # 常规查找失败才进；_model/_ctrl 未及赋值时直抛防递归（下划线一面
        # 不透传：wrapper._pipe 等私有面不经代理外泄）
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._model, name)


# ── 自测：状态机五案（节拍首查即热/hold/swap/连败休眠/代理透传），零 GPU ──
if __name__ == "__main__":

    class _FakeModel:
        """generate_subtask 按剧本回放的假引擎（predict 面另用 _FakeUnder）。"""

        def __init__(self, script):
            self.calls = []
            self._script = list(script)

            class _P:
                pass

            self.pipeline = _P()
            self.pipeline.generate_subtask = self._gen

        def _gen(self, goal, obs, max_new_tokens=64):
            self.calls.append((goal, obs, max_new_tokens))
            item = self._script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class _FakeUnder:
        pipeline = object()   # 透传案用

        def __init__(self):
            self.kwargs = None

        def predict(self, obs, prompt=None, state=None, rtc_prefix=None):
            self.kwargs = {"prompt": prompt, "state": state,
                           "rtc_prefix": rtc_prefix}
            return "ACT"

    _logs = []
    _say = _logs.append
    _ok = True

    def _check(name, cond):
        global _ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        _ok = _ok and bool(cond)

    print("== subtask_controller 自测 ==")

    # 案1 warm 首查落地 + 节拍（every=3 → 第 3 次 refresh 复查）
    m = _FakeModel(["pick up the red block",   # warm
                    "pick up the red block",   # @#3 hold
                    "place it into the tray",  # @#6 swap
                    RuntimeError("boom"),      # @#9 fail1
                    RuntimeError("boom"),      # @#12 fail2
                    RuntimeError("boom")])     # @#15 fail3 → 休眠
    c = SubtaskController(m, goal="clean the table", every=3,
                          seed="test", log=_say)
    c.warm({"image": "img"})
    _check("warm 落地 S2 产出", c.current == "pick up the red block"
           and c.n_swaps == 1 and c.s2_ms_last is not None)
    r1 = c.refresh({"image": "i1"})
    r2 = c.refresh({"image": "i2"})
    _check("未到节拍零调用", len(m.calls) == 1 and r1 == r2
           == "pick up the red block" and c.n_replan == 2)
    r3 = c.refresh({"image": "i3"})
    _check("#3 同文 hold", len(m.calls) == 2 and c.n_holds == 1
           and r3 == "pick up the red block")

    # 案2 #6 换文 swap（#4/#5 走零成本路径）
    c.refresh({"image": "i4"})
    c.refresh({"image": "i5"})
    r6 = c.refresh({"image": "i6"})
    _check("#6 换文 swap", c.current == "place it into the tray"
           and c.n_swaps == 2 and r6 == "place it into the tray")

    # 案3 #9/#12/#15 连败 3 → 休眠；休眠后不再调用
    for _ in range(9):
        c.refresh({"image": "x"})
    _check("连败 3 休眠", c._dormant and c.n_s2 == 3 and c.n_replan == 15)
    n_before = len(m.calls)
    c.refresh({"image": "y"})
    c.refresh({"image": "z"})
    _check("休眠后零 S2 调用", len(m.calls) == n_before
           and c.current == "place it into the tray")

    # 案4 refresh 绝不抛（异常路径返回当前串）
    _check("失败路径返回当前串", isinstance(r6, str))

    # 案5 代理：prompt 替换 + 透传
    under = _FakeUnder()
    px = SubtaskProxy(under, c)
    out = px.predict({"obs": 1}, prompt="seed", state="s", rtc_prefix="r")
    _check("代理 prompt=当前串", under.kwargs["prompt"]
           == "place it into the tray" and out == "ACT")
    _check("代理 rtc_prefix/state 透传", under.kwargs["rtc_prefix"] == "r"
           and under.kwargs["state"] == "s")
    _check("代理属性透传 pipeline", px.pipeline is under.pipeline)
    try:
        px._pipe
        _check("私有面不透传", False)
    except AttributeError:
        _check("私有面不透传", True)

    # 案6 ckpt 预检（真 ckpt 若在：必须有头；不在：跳过）
    _FA = ("/home/galbot/model_weights/pi05/"
           "fineart_vla_rtc_subtask_relative_loss_monitor_050000/"
           "pretrained_model")
    if os.path.isfile(os.path.join(_FA, "model.safetensors")):
        _check("050000 ckpt 有 S2 文本头", _ckpt_has_text_head(_FA))
    else:
        print("  SKIP  ckpt 不在本机，预检案跳过")

    print(c.summary())
    print("== 全部通过 ==" if _ok else "== 有 FAIL ==")
    raise SystemExit(0 if _ok else 1)
