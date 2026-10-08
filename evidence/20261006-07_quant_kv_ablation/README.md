# 逐层 K/V cos 量化消融原始记录（2026-10-06 ~ 10-07）

**探针**：`scripts/probes/rot_equivalence_probe.py`（零 FlashRT 改动）
**条件（各批相同，全部默认参数）**：
- 数据集 only_place ep0（437 帧），frames=[0, 60, 120, 180, 240, 300]
- ckpt = models/pi05_g1_place_deploy
- NO_GRAPH eager（PI05_NO_GRAPH=1）、state-prompt=fixed
- K/V cos 口径：编码器 18 层 GQA cache（=解码器交叉注意力的实际输入），
  每层对 bf16 同帧同噪声逐 cos 后取帧均
- 端到端口径：① 输出 vs bf16（cos_arm/cos_grip/grip|d|pp）② teacher-forced
  均值 delta vs 数据集 10 帧 delta（tf-cos）

**⚠ 口径注脚**：10-06 批基线 tf-cos=0.949，10-07 批基线 tf-cos=0.683（探针
条件虽同、两批绝对值不可跨批比，批内自洽）。跨批比较一律看批内相对值。

**来源注记**：原始 stdout 落在 /tmp 探针日志，2026-10-07 16:44 整机重启后
丢失；本目录文本从当期会话记录（transcript）中逐字抢救回来。教训已写入
`.claude/skills/bench-report/SKILL.md` 数据流水线铁律（探针输出必须当场落盘）。

| 文件 | 内容 | 本地时间（UTC+8） |
|---|---|---|
| int8_rot8_20261006.txt | 编码器 W8A8 per-row vs +Hadamard 旋转（真实 kernel 路径复核） | 10-06 17:54 |
| chan8_chan8all_20261006.txt | 编码器权重按通道（per-channel）静态 scale：qkv+gu / 全投影 | 10-06 18:31 |
| dec8_20261007.txt | 编码器 bf16 + 解码器 W8A8（KV 应全 1.0 = 编码器路径未动的 sanity 门） | 10-07 13:54 |
| w8a16_20261007.txt | 解码器 weight-only INT8（KV sanity 门 + 端到端） | 10-07 15:21 |

**复现**：
```
~/holy/run.sh ~/holy/scripts/probes/rot_equivalence_probe.py --configs bf16,int8,rot8
~/holy/run.sh ~/holy/scripts/probes/rot_equivalence_probe.py --configs calib        # 先产通道 scale
~/holy/run.sh ~/holy/scripts/probes/rot_equivalence_probe.py --configs bf16,chan8,chan8all
~/holy/run.sh ~/holy/scripts/probes/rot_equivalence_probe.py --configs bf16,dec8
~/holy/run.sh ~/holy/scripts/probes/rot_equivalence_probe.py --configs bf16,w8a16   # 需 W8A16 kernel flash_rt_kernels.so
```

配套：模块耗时探针同日脚本已抢救回 `scripts/probes/module_timing_probe.py`。
