# Policy 层（多 policy 分类，lerobot 风格）

每个 policy 一个子目录，加载引擎时按目录区分，互不混淆：

```
policies/
├── pi05/            在役：π0.5（FlashRT config="pi05"）
├── fineart_vla/     规划中：FineART-VLA（评估见 agent_memory/galbot-fineart-vla-flashrt-port.md）
└── <新 policy>/     照下方契约添加
```

## 目录契约（加新 policy 照此办）

每个 policy 子目录两件套：

| 文件 | 职责 |
|---|---|
| `engine.py` | 统一加载入口：暴露 `load(**overrides) -> flash_rt VLAModel`。**只固定引擎侧参数**（config 名、framework、量化门控）；权重路径/视角数/动作维等由调用方（闭环脚本 / config/*.toml）注入 |
| `README.md` | 状态（在役/规划/退役）、权重位置、引擎参数定档、适用任务、已知坑 |

## 与 FlashRT 的关系（重要）

- 引擎本体在 `~/holy/FlashRT`（**环境轨**，tar 分发，不入 git）；多模型切换靠
  `flash_rt.api.load_model(config=<名>)` 的模型注册表。
- 新模型的前端/管线代码（如 `frontends/torch/fineart_vla_rtx.py`）落 FlashRT 对应路径，
  **ops 仓以 `hotfix_flashrt/` 镜像随 git 走**（沿用心修机制，bundle 重解后 apply 恢复）。
- `policies/` 属**代码轨**：policy 侧的加载器/文档/配方入 git，换机 git pull 即得。

## 用法

```python
import sys; sys.path.insert(0, "/home/galbot/holy")
from policies.pi05.engine import load
pipe = load(ckpt="/home/galbot/holy/models/pi05_g1_onlypick_deploy", num_views=3)
out = pipe.infer({"image": img, "wrist_image": wrist, "wrist_image_right": wrist_r})
```
