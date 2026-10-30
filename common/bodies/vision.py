"""帧口径纯件：letterbox224 / 头图训练裁剪（common 收编版 10-30 步2a）。

原文=run_g1_loop_traj.py 逐行（S1 移植版同文逐位一致——两机训练口径件），
搬运不改逻辑。
"""
import cv2
import numpy as np


def letterbox224(img):
    """等比缩放长边到 224 + 上下纯黑条填充——精确复现训练数据集的图像变换。

    2026-10-12 实锤（本机 pick_place_balence 抽帧判读）：三视角训练帧全部为
    letterbox——224×224、上下各 28px 纯黑条、内容带 168px=224×960/1280（源帧
    1280×960 等比缩放，非拉伸非裁剪）。此前 grab_views 直拉 resize 满幅无条
    = train/serve 图像分布失配（10-10 视觉-臂接地微弱的第一嫌疑，推翻当日
    "喂图净"结论——该验证未覆盖预处理变换维度）。
    """
    h, w = img.shape[:2]
    _s = 224.0 / max(h, w)
    _nh, _nw = round(h * _s), round(w * _s)
    _cv = np.zeros((224, 224, 3), dtype=np.uint8)
    _y, _x = (224 - _nh) // 2, (224 - _nw) // 2
    _cv[_y:_y + _nh, _x:_x + _nw] = cv2.resize(img, (_nw, _nh))
    return _cv


# ── 头图训练裁剪放大（--head-crop，2026-10-19 trained-RTC 权重试跑）────────
# 复现同事数据管线对头相机的处理：/home/galbot/image_preprocessor.py
# ImagePreprocessor._apply_head_training_crop 原样内联（其顶层 import
# emulate_data_collection 在 /home/galbot/infer_policy/，跨目录 import 脆弱，
# 故不 import 只内联，常量逐一对齐源文件；与 ab_test/1015 B 线内联逐位一致）。
# 训练视频头图 = letterbox224 帧上裁 [126:214, 52:172] → Lanczos4 拉回
# 164×224（≈1.87× 放大）→ 上下各补 30px 黑条回 224。仅作用头相机键 "image"
# （双腕图训练无此处理）；裁剪/缩放/补边均为逐通道空间算子，BGR/RGB 通道序无关。
HEAD_TRAINING_CROP_YX = (126, 214, 52, 172)   # y0, y1, x0, x1（224×224 输入上）
HEAD_TRAINING_CONTENT_HW = (164, 224)         # 裁剪内容拉回的目标 (H, W)
HEAD_TRAINING_PAD_TBLR = (30, 30, 0, 0)       # 上/下/左/右黑边 px


def head_training_crop(img):
    """letterbox224 后的头图 → 训练同款裁剪放大回 224×224。"""
    if img.ndim != 3 or img.shape[:2] != (224, 224):
        raise ValueError(f"head_training_crop 期望 224x224 HWC 输入，"
                         f"got shape={img.shape}")
    y0, y1, x0, x1 = HEAD_TRAINING_CROP_YX
    if not (0 <= y0 < y1 <= img.shape[0] and 0 <= x0 < x1 <= img.shape[1]):
        raise ValueError(f"head_training_crop 窗口 [{y0}:{y1}, {x0}:{x1}] 越界")
    ch, cw = HEAD_TRAINING_CONTENT_HW
    content = cv2.resize(img[y0:y1, x0:x1], (cw, ch),
                         interpolation=cv2.INTER_LANCZOS4)
    pt, pb, pl, pr = HEAD_TRAINING_PAD_TBLR
    out = cv2.copyMakeBorder(content, pt, pb, pl, pr,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if out.shape[:2] != (224, 224):
        raise ValueError(f"head_training_crop 输出形状异常: {out.shape[:2]}")
    return out
