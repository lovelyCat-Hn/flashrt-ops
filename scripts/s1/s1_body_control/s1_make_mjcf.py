"""S1 URDF → MuJoCo 可用模型转换器（自 G1 数字孪生流程沉淀）

厂商 S1 描述是 ROS URDF（/data/galbot/config/galbot_description/
galbot_s1_description/galbot_s1_cali.urdf），mesh 用 package:// 引用——
MuJoCo 不认。本脚本做两件事：
  ① 纯文本重写：package:// 前缀 → 绝对 mesh 路径 + 注入 <mujoco><compiler/>
     块（balanceinertia 兜 G1 时代同款 inertia schema 病；discardvisual=false
     保留渲染几何），产出 s1_twin_ready.urdf（无 mujoco 也能跑本步）
  ② 有 mujoco 的机器上当场编译并 mj_saveLastXML 存 MJCF（便于后续手工加
     端工具 site/tcp），列出关节清单核对

G1 流程对照：G1 用现成 MJCF 仓库（galbot_one_golf_description），S1 无 MJCF
只有 URDF，本脚本补齐这一环。570M meshes 不入 holy 仓——Windows 端整目录
拷贝 galbot_s1_description 后指向本地路径即可（同 G1 的 D:\\work\\... 玩法）。

用法:
  # 本机（只做①）:
  ~/miniforge3/envs/flash_pyrt311/bin/python s1_make_mjcf.py
  # 任意装了 mujoco 的机器（①+②）:
  python s1_make_mjcf.py --compile
"""
import argparse
import pathlib
import re
import sys

DEFAULT_DESC_DIR = "/data/galbot/config/galbot_description/galbot_s1_description"
DEFAULT_URDF = f"{DEFAULT_DESC_DIR}/galbot_s1_cali.urdf"
# URDF 里的 package 名（filename="package://galbot_s1_description/..."）
PACKAGE_NAME = "galbot_s1_description"
# 注入的编译块：balanceinertia 兜惯性病；discardvisual=false 保渲染几何
MUJOCO_BLOCK = (
    "<mujoco>\n"
    "  <compiler balanceinertia=\"true\" discardvisual=\"false\"/>\n"
    "</mujoco>\n"
)


def rewrite(urdf_path, mesh_root, out_path):
    """package:// 前缀重写为绝对路径 + 注入 compiler 块。返回 (关节数, 剩余package引用数)。"""
    text = pathlib.Path(urdf_path).read_text()
    # 绝对路径喂 MuJoCo 最省事（避免相对路径对 cwd 的依赖）
    text = text.replace(f"package://{PACKAGE_NAME}/", f"{mesh_root.rstrip('/')}/")
    # compiler 块插在 <robot ...> 开标签后（已存在则跳过，幂等）
    if "<mujoco>" not in text:
        text = re.sub(r"(<robot[^>]*>)", r"\1\n" + MUJOCO_BLOCK, text, count=1)
    pathlib.Path(out_path).write_text(text)
    leftover = text.count("package://")
    joints = re.findall(r'<joint name="([^"]+)" type="(revolute|prismatic|continuous)"',
                        text)
    return joints, leftover


def compile_mjcf(ready_urdf, out_xml):
    import mujoco  # noqa: 延迟导入，无 mujoco 时①仍可用
    model = mujoco.MjModel.from_xml_path(str(ready_urdf))
    mujoco.mj_saveLastXML(str(out_xml), model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model.njnt)]
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("用法")[0])
    ap.add_argument("--urdf", default=DEFAULT_URDF, help="厂商 URDF 路径")
    ap.add_argument("--mesh-root", default=DEFAULT_DESC_DIR,
                    help="mesh 根目录（package:// 将重写到此处）")
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).parent),
                    help="产物目录（默认脚本旁）")
    ap.add_argument("--compile", action="store_true",
                    help="重写后立即用 mujoco 编译并存 MJCF")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    ready = out_dir / "s1_twin_ready.urdf"
    joints, leftover = rewrite(args.urdf, args.mesh_root, ready)

    print(f"✓ 重写完成 → {ready}")
    move_joints = [n for n, t in joints if t in ("revolute", "prismatic")]
    print(f"  可动关节 {len(move_joints)} 个: {move_joints}")
    if leftover:
        print(f"⚠ 仍残留 {leftover} 处 package:// 引用（--mesh-root 是否指对目录？）")
    else:
        print("  package:// 引用已清零")

    if not args.compile:
        print("\n本机未要求编译。下一步二选一：")
        print("  a) 有 mujoco 的机器跑: python s1_make_mjcf.py --compile ...")
        print("  b) viewer 直接吃本产物: python s1_viewer.py --model s1_twin_ready.urdf")
        print("     （装 mujoco 按惯例发命令清单给用户跑）")
        return

    out_xml = out_dir / "s1_twin.xml"
    try:
        names = compile_mjcf(ready, out_xml)
    except ImportError:
        print("✗ 本机无 mujoco，编译步未做（①产物已可用，见上）")
        sys.exit(2)
    print(f"✓ 编译成功: {len(names)} joints → {out_xml}")
    print(f"  关节清单: {names}")


if __name__ == "__main__":
    main()
