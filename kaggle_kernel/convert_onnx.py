import os
import sys
import shutil
import subprocess

# 1. 检查目录是否准备好
output_dir = r"C:\asr\kaggle_output\outputs"
export_dir = r"C:\asr\kaggle_kernel\export"
os.makedirs(export_dir, exist_ok=True)

if not os.path.exists(output_dir):
    print(f"[ERROR] 找不到目录: {output_dir}")
    print("请先去 Kaggle 页面，把 outputs 文件夹里的内容下载并解压到这个目录！")
    print("至少需要下载: model.pt.best 和 config.yaml")
    sys.exit(1)

best_model = os.path.join(output_dir, 'model.pt.best')
if not os.path.exists(best_model):
    print(f"[ERROR] 找不到最佳模型权重: {best_model}")
    print("请确认你从 Kaggle 下载了 model.pt.best 并放到了正确位置。")
    sys.exit(1)

# 2. 准备模型结构以适配导出脚本
model_pt = os.path.join(output_dir, 'model.pt')
if not os.path.exists(model_pt):
    shutil.copy2(best_model, model_pt)
    print("[OK] 已将 model.pt.best 复制为 model.pt 以供导出")

# 3. 执行 FunASR 导出
print("[RUN] 开始转换 ONNX...")
import funasr
export_script = os.path.join(os.path.dirname(funasr.__file__), "bin", "export.py")

cmd = [
    sys.executable, export_script,
    f'++model={output_dir}',
    f'++output_dir={export_dir}',
    '++type=onnx',
    '++quantize=false',
    '++device=cpu',
]

proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    print("[ERROR] 转换报错:")
    print(proc.stderr)
else:
    print("[OK] 初始转换成功！")

# 4. 后处理 (注入元数据和修复节点)
print("[RUN] 开始 ONNX 后处理...")
try:
    import onnx
    import torch
    import numpy as np
    from onnx import helper, TensorProto
    from onnx.external_data_helper import load_external_data_for_model
except ImportError:
    print("[ERROR] 缺少必要的依赖，请先执行: pip install onnx torch numpy")
    sys.exit(1)

onnx_path = os.path.join(export_dir, 'model.onnx')
if not os.path.exists(onnx_path):
    print(f"[ERROR] 找不到初始转换的 ONNX 文件: {onnx_path}")
    sys.exit(1)

model = onnx.load(onnx_path, load_external_data=False)
load_external_data_for_model(model, os.path.dirname(onnx_path))
graph = model.graph

# 修复 Less 节点
less_nodes = [n for n in graph.node if n.op_type == "Less"]
for less_node in less_nodes:
    for i in range(2):
        orig = less_node.input[i]
        cast_out = orig + "_float"
        cast = helper.make_node("Cast", inputs=[orig], outputs=[cast_out], to=TensorProto.FLOAT)
        less_node.input[i] = cast_out
        idx = list(graph.node).index(less_node)
        graph.node.insert(idx, cast)
print(f"[OK] 修复了 {len(less_nodes)} 个 Less 节点")

# 读取原始权重检测 Embedding
state = torch.load(best_model, map_location="cpu", weights_only=True)
state_dict = state.get("model_state_dict", state)
embed = state_dict.get("embed.weight", None)
is_reduced = embed is not None and embed.shape[0] < 100
print(f"[OK] 词表检测: {'Reduced (微调极简版)' if is_reduced else 'Full (完整版)'}")

metadata = {
    "vocab_size": "25055",
    "lfr_window_size": "7",
    "lfr_window_shift": "6",
    "normalize_samples": "True",
    "with_itn": "14" if is_reduced else "25016",
    "without_itn": "15" if is_reduced else "25017",
    "lang_zh": "3" if is_reduced else "24884",
    "lang_en": "4" if is_reduced else "24885",
    "lang_ja": "11" if is_reduced else "24892",
    "lang_ko": "12" if is_reduced else "24896",
    "lang_yue": "7" if is_reduced else "24888",
}

for key, val in metadata.items():
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = val

final_path = r"C:\asr\kaggle_kernel\model_v3_deploy.onnx"
onnx.save(model, final_path)
size_mb = os.path.getsize(final_path) / 1024 / 1024
print(f"\n[DONE] 大功告成！最终部署模型: {final_path} ({size_mb:.0f} MB)")
