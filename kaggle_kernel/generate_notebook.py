"""
重新生成完整 notebook —— 不降级 numpy，不在主进程 import funasr/torch
"""
import json

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": []
}

def md(s):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": s})

def code(s):
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})


# ===== Header =====
md("""# SenseVoice V3 微调训练 — Kaggle 版

基于官方 [FunASR finetune.sh](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/sense_voice/finetune.sh)。

| 项目 | 值 |
|------|----|
| 数据 | `shadiao/asr0001` — 3720 条增强数据 |
| 预训练模型 | `vinhtrannhat/sensevoice-small-model`（本地挂载） |
| GPU | P100 16GB / T4 |
| 训练 | 50 epochs, lr=2e-5, token batch=2000 |
| 导出 | ONNX FP32 全精度 |

**设置：** Accelerator → GPU, Internet → ON

**⚠️ 必须从头 Run All。**
""")


# ===== Cell 1: 安装 =====
code(r"""# ============================================================
# Cell 1: 安装依赖
# ============================================================
# 策略：
#   - torch 降级到 2.2.0+cu118（支持 P100 sm_60）
#   - numpy 不降级！保持 Kaggle 预装的 2.x
#     (降级会导致 numba 等 C 扩展 "dtype size changed" 崩溃)
#   - torch 对 numpy 2.x 的警告是 UserWarning，不影响训练

import subprocess, sys

def run(cmd):
    print(f">>> {cmd}")
    r = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if r.stdout:
        # 只打最后几行，避免刷屏
        lines = r.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(line)
    if r.returncode != 0 and r.stderr:
        print("WARN:", r.stderr.strip()[-200:])

# 1) 卸载旧 torch
run("pip uninstall -y torch torchvision torchaudio -q 2>/dev/null")

# 2) 安装 cu118 版 PyTorch
run("pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 "
    "--index-url https://download.pytorch.org/whl/cu118 --no-cache-dir -q")

# 3) FunASR (--no-deps 避免拉入不兼容 torch)
run("pip install funasr==1.3.9 --no-deps -q")

# 4) FunASR 运行时依赖
run("pip install soundfile librosa hydra-core kaldiio modelscope tensorboardX torch_complex -q")

# 5) ONNX
run("pip install onnx onnxruntime -q")

# ⚠️ 不降级 numpy —— 保持 2.x

# 验证（子进程，不影响主进程模块状态）
print("\n" + "="*60)
print("验证安装...")
print("="*60)

verify_code = (
    "import warnings; warnings.filterwarnings('ignore'); "
    "import torch; import numpy as np; "
    "print(f'PyTorch:  {torch.__version__}'); "
    "print(f'NumPy:    {np.__version__}'); "
    "print(f'CUDA:     {torch.cuda.is_available()}'); "
    "assert torch.cuda.is_available(), 'CUDA not available'; "
    "t = torch.randn(2, 2, device='cuda'); "
    "print(f'GPU:      {torch.cuda.get_device_name(0)}'); "
    "print('OK')"
)

ok = subprocess.run([sys.executable, "-c", verify_code],
                    capture_output=True, text=True)
if ok.returncode == 0:
    print(ok.stdout)
else:
    print("STDOUT:", ok.stdout)
    print("STDERR:", ok.stderr[-500:])
    raise RuntimeError("环境验证失败，停止执行")
""")


# ===== Cell 2: 数据准备 =====
code(r"""# ============================================================
# Cell 2: 准备数据和模型
# ============================================================
import os, json, shutil

# ---------- 2.1 查找数据集 ----------
input_root = '/kaggle/input'
print(f"扫描 {input_root} ...")

data_dir = None
for root, dirs, files in os.walk(input_root):
    if 'augmented_audio' in dirs:
        data_dir = root
        break
    if 'train.jsonl' in files and 'val.jsonl' in files:
        data_dir = root
        break

if not data_dir:
    print("可用目录：")
    for root, dirs, files in os.walk(input_root):
        for d in dirs:
            print(f"  {os.path.join(root, d)}/")
    raise FileNotFoundError(f"在 {input_root} 中未找到数据集")
print(f"✅ 数据目录: {data_dir}")

# ---------- 2.2 Symlink 音频 ----------
input_audio = os.path.join(data_dir, 'augmented_audio')
assert os.path.isdir(input_audio), f"augmented_audio 不存在: {input_audio}"

work_dir = '/kaggle/working/data'
work_audio = os.path.join(work_dir, 'augmented_audio')
os.makedirs(work_dir, exist_ok=True)
if os.path.islink(work_audio):
    os.unlink(work_audio)
elif os.path.exists(work_audio):
    shutil.rmtree(work_audio)
os.symlink(input_audio, work_audio)

wav_count = len([f for f in os.listdir(input_audio) if f.endswith('.wav')])
print(f"✅ 音频 symlink: {wav_count} 个 wav")

# ---------- 2.3 复制 JSONL ----------
for fn in ['train.jsonl', 'val.jsonl']:
    src = os.path.join(data_dir, fn)
    dst = os.path.join(work_dir, fn)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        n = sum(1 for _ in open(dst))
        print(f"✅ {fn}: {n} 条")
    else:
        print(f"⚠️ {fn} 不存在")

# ---------- 2.4 Symlink 模型 ----------
model_src = None
for d in os.listdir(input_root):
    dp = os.path.join(input_root, d)
    if os.path.isfile(os.path.join(dp, 'model.pt')):
        model_src = dp
        break

cache_dir = os.path.expanduser('~/.cache/modelscope/hub/models/iic/SenseVoiceSmall')
if model_src:
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    if os.path.islink(cache_dir):
        os.unlink(cache_dir)
    elif os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    # 必须 copy 而非 symlink！/kaggle/input 是只读的，
    # ModelScope 需要在目录里写 .mdl 元数据文件
    shutil.copytree(model_src, cache_dir)
    print(f"✅ 模型已复制: {model_src} → {cache_dir}")
    total_mb = sum(os.path.getsize(os.path.join(cache_dir, f))
                   for f in os.listdir(cache_dir)
                   if os.path.isfile(os.path.join(cache_dir, f))) // 1024 // 1024
    print(f"   文件: {os.listdir(cache_dir)} ({total_mb} MB)")
else:
    print("⚠️ 未找到本地模型，将从 ModelScope 下载")

free_mb = shutil.disk_usage('/kaggle/working').free // 1024 // 1024
print(f"\n磁盘剩余: {free_mb} MB")
""")


# ===== Cell 3: 修正 JSONL =====
code(r"""# ============================================================
# Cell 3: 修正 JSONL 音频路径 + 过滤无效条目
# ============================================================
import json, os
import soundfile as sf

work_dir = '/kaggle/working/data'
total_orig = 0
total_valid = 0

for fn in ['train.jsonl', 'val.jsonl']:
    fpath = os.path.join(work_dir, fn)
    if not os.path.exists(fpath):
        print(f"跳过 {fn}")
        continue

    lines = open(fpath, encoding='utf-8').readlines()
    total_orig += len(lines)
    valid = []
    skipped = 0

    for line in lines:
        rec = json.loads(line.strip())
        basename = os.path.basename(rec['source'])
        rec['source'] = os.path.join(work_dir, 'augmented_audio', basename)

        if not os.path.exists(rec['source']):
            skipped += 1
            continue

        try:
            with sf.SoundFile(rec['source']) as f:
                duration = f.frames / f.samplerate
                rec['source_len'] = max(1, int(duration * 100 / 6))
        except Exception as e:
            print(f"  跳过: {basename} ({e})")
            skipped += 1
            continue

        valid.append(rec)

    with open(fpath, 'w', encoding='utf-8') as f:
        for rec in valid:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    total_valid += len(valid)
    print(f"✅ {fn}: {len(lines)} → {len(valid)} 条 (过滤 {skipped})")

print(f"\n总计: {total_orig} → {total_valid} 条")
""")


# ===== Cell 4: 训练 =====
code(r"""# ============================================================
# Cell 4: 训练（严格对齐官方 finetune.sh）
# ============================================================
# 官方只需传模型名 + 数据路径 + 训练超参
# 架构参数 (encoder/frontend/tokenizer/specaug) 由模型 config.yaml 自动提供
#
# ⚠️ 不在主进程 import funasr/torch —— 全部用子进程
#   避免 numpy 2.x 与 torch 2.2.0 的 import 警告影响 notebook

import os, sys, shutil, subprocess, glob

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 用子进程获取 train_ds.py 路径
r = subprocess.run(
    [sys.executable, '-c',
     'import funasr, os; print(os.path.join(os.path.dirname(funasr.__file__), "bin", "train_ds.py"))'],
    capture_output=True, text=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-500:])
    raise RuntimeError("无法定位 funasr")
train_script = r.stdout.strip()

train_data = '/kaggle/working/data/train.jsonl'
val_data = '/kaggle/working/data/val.jsonl'
output_dir = '/kaggle/working/outputs'

print(f"Script:  {train_script}")
print(f"Train:   {train_data}")
print(f"Val:     {val_data}")
print(f"Output:  {output_dir}")
free_mb = shutil.disk_usage('/kaggle/working').free // 1024 // 1024
print(f"磁盘:    {free_mb} MB")

cmd = [
    sys.executable, train_script,
    '++model=iic/SenseVoiceSmall',
    f'++train_data_set_list={train_data}',
    f'++valid_data_set_list={val_data}',
    '++dataset_conf.data_split_num=1',
    '++dataset_conf.batch_sampler=BatchSampler',
    '++dataset_conf.batch_size=2000',
    '++dataset_conf.sort_size=1024',
    '++dataset_conf.batch_type=token',
    '++dataset_conf.num_workers=4',
    '++train_conf.max_epoch=50',
    '++train_conf.log_interval=1',
    '++train_conf.resume=true',
    '++train_conf.validate_interval=500',
    '++train_conf.save_checkpoint_interval=500',
    '++train_conf.keep_nbest_models=3',     # 省磁盘（3×2.5GB=7.5GB）
    '++train_conf.avg_nbest_model=3',
    '++train_conf.use_deepspeed=false',
    '++optim_conf.lr=0.00002',
    '++scheduler_conf.warmup_steps=100',    # 关键！原值 25000 是为大数据集设计的
                                            # 19步/epoch，100步≈5个epoch预热完
    f'++output_dir={output_dir}',
]

print("\n" + "="*60)
print("开始训练...")
print("="*60)

with open('/kaggle/working/train.log', 'w') as log:
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

print(f"\nExit code: {proc.returncode}")

with open('/kaggle/working/train.log') as f:
    lines = f.readlines()
    print(f"日志共 {len(lines)} 行，最后 60 行:")
    print("".join(lines[-60:]))

if proc.returncode != 0:
    print("\n===== 前 30 行 =====")
    print("".join(lines[:30]))
    raise RuntimeError(f"训练失败 (exit code {proc.returncode})")

# 清理 checkpoint
if os.path.exists(output_dir):
    avg_dirs = [d for d in os.listdir(output_dir) if d.startswith('avg_')]
    print(f"\navg 目录: {avg_dirs}")
    for item in os.listdir(output_dir):
        p = os.path.join(output_dir, item)
        if os.path.isdir(p) and item.startswith('checkpoint_'):
            shutil.rmtree(p)
            print(f"  删除 {item}")
    for f in glob.glob(f'{output_dir}/avg_*/optimizer*'):
        os.remove(f)
        print(f"  删除 {os.path.basename(f)}")
    free_mb = shutil.disk_usage('/kaggle/working').free // 1024 // 1024
    print(f"清理后: {free_mb} MB")
""")


# ===== Cell 5: 导出 ONNX =====
code(r"""# ============================================================
# Cell 5: 导出 ONNX（FP32 全精度）
# ============================================================
import os, sys, shutil, subprocess

free_mb = shutil.disk_usage('/kaggle/working').free // 1024 // 1024
print(f"导出前磁盘: {free_mb} MB")

output_dir = '/kaggle/working/outputs'
avg_dir = None
for d in sorted(os.listdir(output_dir)):
    if d.startswith('avg_'):
        avg_dir = os.path.join(output_dir, d)
if not avg_dir:
    raise FileNotFoundError("未找到 avg 模型目录")
print(f"模型: {avg_dir}")
print(f"文件: {os.listdir(avg_dir)}")

export_dir = '/kaggle/working/export'

r = subprocess.run(
    [sys.executable, '-c',
     'import funasr, os; print(os.path.join(os.path.dirname(funasr.__file__), "bin", "export.py"))'],
    capture_output=True, text=True)
export_script = r.stdout.strip()

cmd = [
    sys.executable, export_script,
    f'++model={avg_dir}',
    f'++output_dir={export_dir}',
    '++type=onnx',
    '++quantize=false',
    '++device=cpu',
]

print(f"命令: {' '.join(cmd)}")
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
if proc.stdout: print(proc.stdout[-2000:])
if proc.stderr: print("STDERR:", proc.stderr[-1000:])
print(f"Exit code: {proc.returncode}")

onnx_path = os.path.join(export_dir, 'model.onnx')
if os.path.exists(onnx_path):
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\n✅ model.onnx: {size_mb:.0f} MB (FP32)")
    shutil.copy2(onnx_path, '/kaggle/working/model.onnx')
else:
    if os.path.exists(export_dir):
        print(f"目录内容: {os.listdir(export_dir)}")
    raise RuntimeError("ONNX 导出失败")
""")


# ===== Cell 6: ONNX 后处理 =====
code(r"""# ============================================================
# Cell 6: ONNX 后处理（FixCast + 元数据注入）
# ============================================================
import os, subprocess, sys

# 用子进程执行，避免主进程 import torch 的 numpy 警告
postprocess_code = '''
import warnings
warnings.filterwarnings("ignore")

import onnx, torch, os
import numpy as np
from onnx import helper, TensorProto
from onnx.external_data_helper import load_external_data_for_model

onnx_path = "/kaggle/working/model.onnx"
model = onnx.load(onnx_path, load_external_data=False)
load_external_data_for_model(model, os.path.dirname(onnx_path))
graph = model.graph

# FixCast: Less 节点输入转 float
less_nodes = [n for n in graph.node if n.op_type == "Less"]
for less_node in less_nodes:
    for i in range(2):
        orig = less_node.input[i]
        cast_out = orig + "_float"
        cast = helper.make_node("Cast", inputs=[orig], outputs=[cast_out], to=TensorProto.FLOAT)
        less_node.input[i] = cast_out
        idx = list(graph.node).index(less_node)
        graph.node.insert(idx, cast)
print(f"Fixed {len(less_nodes)} Less nodes")

# Embedding 类型判断
output_dir = "/kaggle/working/outputs"
avg_dir = None
for d in sorted(os.listdir(output_dir)):
    if d.startswith("avg_"):
        avg_dir = os.path.join(output_dir, d)

state = torch.load(f"{avg_dir}/model.pt", map_location="cpu", weights_only=True)
state_dict = state.get("model_state_dict", state)
embed = state_dict.get("embed.weight", None)
is_reduced = embed is not None and embed.shape[0] < 100
print(f"Embedding: {embed.shape if embed is not None else '?'} -> {'reduced' if is_reduced else 'full'}")

# 元数据
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

cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models/iic/SenseVoiceSmall")
mvn_path = os.path.join(cache_dir, "am.mvn")
if os.path.exists(mvn_path):
    mvn = np.load(mvn_path, allow_pickle=True)
    mvn_item = mvn.item() if mvn.dtype == np.object_ else mvn[0]
    metadata["neg_mean"] = ",".join(f"{x:.8f}" for x in mvn_item["neg_mean"].flatten())
    metadata["inv_stddev"] = ",".join(f"{x:.8f}" for x in mvn_item["inv_stddev"].flatten())
    print("CMVN injected")

for key, val in metadata.items():
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = val
print(f"Injected {len(metadata)} metadata props")

final_path = "/kaggle/working/model_v3_deploy.onnx"
onnx.save(model, final_path)
size_mb = os.path.getsize(final_path) / 1024 / 1024
print(f"model_v3_deploy.onnx: {size_mb:.0f} MB (FP32)")
print("DONE")
'''

proc = subprocess.run([sys.executable, '-c', postprocess_code],
                      capture_output=True, text=True, timeout=300)
print(proc.stdout)
if proc.returncode != 0:
    print("STDERR:", proc.stderr[-1000:])
    raise RuntimeError("ONNX 后处理失败")

size_mb = os.path.getsize('/kaggle/working/model_v3_deploy.onnx') / 1024 / 1024
print(f"\n✅ 最终模型: model_v3_deploy.onnx ({size_mb:.0f} MB, FP32)")
""")


# ===== Footer =====
md("""## ✅ 完成！

最终模型: `/kaggle/working/model_v3_deploy.onnx` (FP32 全精度)

**下载方式：**
1. Save & Run All (Commit) → Output 标签页下载
2. 直接 Run All → 右侧 Data → Output → 下载
""")


# ===== 保存 + 验证 =====
out_path = r'C:\asr\kaggle_kernel\asr-v3-train.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

with open(out_path, 'r', encoding='utf-8') as f:
    check = json.load(f)

print(f"生成 {len(check['cells'])} 个 cells:")
for i, c in enumerate(check['cells']):
    t = c['cell_type']
    s = c['source']
    assert isinstance(s, str), f"Cell {i} source is {type(s)}!"
    first = s.strip()[:40].replace('\n', ' ')
    print(f"  Cell {i}: {t:8s} | {len(s):5d} chars | {first}")

# 确认没有 numpy 降级
full = json.dumps(check)
assert 'numpy==' not in full, "仍然包含 numpy 降级！"
assert 'numpy<2' not in full, "仍然包含 numpy<2！"
print("\n✅ 无 numpy 降级指令")
print("✅ 格式验证通过")
