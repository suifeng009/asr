import json

f = open(r'C:\asr\kaggle_kernel\asr-v3-train.ipynb', 'r', encoding='utf-8')
nb = json.load(f)
f.close()

# Fix ALL cells: if source is a list of single chars, join them back into a string
for i, cell in enumerate(nb['cells']):
    src = cell['source']
    if isinstance(src, list):
        # Check if it's a list of single characters (the bug)
        if all(isinstance(s, str) and len(s) <= 1 for s in src):
            joined = ''.join(src)
            cell['source'] = joined
            print(f'Cell {i}: FIXED char-list -> string ({len(src)} chars)')
        else:
            # Normal list of lines, join them
            cell['source'] = ''.join(src)
            print(f'Cell {i}: joined line-list -> string ({len(src)} lines)')
    else:
        print(f'Cell {i}: OK (already string, {len(src)} chars)')

# Now apply Hydra fixes to the training cell (cell 4)
src = nb['cells'][4]['source']

# Fix <unk> quoting if present
old_unk = "++tokenizer_conf.unk_symbol=<unk>"
new_unk = "++tokenizer_conf.unk_symbol='<unk>'"
if old_unk in src:
    src = src.replace(old_unk, new_unk)

# Fix model name: ensure iic/ prefix, and keep it valid syntax
# We no longer force single quotes around model name inside the python string to avoid SyntaxError.
# '++model=iic/SenseVoiceSmall' is already perfectly valid.
src = src.replace("++model=SenseVoiceSmall", "++model=iic/SenseVoiceSmall")
src = src.replace("++model='iic/SenseVoiceSmall'", "++model=iic/SenseVoiceSmall")
src = src.replace("++model=''iic/SenseVoiceSmall''", "++model=iic/SenseVoiceSmall")

nb['cells'][4]['source'] = src

# Verify
if old_unk in src or new_unk in src:
    assert "++tokenizer_conf.unk_symbol='<unk>'" in src, "unk fix failed"
assert "++model=iic/SenseVoiceSmall" in src, "model fix failed"
print("\n--- Verification ---")
print("unk_symbol quoted:", "++tokenizer_conf.unk_symbol='<unk>'" in src if (old_unk in src or new_unk in src) else "Not present")
print("model setting:", "++model=iic/SenseVoiceSmall" in src)

# Save with standard notebook indent (1 space)
f = open(r'C:\asr\kaggle_kernel\asr-v3-train.ipynb', 'w', encoding='utf-8')
json.dump(nb, f, ensure_ascii=False, indent=1)
f.close()
print("\nSaved OK")
