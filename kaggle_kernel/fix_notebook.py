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

# Fix <unk> quoting
old_unk = "++tokenizer_conf.unk_symbol=<unk>"
new_unk = "++tokenizer_conf.unk_symbol='<unk>'"
src = src.replace(old_unk, new_unk)

# Fix model name: ensure iic/ prefix and quoted
src = src.replace("++model=SenseVoiceSmall", "++model='iic/SenseVoiceSmall'")
src = src.replace("++model=iic/SenseVoiceSmall", "++model='iic/SenseVoiceSmall'")
src = src.replace("++model=''iic/SenseVoiceSmall''", "++model='iic/SenseVoiceSmall'")
src = src.replace("++model='iic/SenseVoiceSmall'", "++model='iic/SenseVoiceSmall'")

nb['cells'][4]['source'] = src

# Verify
assert "++tokenizer_conf.unk_symbol='<unk>'" in src, "unk fix failed"
assert "++model='iic/SenseVoiceSmall'" in src, "model fix failed"
print("\n--- Verification ---")
print("unk_symbol quoted:", "++tokenizer_conf.unk_symbol='<unk>'" in src)
print("model quoted:", "++model='iic/SenseVoiceSmall'" in src)

# Save with standard notebook indent (1 space)
f = open(r'C:\asr\kaggle_kernel\asr-v3-train.ipynb', 'w', encoding='utf-8')
json.dump(nb, f, ensure_ascii=False, indent=1)
f.close()
print("\nSaved OK")
