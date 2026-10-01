"""Show first half markdown cells fully"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('e:/LB5/lab5_bbc_news_029.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
    cell = nb['cells'][i]
    src = cell.get('source', '')
    if isinstance(src, list):
        src = ''.join(src)
    print(f"=== Cell {i:02d} ===")
    print(src[:500])
    print("---\n")
