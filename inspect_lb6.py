"""Check full cell 16 including evaluate"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('e:/LB6/lab6_timeseries_029.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][16]
src = cell.get('source', '')
if isinstance(src, list):
    src = ''.join(src)
print(src)
