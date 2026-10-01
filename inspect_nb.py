import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('lab5_bbc_news_029.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    if ctype == 'markdown':
        lines = [l.strip() for l in ''.join(c['source']).split('\n') if l.strip()]
        h = [l for l in lines if l.startswith('#')]
        title = h[0] if h else (lines[0][:60] if lines else 'empty')
        print(f"Cell {i:2d} [MD]  : {title}")
    else:
        src = ''.join(c['source']).split('\n')
        code_lines = [l.strip() for l in src if l.strip() and not l.strip().startswith('#')]
        summary = code_lines[0][:60] if code_lines else 'comments only'
        print(f"Cell {i:2d} [CODE]: {summary}")
