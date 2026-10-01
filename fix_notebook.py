"""Fix the notebook cell that uses .values to use .tolist() and np.array() instead."""
import json

with open('lab5_bbc_news_029.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed = 0
for i, cell in enumerate(nb['cells']):
    src = cell.get('source', '')
    if "df['cleaned_text'].astype(str).tolist()" in src:
        print(f"Cell {i}: Already fixed!")
        fixed += 1
    elif "df['cleaned_text'].values" in src or "df['label'].values" in src:
        print(f"Cell {i}: FOUND old code, fixing...")
        src = src.replace(
            "texts = df['cleaned_text'].values\nlabels = df['label'].values",
            "texts = df['cleaned_text'].astype(str).tolist()\nlabels = np.array(df['label'].tolist(), dtype=int)"
        )
        cell['source'] = src
        fixed += 1
        print(f"  -> Fixed!")

if fixed == 0:
    # Try multiline source (list format)
    for i, cell in enumerate(nb['cells']):
        if isinstance(cell.get('source'), list):
            for j, line in enumerate(cell['source']):
                if "df['cleaned_text'].values" in line:
                    print(f"Cell {i}, line {j}: FOUND old code (list format), fixing...")
                    cell['source'][j] = line.replace(
                        "df['cleaned_text'].values",
                        "df['cleaned_text'].astype(str).tolist()"
                    )
                    fixed += 1
                if "df['label'].values" in line:
                    print(f"Cell {i}, line {j}: FOUND old label code (list format), fixing...")
                    cell['source'][j] = line.replace(
                        "df['label'].values",
                        "np.array(df['label'].tolist(), dtype=int)"
                    )
                    fixed += 1

with open('lab5_bbc_news_029.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nTotal fixes: {fixed}")
print("Notebook saved!")
