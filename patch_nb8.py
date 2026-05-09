import json

with open('notebooks/face_clustering/eda_merge_ml.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'train_test_split' in src and 'stratify' in src:
        new = src.replace(
            'test_size = max(1, int(0.2 * len(df_bal)))\n'
            'X_tr, X_te, y_tr, y_te = train_test_split(\n'
            '    X_sc, y, test_size=test_size, random_state=42,\n'
            '    stratify=y if len(df_bal) >= 4 else None,\n'
            ')',
            'test_size = max(2, int(0.2 * len(df_bal)))\n'
            'X_tr, X_te, y_tr, y_te = train_test_split(\n'
            '    X_sc, y, test_size=test_size, random_state=42,\n'
            ')',
        )
        cell['source'] = [new]
        print(f'Patched cell {i}')
        break

with open('notebooks/face_clustering/eda_merge_ml.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('done')
