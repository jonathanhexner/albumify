# DataLoader Comparison

Dump dataloaders from both experiments to CSV for series_id comparison.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/dump_dataloader_to_csv.py
```

## Output

`outputs/dataloader_comparison/`
- `exp1_train.csv` - Exp1 train pairs (PhotoTriageData)
- `exp1_val.csv` - Exp1 val pairs
- `exp2_train.csv` - Exp2 train pairs (External MyDataset)  
- `exp2_val.csv` - Exp2 val pairs

## CSV Format

```
image1,image2,series_id1,series_id2,winner
000001-01.JPG,000001-02.JPG,1,1,0
```

## Analysis

Check if series_id appears in both train and val:
- `series_id1` and `series_id2` extracted from filenames
- Should have 0 overlap between train/val for proper splitting
