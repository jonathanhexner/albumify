# DataLoader Comparison - Simple CSV Dump

## What Was Created

**`scripts/dump_dataloader_to_csv.py`** - Clean 60-line script that:
- Loads both experiment configs
- Recreates dataloaders exactly as used in training  
- Iterates through all batches
- Saves all pairs to CSV with series_id extracted

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/dump_dataloader_to_csv.py
```

## Output

`outputs/dataloader_comparison/`
```
exp1_train.csv  - PhotoTriageData train loader
exp1_val.csv    - PhotoTriageData val loader
exp2_train.csv  - External MyDataset train loader
exp2_val.csv    - External MyDataset val loader
```

Each CSV:
```
image1,image2,series_id1,series_id2,winner
000001-01.JPG,000001-02.JPG,1,1,0
```

## Analysis

Compare series_id between train and val:
1. Load CSVs
2. Get unique series_id from exp1_train.csv (union of series_id1 and series_id2)
3. Get unique series_id from exp1_val.csv
4. Check overlap: should be 0 for proper splitting
5. Repeat for exp2

**Expected:** PhotoTriageData (exp1) should have 0 overlap. External MyDataset (exp2) TBD.
