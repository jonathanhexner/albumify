"""Dump dataloader pairs to CSV for comparison."""
import sys
import yaml
from pathlib import Path
import pandas as pd
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.datasets.transform_factory import create_transform


def extract_series_id(filename: str) -> int:
    """Extract series_id from '000123-02.JPG' -> 123"""
    return int(filename.split('-')[0])


def dump_loader_to_csv(loader, output_path: Path):
    """Iterate loader and save all pairs to CSV."""
    pairs = []
    for batch in loader:
        for i in range(len(batch['image1'])):
            pairs.append({
                'image1': batch['image1'][i],
                'image2': batch['image2'][i],
                'series_id1': extract_series_id(batch['image1'][i]),
                'series_id2': extract_series_id(batch['image2'][i]),
                'winner': int(batch['winner'][i])
            })
    pd.DataFrame(pairs).to_csv(output_path, index=False)
    return len(pairs)


def create_loader(config_path: Path, split: str):
    """Create train or val loader from experiment config."""
    config = yaml.safe_load(open(config_path))
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size = config['training']['batch_size']
    factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=seed)
    
    use_external = config.get('use_external_dataloader', False)
    if use_external:
        sys.path.insert(0, config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection'))
        from data.dataloader import MyDataset
        image_root = config['data'].get('image_root') or config['data']['root_dir']
        train_ds = MyDataset(train=True, image_root=image_root, seed=seed)
        val_ds = MyDataset(train=False, image_root=image_root, seed=seed)
        loaders = factory.create_from_external(train_ds, val_ds, None)
        return loaders[0] if split == 'train' else loaders[1]
    
    data = PhotoTriageData(config['data']['root_dir'], config['data']['min_agreement'], config['data']['min_reviewers'])
    train_df, val_df, test_df = data.get_series_based_splits(0.8, 0.1, 0.1, seed, config['data'].get('quick_experiment'))
    loaders = factory.create_from_phototriage(data, train_df, val_df, test_df, create_transform(config))
    return loaders[0] if split == 'train' else loaders[1]


output_dir = Path('outputs/dataloader_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

for exp_name, exp_path in [('exp1', 'outputs/siamese_e2e/20260111_224525'), ('exp2', 'outputs/siamese_e2e/20260111_005327')]:
    for split in ['train', 'val']:
        loader = create_loader(Path(exp_path) / 'config.yaml', split)
        n = dump_loader_to_csv(loader, output_dir / f'{exp_name}_{split}.csv')
        print(f"{exp_name}_{split}: {n} pairs")
