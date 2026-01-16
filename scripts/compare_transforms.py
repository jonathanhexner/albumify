"""Compare transforms between experiments."""
import sys
import yaml
import inspect
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim_bench.datasets.transform_factory import create_transform

configs = {
    'exp1': yaml.safe_load(open('outputs/siamese_e2e/20260111_224525/config.yaml')),
    'exp2': yaml.safe_load(open('outputs/siamese_e2e/20260111_005327/config.yaml'))
}

for name, config in configs.items():
    transform = create_transform(config)
    logger.info(f"{name}: use_external={config.get('use_external_dataloader')}, "
                f"use_paper={config['model'].get('use_paper_preprocessing')}, "
                f"transform={type(transform).__name__ if transform else 'None (External)'}")
    
sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')
from data.dataloader import MyDataset
logger.info(f"\nMyDataset source:\n{inspect.getsource(MyDataset)}")
