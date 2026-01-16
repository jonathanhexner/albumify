"""
Simple training example using both DatasetFactory and DataLoaderFactory.

This shows the minimal code needed to switch between data sources.
"""
import yaml

from sim_bench.datasets.dataset_factory import DatasetFactory
from sim_bench.datasets.dataloader_factory import DataLoaderFactory


def train(config_path: str):
    """
    Train a model using configuration file.

    The data source is determined by config['data_source'].
    Everything else is the same!
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Training with data source: {config.get('data_source', 'phototriage')}")

    # Step 1: Create datasets using DatasetFactory
    dataset_factory = DatasetFactory(
        source=config.get('data_source', 'phototriage'),
        config=config
    )
    datasets = dataset_factory.create_datasets()

    # Step 2: Create model (provides transform)
    from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker
    model = SiameseCNNRanker(config['model'])
    transform = model.preprocess

    # Step 3: Create loaders using DataLoaderFactory
    loader_factory = DataLoaderFactory(
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    # Handle different dataset formats
    if config.get('data_source', 'phototriage') == 'phototriage':
        data, train_df, val_df, test_df = datasets
        train_loader, val_loader, test_loader = loader_factory.create_from_phototriage(
            data, train_df, val_df, test_df, transform
        )
    else:  # external
        train_dataset, val_dataset, test_dataset = datasets
        train_loader, val_loader, test_loader = loader_factory.create_from_external(
            train_dataset, val_dataset, test_dataset
        )

    print(f"Loaders created: {len(train_loader)} train batches")

    # Step 4: Train (same code regardless of source!)
    # from sim_bench.training.train_siamese_e2e import train_model, create_optimizer
    # optimizer = create_optimizer(model, config)
    # train_model(model, train_loader, val_loader, optimizer, config, output_dir)


if __name__ == '__main__':
    # Example usage:
    # python simple_train_example.py configs/phototriage.yaml
    # python simple_train_example.py configs/external.yaml

    import sys
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        print("Usage: python simple_train_example.py <config.yaml>")
        print("\nCreate a config.yaml with:")
        print("""
# For PhotoTriage:
data_source: phototriage
data:
  root_dir: data/phototriage
  min_agreement: 0.7
  min_reviewers: 2

# OR for external:
data_source: external
data:
  external_path: D:\\Projects\\Series-Photo-Selection
  image_root: D:\\path\\to\\images

# Common:
model:
  cnn_backbone: resnet50
training:
  batch_size: 16
seed: 42
        """)
