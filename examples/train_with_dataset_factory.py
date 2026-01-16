"""
Example: Using DatasetFactory + DataLoaderFactory for clean training code.

This shows how to easily switch between PhotoTriage and external datasets
with minimal code changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.datasets.dataset_factory import DatasetFactory
from sim_bench.datasets.dataloader_factory import DataLoaderFactory


def train_with_phototriage():
    """Example: Training with PhotoTriage data."""
    print("\n" + "="*60)
    print("Example 1: Training with PhotoTriage")
    print("="*60)

    # Configuration
    config = {
        'data_source': 'phototriage',  # <-- Just change this to switch!
        'data': {
            'root_dir': 'data/phototriage',
            'min_agreement': 0.7,
            'min_reviewers': 2,
            'quick_experiment': 0.1  # Use 10% of data for quick test
        },
        'seed': 42,
        'training': {
            'batch_size': 16
        }
    }

    # Step 1: Create datasets using DatasetFactory
    dataset_factory = DatasetFactory(source='phototriage', config=config)
    data, train_df, val_df, test_df = dataset_factory.create_datasets()

    print(f"[OK] Datasets created: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Step 2: Create dataloaders using DataLoaderFactory
    # Assume we have a model with model.preprocess
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    loader_factory = DataLoaderFactory(
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    train_loader, val_loader, test_loader = loader_factory.create_from_phototriage(
        data=data,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        transform=transform
    )

    print(f"[OK] Loaders created: {len(train_loader)} train batches")
    print("[OK] Ready to train!")

    # Step 3: Train (same code regardless of source!)
    # train_model(model, train_loader, val_loader, optimizer, config, output_dir)


def train_with_external():
    """Example: Training with external dataset (Series-Photo-Selection)."""
    print("\n" + "="*60)
    print("Example 2: Training with External Dataset")
    print("="*60)

    # Configuration - ONLY this changes!
    config = {
        'data_source': 'external',  # <-- Different source!
        'data': {
            'external_path': r'D:\Projects\Series-Photo-Selection',
            'image_root': r'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs'
        },
        'training': {
            'batch_size': 8
        }
    }

    # Step 1: Create datasets - SAME CODE!
    dataset_factory = DatasetFactory(source='external', config=config)
    train_dataset, val_dataset, test_dataset = dataset_factory.create_datasets()

    print(f"[OK] Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")

    # Step 2: Create dataloaders - SAME CODE!
    loader_factory = DataLoaderFactory(
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    train_loader, val_loader, test_loader = loader_factory.create_from_external(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    print(f"[OK] Loaders created: {len(train_loader)} train batches")
    print("[OK] Ready to train!")

    # Step 3: Train (IDENTICAL code!)
    # train_model(model, train_loader, val_loader, optimizer, config, output_dir)


def unified_training_example():
    """
    Example: Unified interface - switch sources with one config change.

    This is the cleanest approach.
    """
    print("\n" + "="*60)
    print("Example 3: Unified Training (Cleanest!)")
    print("="*60)

    # ONLY THING THAT CHANGES: data_source in config
    USE_EXTERNAL = False  # <-- Toggle this!

    config = {
        'data_source': 'external' if USE_EXTERNAL else 'phototriage',
        'data': {
            # PhotoTriage config
            'root_dir': 'data/phototriage',
            'min_agreement': 0.7,
            'min_reviewers': 2,
            # External config
            'external_path': r'D:\Projects\Series-Photo-Selection',
            'image_root': r'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs',
        },
        'seed': 42,
        'training': {
            'batch_size': 16 if not USE_EXTERNAL else 8
        }
    }

    print(f"Using source: {config['data_source']}")

    # Create datasets
    dataset_factory = DatasetFactory(source=config['data_source'], config=config)
    datasets = dataset_factory.create_datasets()

    # Create loaders based on source
    loader_factory = DataLoaderFactory(
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    if config['data_source'] == 'phototriage':
        data, train_df, val_df, test_df = datasets
        from torchvision import transforms
        transform = transforms.ToTensor()

        loaders = loader_factory.create_from_phototriage(
            data=data,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            transform=transform
        )
    else:  # external
        train_dataset, val_dataset, test_dataset = datasets
        loaders = loader_factory.create_from_external(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )

    train_loader, val_loader, test_loader = loaders

    print(f"[OK] Created {len(train_loader)} training batches")
    print("[OK] Rest of training code is IDENTICAL!")

    # Everything below is the SAME regardless of source!
    # model = create_model(config)
    # optimizer = create_optimizer(model, config)
    # train_model(model, train_loader, val_loader, optimizer, config, output_dir)
    # evaluate(model, test_loader, device, output_dir, epoch, 'test')


def main():
    """Run examples."""
    print("\n" + "="*60)
    print("DATASET FACTORY EXAMPLES")
    print("="*60)

    # Show how the pattern works (won't actually run since we don't have real data)
    print("""
These examples demonstrate the DatasetFactory pattern.

Key Benefits:
1. No manual importing of external datasets
2. No sys.path manipulation in training code
3. Switch sources by changing ONE config value
4. Same training code for all sources

Pattern:
    DatasetFactory (creates datasets)
         ↓
    DataLoaderFactory (creates loaders)
         ↓
    train_model() (same code!)
    """)

    # Uncomment to run actual examples (needs real data):
    # train_with_phototriage()
    # train_with_external()
    # unified_training_example()


if __name__ == '__main__':
    main()
