"""
Train the reference model with per-batch state dumps for comparison.

This script wraps the reference model training code with minimal changes,
just adding model state dumps after each batch.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add reference code to path
sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')

from models.ResNet50 import make_network
from models.lr_scheduler import LR_Scheduler
from sklearn import metrics


def make_loader(batch_size):
    """Import and call their make_loader function."""
    # You'll need to implement this based on their data loading code
    # For now, placeholder
    raise NotImplementedError("Need to import their data loading code")


def train_reference_with_dumps(output_dir, batch_dump_interval=10):
    """
    Train reference model with batch dumps.

    This is their training code with minimal modification:
    just adds model dumps every N batches.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === THEIR CODE STARTS HERE (unchanged) ===
    best_pred = 0.0
    best_acc = 0.0
    best_macro = 0.0
    best_micro = 0.0
    lr = 0.00001
    num_epochs = 100
    batch_size = 8  # Reduced for CPU training

    train_data, val_data, trainloader, valloader = make_loader(batch_size=batch_size)
    model = make_network()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")  # Force CPU usage
    print(f"Using device: {device}")
    model.to(device)
    criterion.to(device)

    train_params = [
        {'params': model.get_1x_lr_params(), 'lr': lr},
        {'params': model.get_10x_lr_params(), 'lr': lr * 10}
    ]
    optimizer = optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = LR_Scheduler(mode='step', base_lr=lr, num_epochs=num_epochs,
                            iters_per_epoch=len(trainloader), lr_step=25)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        acc = 0.0
        micro = 0.0
        macro = 0.0
        count = 0
        model.train()

        for batch_idx, (dataA, dataB, target) in enumerate(trainloader):
            dataA, dataB, target = dataA.to(device), dataB.to(device), target.to(device)
            scheduler(optimizer, batch_idx, epoch, best_pred)
            optimizer.zero_grad()
            pred = model(dataA, dataB)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            predict = torch.argmax(pred, 1)
            a = metrics.accuracy_score(target.cpu(), predict.cpu())
            b = metrics.f1_score(target.cpu(), predict.cpu(), average='micro')
            c = metrics.f1_score(target.cpu(), predict.cpu(), average='macro')
            acc += a
            micro += b
            macro += c
            count += 1

            correct = torch.eq(predict, target).sum().double().item()
            running_loss += loss.item()
            running_correct += correct
            running_total += target.size(0)

            # === OUR ADDITION: Dump model state ===
            if (batch_idx + 1) % batch_dump_interval == 0:
                dump_dir = output_dir / "batch_dumps" / f"epoch_{epoch:03d}"
                dump_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'acc': a
                }, dump_dir / f"batch_{batch_idx+1:04d}.pt")
                print(f"Dumped model state: epoch {epoch}, batch {batch_idx+1}")
            # === END ADDITION ===

        loss = running_loss * batch_size / running_total
        # Rest of their epoch-end code would go here...
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc/count:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Where to save batch dumps')
    parser.add_argument('--batch_dump_interval', type=int, default=10,
                       help='Dump model every N batches')
    args = parser.parse_args()

    train_reference_with_dumps(args.output_dir, args.batch_dump_interval)
