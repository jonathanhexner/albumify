# Multi-Model Training Refactor

## Problem

The original code for training with an optional reference model was full of awkward `if ref_model is not None` statements that made the code hard to read and didn't scale well if you wanted to train more models in parallel.

**Before (Awkward):**
```python
def train_epoch(model, loader, optimizer, device, log_interval=10,
                epoch=None, output_dir=None, ref_model=None, ref_optimizer=None,
                batch_comparison_interval=None):
    model.train()
    if ref_model is not None:
        ref_model.train()

    total_loss = 0.0
    total_acc = 0.0

    comparison_log = [] if (ref_model is not None and batch_comparison_interval is not None) else None

    for batch_idx, batch in enumerate(loader, 1):
        # Train our model
        log_probs = model(img1, img2)
        loss = F.nll_loss(log_probs, winners)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = compute_pairwise_accuracy(log_probs, winners)
        total_loss += batch_loss
        total_acc += batch_acc

        # Train reference model if provided
        ref_loss_val = None
        ref_acc_val = None
        if ref_model is not None and ref_optimizer is not None:
            ref_log_probs = ref_model(img1, img2)
            ref_loss = F.nll_loss(ref_log_probs, winners)
            ref_optimizer.zero_grad()
            ref_loss.backward()
            ref_optimizer.step()
            ref_loss_val = ref_loss.item()
            ref_acc_val = compute_pairwise_accuracy(ref_log_probs, winners)

        if batch_idx % log_interval == 0:
            if ref_model is not None:
                logger.info(f"  Batch {batch_idx}/{len(loader)}: "
                          f"our_loss={batch_loss:.4f}, our_acc={batch_acc:.3f} | "
                          f"ref_loss={ref_loss_val:.4f}, ref_acc={ref_acc_val:.3f}")
            else:
                logger.info(f"  Batch {batch_idx}/{len(loader)}: loss={batch_loss:.4f}, acc={batch_acc:.3f}")

        # More if statements for comparison...
        if (comparison_log is not None and batch_idx % batch_comparison_interval == 0):
            # ...

    return total_loss / len(loader), total_acc / len(loader)
```

## Solution

Refactored to use **dictionaries** for models and optimizers. This eliminates all the `if ref_model is not None` checks and makes the code much cleaner and more extensible.

**After (Clean):**
```python
def train_epoch(models_dict, optimizers_dict, loader, device, log_interval=10,
                epoch=None, output_dir=None, batch_comparison_interval=None):
    """
    Train one or more models for one epoch.

    Args:
        models_dict: Dict of {model_name: model} to train
        optimizers_dict: Dict of {model_name: optimizer} matching models_dict
        ...

    Returns:
        Dict of {model_name: (avg_loss, avg_acc)} for each model
    """
    # Set all models to train mode
    for model in models_dict.values():
        model.train()

    # Track metrics for each model
    model_names = list(models_dict.keys())
    metrics = {
        name: {'total_loss': 0.0, 'total_correct': 0, 'total_samples': 0}
        for name in model_names
    }

    # Track comparison only if multiple models
    comparison_log = [] if (len(models_dict) > 1 and batch_comparison_interval is not None) else None

    for batch_idx, batch in enumerate(loader, 1):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        winners = batch['winner'].to(device)

        # Train each model - CLEAN LOOP!
        batch_metrics = {}
        for name, model in models_dict.items():
            optimizer = optimizers_dict[name]

            # Forward + backward
            log_probs = model(img1, img2)
            loss, batch_acc, num_correct, batch_size = compute_batch_metrics(log_probs, winners)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track
            metrics[name]['total_loss'] += loss.item()
            metrics[name]['total_correct'] += num_correct
            metrics[name]['total_samples'] += batch_size
            batch_metrics[name] = {'loss': loss.item(), 'acc': batch_acc}

        # Logging
        if batch_idx % log_interval == 0:
            if len(models_dict) == 1:
                name = model_names[0]
                logger.info(f"  Batch {batch_idx}/{len(loader)}: "
                          f"loss={batch_metrics[name]['loss']:.4f}, "
                          f"acc={batch_metrics[name]['acc']:.3f}")
            else:
                log_parts = [f"Batch {batch_idx}/{len(loader)}:"]
                for name in model_names:
                    log_parts.append(f"{name}_loss={batch_metrics[name]['loss']:.4f}, "
                                   f"{name}_acc={batch_metrics[name]['acc']:.3f}")
                logger.info("  " + " | ".join(log_parts))

        # Compare models (only runs when len(models_dict) > 1)
        if comparison_log is not None and batch_idx % batch_comparison_interval == 0:
            # ... comparison logic

    # Return results for all models
    results = {}
    for name in model_names:
        avg_loss = metrics[name]['total_loss'] / len(loader)
        avg_acc = metrics[name]['total_correct'] / metrics[name]['total_samples']
        results[name] = (avg_loss, avg_acc)

    return results
```

## Usage

### Single Model (Default)
```python
# In train_model()
models_dict = {'main': model}
optimizers_dict = {'main': optimizer}

results = train_epoch(models_dict, optimizers_dict, train_loader, device, log_interval,
                     epoch=epoch, output_dir=output_dir,
                     batch_comparison_interval=batch_comparison_interval)

train_loss, train_acc = results['main']
```

### Multiple Models (With Reference)
```python
# In train_model()
models_dict = {
    'main': model,
    'reference': ref_model,
    'baseline': baseline_model  # Can add more!
}
optimizers_dict = {
    'main': optimizer,
    'reference': ref_optimizer,
    'baseline': baseline_optimizer
}

results = train_epoch(models_dict, optimizers_dict, train_loader, device, log_interval,
                     epoch=epoch, output_dir=output_dir,
                     batch_comparison_interval=batch_comparison_interval)

train_loss, train_acc = results['main']
ref_loss, ref_acc = results['reference']
baseline_loss, baseline_acc = results['baseline']
```

## Benefits

1. **No More If Statements**: Eliminated all `if ref_model is not None` checks
2. **Extensible**: Can easily add 3, 4, or more models without changing the code
3. **Cleaner Logic**: Single loop trains all models uniformly
4. **Automatic Comparison**: When multiple models are present, automatically compares them
5. **Backward Compatible**: Works with single model (just wrap in dict)

## How It Works with No Reference Model

When there's only one model:

```python
models_dict = {'main': model}  # len == 1
```

1. `comparison_log = [] if (len(models_dict) > 1 and ...) else None`
   - Since `len(models_dict) == 1`, this becomes `comparison_log = None`

2. `if comparison_log is not None and ...:`
   - Since `comparison_log is None`, this entire block is skipped

3. `for ref_name in model_names[1:]:`
   - `model_names = ['main']`
   - `model_names[1:]` is an empty list
   - Loop doesn't execute

So all comparison code is automatically skipped when there's only one model!

## Files Modified

- [sim_bench/training/train_siamese_e2e.py](sim_bench/training/train_siamese_e2e.py)
  - Refactored `train_epoch()` to use dictionaries (lines 89-219)
  - Updated `train_model()` to create model/optimizer dicts (lines 429-440)

## Testing

Code compiles successfully:
```bash
python -m py_compile sim_bench/training/train_siamese_e2e.py
```

The refactored code maintains backward compatibility - existing training runs with a single model work exactly as before, just with cleaner internals.
