"""Simple batch prediction logger - just writes to CSV, no complexity."""
import csv
from pathlib import Path

def log_batch_predictions(output_path, batch_idx, epoch, image1, image2, winners, logits):
    """Write batch predictions to CSV. Creates file if it doesn't exist."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to decide on headers
    file_exists = output_path.exists()
    
    # Convert tensors to CPU numpy
    winners_np = winners.cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    
    # Append to file
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if new file
        if not file_exists:
            writer.writerow(['batch_idx', 'epoch', 'sample_idx', 'image1', 'image2', 'winner', 'logit0', 'logit1'])
        
        # Write data
        for i in range(len(image1)):
            writer.writerow([
                batch_idx, epoch, i,
                image1[i], image2[i],
                int(winners_np[i]),
                f"{logits_np[i, 0]:.10f}",
                f"{logits_np[i, 1]:.10f}"
            ])
        
        f.flush()
