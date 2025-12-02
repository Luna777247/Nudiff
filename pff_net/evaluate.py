# Import các thư viện cần thiết
import torch  # PyTorch
import torch.nn.functional as F  # Functional operations
import numpy as np  # Numerical operations
from sklearn.metrics import jaccard_score  # Jaccard index
from model import PFFNet  # Model PFFNet
from dataset import MoNuSegDataset, get_transform  # Dataset và transforms
from config import *  # Import config
import os  # File operations

# Hàm tính Dice coefficient
def dice_coefficient(pred, target, smooth=1e-6):
    # pred và target là binary masks
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Hàm tính AJI (Aggregated Jaccard Index)
def aggregated_jaccard_index(pred_masks, true_masks):
    # Simplified AJI calculation
    # In practice, need proper instance matching
    pred_flat = pred_masks.flatten()
    true_flat = true_masks.flatten()
    return jaccard_score(true_flat, pred_flat, average='macro')

# Hàm evaluate model
def evaluate_model(model, data_loader, device):
    model.eval()  # Set to evaluation mode
    dice_scores = []  # List để lưu Dice scores
    aji_scores = []  # List để lưu AJI scores

    with torch.no_grad():  # Không tính gradient
        for images, targets in data_loader:
            images = [img.to(device) for img in images]  # Move to device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images, targets)  # Forward pass

            # Lấy semantic predictions
            semantic_logits = outputs['semantic_logits']
            semantic_pred = torch.argmax(semantic_logits, dim=1)  # Argmax để lấy class

            # Lấy instance predictions
            instance_output = outputs['instance_output']
            instance_masks = instance_output[0]['masks']  # Instance masks

            # Tính metrics cho từng sample
            for i in range(len(images)):
                # Semantic segmentation metrics
                true_mask = targets[i]['masks']  # Ground truth masks
                pred_mask = semantic_pred[i]  # Predicted mask

                # Convert to numpy
                true_mask_np = true_mask.cpu().numpy()
                pred_mask_np = pred_mask.cpu().numpy()

                # Tính Dice
                dice = dice_coefficient(pred_mask_np, true_mask_np)
                dice_scores.append(dice)

                # Tính AJI (simplified)
                aji = aggregated_jaccard_index(instance_masks.cpu().numpy(), true_mask_np)
                aji_scores.append(aji)

    # Trung bình các scores
    avg_dice = np.mean(dice_scores)
    avg_aji = np.mean(aji_scores)

    return avg_dice, avg_aji

# Hàm main
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Override config with command line args
    test_images = os.path.join(args.data_path, 'images')
    test_masks = os.path.join(args.data_path, 'labels')
    
    # Check if paths exist
    if not os.path.exists(test_images):
        print(f"❌ Test images not found: {test_images}")
        return
        
    device = DEVICE  # Device

    # Load model
    model = PFFNet(NUM_CLASSES, BACKBONE).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"❌ Checkpoint not found: {args.model_path}")
        return

    # Dataset và DataLoader cho test
    test_dataset = MoNuSegDataset(test_images, test_masks, get_transform(train=False))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda x: tuple(zip(*x)))

    # Evaluate
    dice, aji = evaluate_model(model, test_loader, device)

    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Aggregated Jaccard Index (AJI): {aji:.4f}")

# Chạy main nếu file được execute trực tiếp
if __name__ == "__main__":
    main()