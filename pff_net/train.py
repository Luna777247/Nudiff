# Import các thư viện cần thiết
import torch  # PyTorch
import torch.optim as optim  # Optimizers
import torch.nn.functional as F  # Functional operations
from torch.utils.data import DataLoader  # Data loading
from model import PFFNet, semantic_consistency_loss, mask_quality_loss  # Model và loss functions
from dataset import MoNuSegDataset, get_transform  # Dataset và transforms
from config import *  # Import tất cả config
import os  # File operations

# Hàm train một epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()  # Set model to training mode
    total_loss = 0  # Tổng loss cho epoch
    for images, targets in data_loader:  # Duyệt qua batches
        images = [img.to(device) for img in images]  # Move images to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to device

        optimizer.zero_grad()  # Reset gradients

        outputs = model(images, targets)  # Forward pass

        # Tính losses
        loss_dict = model.mask_rcnn(images, targets)  # Lấy losses từ MaskRCNN
        loss_detection = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] + loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']  # Tổng loss detection
        loss_mask = loss_dict['loss_mask']  # Loss mask

        # Semantic losses
        semantic_pred = outputs['semantic_logits']  # Semantic predictions
        instance_semantic_pred = outputs['instance_semantic_logits']  # Instance semantic predictions
        # Giả sử semantic targets được tính từ masks
        semantic_target = torch.zeros_like(semantic_pred)  # Placeholder cho target
        loss_semantic_main = F.cross_entropy(semantic_pred, semantic_target)  # Loss semantic chính
        loss_semantic_instance = F.cross_entropy(instance_semantic_pred, semantic_target)  # Loss semantic instance
        loss_consistency = semantic_consistency_loss(semantic_pred, instance_semantic_pred)  # Loss consistency

        # Mask quality loss
        if 'mask_qualities' in outputs:
            qualities = torch.cat(outputs['mask_qualities'])  # Concat qualities
            loss_quality = mask_quality_loss(qualities, targets)  # Loss quality
        else:
            loss_quality = torch.tensor(0.0)  # Nếu không có, set 0

        # Tổng loss batch
        total_loss_batch = (
            LOSS_WEIGHTS['detection'] * loss_detection +
            LOSS_WEIGHTS['mask'] * loss_mask +
            LOSS_WEIGHTS['semantic_main'] * loss_semantic_main +
            LOSS_WEIGHTS['semantic_instance'] * loss_semantic_instance +
            LOSS_WEIGHTS['consistency'] * loss_consistency +
            LOSS_WEIGHTS['quality'] * loss_quality
        )

        total_loss_batch.backward()  # Backward pass
        optimizer.step()  # Update weights

        total_loss += total_loss_batch.item()  # Cộng dồn loss

    print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")  # In loss trung bình

# Hàm main
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()
    
    # Override config with command line args
    train_images = os.path.join(args.data_path, 'images')
    train_masks = os.path.join(args.data_path, 'labels')
    val_images = '../dataset/MoNuSeg/test/images'
    val_masks = '../dataset/MoNuSeg/test/labels'
    
    # Check if paths exist
    if not os.path.exists(train_images):
        print(f"❌ Training images not found: {train_images}")
        print(f"   Looking for structure: {args.data_path}/images/")
        print(f"   Available: {os.listdir(args.data_path) if os.path.exists(args.data_path) else 'Path not found'}")
        return
    
    device = DEVICE  # Device (GPU/CPU)

    model = PFFNet(NUM_CLASSES, BACKBONE).to(device)  # Khởi tạo model

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Optimizer

    # Dataset và DataLoader cho training
    train_dataset = MoNuSegDataset(train_images, train_masks, get_transform(train=True))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, collate_fn=lambda x: tuple(zip(*x)))

    # Dataset và DataLoader cho validation
    val_dataset = MoNuSegDataset(val_images, val_masks, get_transform(train=False))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda x: tuple(zip(*x)))

    # Training loop
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)  # Train một epoch
        # Validation có thể thêm ở đây

        if (epoch + 1) % 10 == 0:  # Lưu checkpoint mỗi 10 epochs
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'pffnet_epoch_{epoch+1}.pth'))

# Chạy main nếu file được execute trực tiếp
if __name__ == "__main__":
    main()