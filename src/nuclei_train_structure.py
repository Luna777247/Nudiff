"""
Training script for Unconditional Nuclei Structure Synthesis
Stage 1 of NuDiff pipeline
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy

from nuclei_diffusion_aug import (
    UnconditionalNucleiDiffusion,
    DiffusionProcess,
    NucleiStructureDataset,
    train_step_unconditional,
    sample_unconditional
)


def save_checkpoint(model, optimizer, epoch, loss, save_path, ema_model=None):
    """Save model checkpoint with optional EMA model"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def visualize_samples(samples, save_path):
    """Visualize generated structure samples"""
    n_samples = min(4, samples.shape[0])
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        sample = samples[i].cpu().numpy()
        
        # Semantic channel
        axes[i, 0].imshow(sample[0], cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].set_title('Semantic')
        axes[i, 0].axis('off')
        
        # Horizontal distance
        axes[i, 1].imshow(sample[1], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 1].set_title('Horizontal Distance')
        axes[i, 1].axis('off')
        
        # Vertical distance
        axes[i, 2].imshow(sample[2], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 2].set_title('Vertical Distance')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Samples saved: {save_path}")


def train(args):
    """Main training function"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = NucleiStructureDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Dataset size: {len(dataset)} patches")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: {len(dataloader)}")
    
    # Calculate total steps for LR annealing
    total_steps = args.epochs * len(dataloader)
    print(f"Total training steps: {total_steps}")
    
    # Initialize model
    print("\nInitializing model...")
    model = UnconditionalNucleiDiffusion(
        in_channels=3,
        channels=[128, 128, 256, 256, 512, 512] if args.image_size <= 256 else [256, 256, 512, 512, 1024, 1024]
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(timesteps=args.diffusion_steps)
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize GradScaler for mixed precision (FP16) - optional
    # Some versions of autocast work better with float32 for diffusion
    use_amp = args.use_amp
    if use_amp:
        scaler = GradScaler(device='cuda')
    else:
        scaler = None
    
    # Initialize EMA model (Exponential Moving Average)
    print("\nInitializing EMA model...")
    ema_model = copy.deepcopy(model)
    ema_decay = 0.99
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nLoading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'ema_model_state_dict' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"  EMA decay: {ema_decay}")
    print(f"  LR annealing: {args.lr} → 0 over {total_steps} steps")
    if use_amp:
        print(f"  Mixed Precision: FP16 enabled ✓")
    else:
        print(f"  Mixed Precision: FP32 (Float32)")
    print(f"{'='*60}\n")
    
    model.train()
    ema_model.eval()  # EMA model in eval mode
    global_step = 0
    
    # Early stopping variables
    best_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            
            # Training step with optional mixed precision (FP16)
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss = train_step_unconditional(model, diffusion, data, optimizer)
            else:
                loss = train_step_unconditional(model, diffusion, data, optimizer)
            
            epoch_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            
            # =================== EMA Update ===================
            # Update EMA model parameters
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data = ema_decay * ema_param.data + (1 - ema_decay) * param.data
            # ===================================================
            
            # ================ Learning Rate Annealing ================
            # Decay learning rate from args.lr to 0
            frac_done = (epoch * len(dataloader) + batch_idx) / total_steps
            lr = args.lr * (1 - frac_done)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # =========================================================
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{lr:.2e}', 'ema_decay': f'{ema_decay:.4f}'})
            
            global_step += 1
            
            # Log every N steps
            if global_step % args.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-args.log_interval:])
                print(f"Step {global_step}: loss = {avg_loss:.4f}, lr = {lr:.2e}")
        
        # Epoch summary
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            print(f"✓ Loss improved to {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⚠ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n⛔ Early stopping triggered! Best loss: {best_loss:.4f}")
                break
        
        # Save checkpoint automatically every epoch (for resume capability)
        checkpoint_path = os.path.join(
            args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1:03d}.pt'
        )
        save_checkpoint(model, optimizer, epoch, avg_epoch_loss, checkpoint_path, ema_model=ema_model)
        
        print()
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'checkpoints', 'final_model.pt')
    save_checkpoint(model, optimizer, args.epochs-1, avg_epoch_loss, final_path, ema_model=ema_model)
    
    # Also save EMA model as best model (usually better than raw model)
    ema_final_path = os.path.join(args.output_dir, 'checkpoints', 'ema_final_model.pt')
    torch.save({'ema_model_state_dict': ema_model.state_dict()}, ema_final_path)
    print(f"✓ EMA model saved: {ema_final_path} (recommended for inference)")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"  - Final model: final_model.pt (raw)")
    print(f"  - EMA model: ema_final_model.pt (recommended)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train unconditional nuclei structure synthesis model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing structure .npy files')
    parser.add_argument('--output_dir', type=str, default='outputs/structure_synthesis',
                        help='Output directory for checkpoints and samples')
    
    # Model arguments
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (256 or 1000)')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N steps')
    
    # Sampling arguments
    parser.add_argument('--sample_steps', type=int, default=250,
                        help='Number of denoising steps for sampling')
    
    # Mixed precision argument
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision (FP16) for training')
    
    # Early stopping argument
    parser.add_argument('--patience', type=int, default=15,
                        help='Number of epochs with no improvement before early stopping (default: 15)')
    
    args = parser.parse_args()
    
    train(args)
