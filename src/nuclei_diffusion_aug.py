import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import os
import glob
import cv2
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# SPADE Module - Spatially-Adaptive Normalization
# ============================================================================

class SPADE(nn.Module):
    """Spatially-Adaptive Normalization module"""
    def __init__(self, norm_channels, condition_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, norm_channels, affine=False)
        
        # Shared convolution for condition
        self.conv_shared = nn.Sequential(
            nn.Conv2d(condition_channels, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Separate convolutions for scale and bias
        self.conv_gamma = nn.Conv2d(128, norm_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, norm_channels, kernel_size=3, padding=1)
    
    def forward(self, x, condition):
        # Normalize input
        normalized = self.norm(x)
        
        # Resize condition to match x spatial dimensions
        condition = F.interpolate(condition, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Generate spatially-adaptive parameters
        actv = self.conv_shared(condition)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)
        
        # Apply affine transformation
        out = normalized * (1 + gamma) + beta
        return out


# ============================================================================
# Diffusion Process Components
# ============================================================================

def get_beta_schedule(timesteps, schedule_type='linear'):
    """Generate variance schedule for diffusion process"""
    if schedule_type == 'linear':
        beta_start = 0.0001
        beta_end = 0.02
        return np.linspace(beta_start, beta_end, timesteps)
    else:
        raise NotImplementedError(f"Schedule type {schedule_type} not implemented")


class DiffusionProcess:
    """Handles forward and reverse diffusion processes"""
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        
        # Generate beta schedule
        self.betas = torch.from_numpy(get_beta_schedule(timesteps)).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-compute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, condition=None):
        """Reverse diffusion: denoise x_t to x_{t-1}"""
        # Predict noise
        if condition is not None:
            predicted_noise = model(x_t, t, condition)
        else:
            predicted_noise = model(x_t, t)
        
        # Compute coefficients
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        # Compute mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
        )
        
        # Add noise if not final step
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            return mean + sigma_t * noise
        else:
            return mean


# ============================================================================
# Neural Network Blocks
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm"""
    def __init__(self, in_channels, out_channels, time_channels=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb = nn.Linear(time_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Add time embedding
        h = h + self.time_emb(F.silu(t_emb))[:, :, None, None]
        
        h = self.conv2(F.silu(self.norm2(h)))
        
        return h + self.skip(x)


class CondResBlock(nn.Module):
    """Conditional residual block with SPADE"""
    def __init__(self, in_channels, out_channels, condition_channels=3, time_channels=256):
        super().__init__()
        self.spade = SPADE(in_channels, condition_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb = nn.Linear(time_channels, out_channels)
        
        self.spade2 = SPADE(out_channels, condition_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb, condition):
        h = self.conv1(F.silu(self.spade(x, condition)))
        
        # Add time embedding
        h = h + self.time_emb(F.silu(t_emb))[:, :, None, None]
        
        h = self.conv2(F.silu(self.spade2(h, condition)))
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalize and compute Q, K, V
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        
        # Compute attention
        scale = c ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(b, c, h, w)
        
        return x + self.proj(out)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.proj(emb)


# ============================================================================
# Unconditional Nuclei Structure Synthesis Model
# ============================================================================

class UnconditionalNucleiDiffusion(nn.Module):
    """Unconditional diffusion model for nuclei structure synthesis"""
    def __init__(self, in_channels=3, channels=[256, 256, 512, 512, 1024, 1024]):
        super().__init__()
        self.time_embed_dim = 1024  # 256 * 4 from TimeEmbedding
        self.time_embed = TimeEmbedding(256)
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(len(channels)):
            layers = []
            in_ch = channels[i-1] if i > 0 else channels[0]
            out_ch = channels[i]
            
            # Two ResBlocks per level
            layers.append(ResBlock(in_ch, out_ch, time_channels=self.time_embed_dim))
            layers.append(ResBlock(out_ch, out_ch, time_channels=self.time_embed_dim))
            
            # Attention for last 3 levels
            if i >= len(channels) - 3:
                layers.append(AttentionBlock(out_ch))
            
            # Downsample (except last level)
            if i < len(channels) - 1:
                layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            
            self.encoder.append(nn.ModuleList(layers))
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            layers = []
            out_ch = channels[i]
            in_ch = channels[i+1] if i < len(channels)-1 else out_ch
            
            # Upsample (except first level)
            if i < len(channels) - 1:
                layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            
            # Two ResBlocks
            layers.append(ResBlock(out_ch * 2 if i < len(channels)-1 else out_ch, out_ch, time_channels=self.time_embed_dim))
            layers.append(ResBlock(out_ch, out_ch, time_channels=self.time_embed_dim))
            
            # Attention for last 3 levels
            if i >= len(channels) - 3:
                layers.append(AttentionBlock(out_ch))
            
            self.decoder.append(nn.ModuleList(layers))
        
        # Final projection
        self.final = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial projection
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skips = []
        for layers in self.encoder:
            # Process all layers in this encoder block
            for layer in layers:
                if isinstance(layer, ResBlock):
                    x = layer(x, t_emb)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
                elif isinstance(layer, nn.Conv2d):  # Downsample
                    # Store skip BEFORE downsampling
                    skips.append(x)
                    x = layer(x)
        
        # Store the bottleneck
        skips.append(x)
        
        # Decoder
        x = skips.pop()  # Start from bottleneck
        for i, layers in enumerate(self.decoder):
            for j, layer in enumerate(layers):
                if isinstance(layer, nn.ConvTranspose2d):
                    x = layer(x)
                    # Concatenate skip connection after upsampling
                    if len(skips) > 0:
                        skip = skips.pop()
                        x = torch.cat([x, skip], dim=1)
                elif isinstance(layer, ResBlock):
                    x = layer(x, t_emb)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
        
        return self.final(x)


# ============================================================================
# Conditional Histopathology Image Synthesis Model
# ============================================================================

class ConditionalHistopathologyDiffusion(nn.Module):
    """Conditional diffusion model for histopathology image synthesis"""
    def __init__(self, in_channels=3, condition_channels=3, channels=[256, 256, 512, 512, 1024, 1024]):
        super().__init__()
        self.time_embed_dim = 1024  # 256 * 4 from TimeEmbedding
        self.time_embed = TimeEmbedding(256)
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(len(channels)):
            layers = []
            in_ch = channels[i-1] if i > 0 else channels[0]
            out_ch = channels[i]
            
            # Two ResBlocks
            layers.append(ResBlock(in_ch, out_ch, time_channels=self.time_embed_dim))
            layers.append(ResBlock(out_ch, out_ch, time_channels=self.time_embed_dim))
            
            # Attention for last 2 levels
            if i >= len(channels) - 2:
                layers.append(AttentionBlock(out_ch))
            
            # Downsample
            if i < len(channels) - 1:
                layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            
            self.encoder.append(nn.ModuleList(layers))
        
        # Decoder with conditional blocks
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            layers = []
            out_ch = channels[i]
            in_ch = channels[i+1] if i < len(channels)-1 else out_ch
            
            # Upsample
            if i < len(channels) - 1:
                layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            
            # Two CondResBlocks
            layers.append(CondResBlock(out_ch * 2 if i < len(channels)-1 else out_ch, 
                                      out_ch, condition_channels, time_channels=self.time_embed_dim))
            layers.append(CondResBlock(out_ch, out_ch, condition_channels, time_channels=self.time_embed_dim))
            
            # Attention for last 2 levels
            if i >= len(channels) - 2:
                layers.append(AttentionBlock(out_ch))
            
            self.decoder.append(nn.ModuleList(layers))
        
        # Final projection
        self.final = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1)
        )
    
    def forward(self, x, t, condition=None):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial projection
        x = self.init_conv(x)
        
        # Encoder
        skips = []
        for layers in self.encoder:
            for layer in layers:
                if isinstance(layer, ResBlock):
                    x = layer(x, t_emb)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
                elif isinstance(layer, nn.Conv2d):  # Downsample
                    # Store skip BEFORE downsampling
                    skips.append(x)
                    x = layer(x)
        
        # Store bottleneck
        skips.append(x)
        
        # Decoder with conditioning
        x = skips.pop()  # Start from bottleneck
        for i, layers in enumerate(self.decoder):
            for j, layer in enumerate(layers):
                if isinstance(layer, nn.ConvTranspose2d):
                    x = layer(x)
                    # Concatenate skip after upsampling
                    if len(skips) > 0:
                        skip = skips.pop()
                        x = torch.cat([x, skip], dim=1)
                elif isinstance(layer, CondResBlock):
                    if condition is not None:
                        x = layer(x, t_emb, condition)
                    else:
                        # Unconditional mode (for classifier-free guidance training)
                        x = layer(x, t_emb, torch.zeros(x.shape[0], 3, *x.shape[2:], device=x.device))
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
        
        return self.final(x)


# ============================================================================
# Training and Sampling
# ============================================================================

def train_step_unconditional(model, diffusion, data, optimizer):
    """Single training step for unconditional model"""
    optimizer.zero_grad()
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.timesteps, (data.shape[0],), device=data.device)
    
    # Sample noise
    noise = torch.randn_like(data)
    
    # Add noise to data
    x_t = diffusion.q_sample(data, t, noise)
    
    # Predict noise
    predicted_noise = model(x_t, t)
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_step_conditional(model, diffusion, images, conditions, optimizer, drop_prob=0.2):
    """Single training step for conditional model with classifier-free guidance"""
    optimizer.zero_grad()
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=images.device)
    
    # Sample noise
    noise = torch.randn_like(images)
    
    # Add noise
    x_t = diffusion.q_sample(images, t, noise)
    
    # Randomly drop conditions for classifier-free guidance
    mask = torch.rand(images.shape[0], device=images.device) > drop_prob
    masked_conditions = conditions * mask[:, None, None, None]
    
    # Predict noise
    predicted_noise = model(x_t, t, masked_conditions)
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def sample_unconditional(model, diffusion, shape, device, num_steps=1000):
    """Generate samples from unconditional model"""
    model.eval()
    
    # Start from random noise
    x = torch.randn(shape, device=device)
    
    # Reverse diffusion
    for i in reversed(range(num_steps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t)
    
    return x


@torch.no_grad()
def sample_conditional(model, diffusion, condition, device, guidance_scale=2.0, num_steps=1000):
    """Generate samples from conditional model with classifier-free guidance"""
    model.eval()
    
    batch_size = condition.shape[0]
    shape = (batch_size, 3, condition.shape[2], condition.shape[3])
    
    # Start from random noise
    x = torch.randn(shape, device=device)
    
    # Reverse diffusion with guidance
    for i in reversed(range(num_steps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        # Predict with and without condition
        noise_cond = model(x, t, condition)
        noise_uncond = model(x, t, torch.zeros_like(condition))
        
        # Apply classifier-free guidance
        noise_pred = (1 + guidance_scale) * noise_cond - guidance_scale * noise_uncond
        
        # Denoise
        alpha_t = diffusion.alphas[i]
        alpha_cumprod_t = diffusion.alphas_cumprod[i]
        beta_t = diffusion.betas[i]
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * noise_pred
        )
        
        if i > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * noise
        else:
            x = mean
    
    return x


# ============================================================================
# Dataset Classes
# ============================================================================

class NucleiStructureDataset(Dataset):
    """Dataset for loading nuclei structures for unconditional training"""
    def __init__(self, structures_dir, transform=None):
        """
        Args:
            structures_dir: Directory containing .npy structure files
            transform: Optional transform to apply
        """
        self.structure_files = sorted(glob.glob(os.path.join(structures_dir, '*.npy')))
        self.transform = transform
        
        if len(self.structure_files) == 0:
            raise ValueError(f"No .npy files found in {structures_dir}")
        
        print(f"Loaded {len(self.structure_files)} structure files from {structures_dir}")
    
    def __len__(self):
        return len(self.structure_files)
    
    def __getitem__(self, idx):
        # Load structure (H, W, 3) float32
        structure = np.load(self.structure_files[idx])
        
        # Convert to tensor (3, H, W)
        structure = torch.from_numpy(structure).permute(2, 0, 1).float()
        
        if self.transform:
            structure = self.transform(structure)
        
        return structure


class ImageStructureDataset(Dataset):
    """Dataset for loading image-structure pairs for conditional training"""
    def __init__(self, images_dir, structures_dir, transform=None):
        """
        Args:
            images_dir: Directory containing .png image files
            structures_dir: Directory containing .npy structure files
            transform: Optional transform to apply to both
        """
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        self.structure_files = sorted(glob.glob(os.path.join(structures_dir, '*.npy')))
        self.transform = transform
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .png files found in {images_dir}")
        if len(self.structure_files) == 0:
            raise ValueError(f"No .npy files found in {structures_dir}")
        if len(self.image_files) != len(self.structure_files):
            raise ValueError(f"Mismatch: {len(self.image_files)} images vs {len(self.structure_files)} structures")
        
        print(f"Loaded {len(self.image_files)} image-structure pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Load structure
        structure = np.load(self.structure_files[idx])
        structure = torch.from_numpy(structure).permute(2, 0, 1).float()
        
        if self.transform:
            # Apply same transform to both
            combined = torch.cat([image, structure], dim=0)
            combined = self.transform(combined)
            image = combined[:3]
            structure = combined[3:]
        
        return {'image': image, 'structure': structure}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    unconditional_model = UnconditionalNucleiDiffusion().to(device)
    conditional_model = ConditionalHistopathologyDiffusion().to(device)
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(timesteps=1000)
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    # Example: Generate nuclei structure
    print("Generating nuclei structure...")
    nuclei_structure = sample_unconditional(
        unconditional_model, 
        diffusion,
        shape=(1, 3, 256, 256),
        device=device,
        num_steps=1000
    )
    
    # Example: Generate histopathology image conditioned on structure
    print("Generating histopathology image...")
    histopathology_image = sample_conditional(
        conditional_model,
        diffusion,
        condition=nuclei_structure,
        device=device,
        guidance_scale=2.0,
        num_steps=1000
    )
    
    print("Generation complete!")
    print(f"Nuclei structure shape: {nuclei_structure.shape}")
    print(f"Histopathology image shape: {histopathology_image.shape}")