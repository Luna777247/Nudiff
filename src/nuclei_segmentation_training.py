import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from scipy.ndimage import label as scipy_label
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# Evaluation Metrics (Dice and AJI)
# ============================================================================

def compute_dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute Dice coefficient for binary segmentation
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
        smooth: Smoothing factor
    Returns:
        dice: Dice coefficient
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def compute_aji(pred_inst, true_inst):
    """
    Compute Aggregated Jaccard Index (AJI) for instance segmentation
    Reference: Kumar et al. "A Dataset and a Technique for Generalized Nuclear 
    Segmentation for Computational Pathology" (IEEE TMI 2017)
    
    Args:
        pred_inst: Predicted instance map (H, W)
        true_inst: Ground truth instance map (H, W)
    Returns:
        aji: AJI score
    """
    true_ids = np.unique(true_inst)
    true_ids = true_ids[true_ids != 0]  # Remove background
    
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids != 0]  # Remove background
    
    # If no nuclei in either map
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0
    
    # Compute pairwise IoU
    true_masks = [(true_inst == i) for i in true_ids]
    pred_masks = [(pred_inst == i) for i in pred_ids]
    
    # Match predicted and true instances
    pairwise_iou = np.zeros((len(true_ids), len(pred_ids)))
    for i, true_mask in enumerate(true_masks):
        for j, pred_mask in enumerate(pred_masks):
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            pairwise_iou[i, j] = intersection / union if union > 0 else 0
    
    # Find best matches (greedy matching)
    matched_true = set()
    matched_pred = set()
    sum_intersect = 0
    
    # Sort by IoU and match greedily
    pairs = []
    for i in range(len(true_ids)):
        for j in range(len(pred_ids)):
            if pairwise_iou[i, j] > 0:
                pairs.append((pairwise_iou[i, j], i, j))
    
    pairs.sort(reverse=True)
    
    for iou, i, j in pairs:
        if i not in matched_true and j not in matched_pred:
            matched_true.add(i)
            matched_pred.add(j)
            intersection = np.logical_and(true_masks[i], pred_masks[j]).sum()
            sum_intersect += intersection
    
    # Compute sum of all areas
    sum_union = 0
    for true_mask in true_masks:
        sum_union += true_mask.sum()
    
    for j, pred_mask in enumerate(pred_masks):
        if j not in matched_pred:
            sum_union += pred_mask.sum()
    
    # Compute AJI
    aji = sum_intersect / sum_union if sum_union > 0 else 0
    
    return aji


def evaluate_segmentation(pred_instances, true_instances):
    """
    Evaluate segmentation results with Dice and AJI
    Args:
        pred_instances: List of predicted instance maps
        true_instances: List of ground truth instance maps
    Returns:
        metrics: Dictionary containing mean Dice and AJI
    """
    dice_scores = []
    aji_scores = []
    
    for pred_inst, true_inst in zip(pred_instances, true_instances):
        # Convert to binary for Dice
        pred_binary = (pred_inst > 0).astype(np.float32)
        true_binary = (true_inst > 0).astype(np.float32)
        
        # Compute Dice
        dice = compute_dice_coefficient(pred_binary, true_binary)
        dice_scores.append(dice)
        
        # Compute AJI
        aji = compute_aji(pred_inst, true_inst)
        aji_scores.append(aji)
    
    metrics = {
        'Dice': np.mean(dice_scores),
        'Dice_std': np.std(dice_scores),
        'AJI': np.mean(aji_scores),
        'AJI_std': np.std(aji_scores)
    }
    
    return metrics


# ============================================================================
# HoVer-Net Architecture
# ============================================================================

class DenseBlock(nn.Module):
    """Dense block for feature extraction"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                )
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class HoVerNet(nn.Module):
    """
    HoVer-Net: Horizontal and Vertical distance regression for instance segmentation
    Reference: Graham et al. "Hover-net: Simultaneous segmentation and classification 
    of nuclei in multi-tissue histology images" (Medical Image Analysis 2019)
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dense1 = DenseBlock(64, 32, 4)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dense2 = DenseBlock(192, 32, 4)
        
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dense3 = DenseBlock(320, 32, 4)
        
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dense4 = DenseBlock(448, 32, 4)
        
        # Decoder for Nuclear Pixel (NP) branch
        self.up_np_1 = nn.ConvTranspose2d(576, 256, 2, stride=2)
        self.conv_np_1 = nn.Sequential(
            nn.Conv2d(256 + 448, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up_np_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_np_2 = nn.Sequential(
            nn.Conv2d(128 + 320, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up_np_3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_np_3 = nn.Sequential(
            nn.Conv2d(64 + 192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up_np_4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out_np = nn.Conv2d(32, num_classes, 1)
        
        # Decoder for HoVer (Horizontal-Vertical) branch
        self.up_hv_1 = nn.ConvTranspose2d(576, 256, 2, stride=2)
        self.conv_hv_1 = nn.Sequential(
            nn.Conv2d(256 + 448, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up_hv_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_hv_2 = nn.Sequential(
            nn.Conv2d(128 + 320, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up_hv_3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_hv_3 = nn.Sequential(
            nn.Conv2d(64 + 192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up_hv_4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out_hv = nn.Conv2d(32, 2, 1)  # h_dist and v_dist
    
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        
        x2 = self.pool1(x1)
        x2 = self.dense1(x2)
        
        x3 = self.pool2(x2)
        x3 = self.dense2(x3)
        
        x4 = self.pool3(x3)
        x4 = self.dense3(x4)
        
        x5 = self.pool4(x4)
        x5 = self.dense4(x5)
        
        # NP branch decoder
        np = self.up_np_1(x5)
        np = torch.cat([np, x4], dim=1)
        np = self.conv_np_1(np)
        
        np = self.up_np_2(np)
        np = torch.cat([np, x3], dim=1)
        np = self.conv_np_2(np)
        
        np = self.up_np_3(np)
        np = torch.cat([np, x2], dim=1)
        np = self.conv_np_3(np)
        
        np = self.up_np_4(np)
        np = self.out_np(np)
        
        # HV branch decoder
        hv = self.up_hv_1(x5)
        hv = torch.cat([hv, x4], dim=1)
        hv = self.conv_hv_1(hv)
        
        hv = self.up_hv_2(hv)
        hv = torch.cat([hv, x3], dim=1)
        hv = self.conv_hv_2(hv)
        
        hv = self.up_hv_3(hv)
        hv = torch.cat([hv, x2], dim=1)
        hv = self.conv_hv_3(hv)
        
        hv = self.up_hv_4(hv)
        hv = self.out_hv(hv)
        
        return {'np': np, 'hv': hv}


def hovernet_post_process(np_pred, hv_pred, threshold=0.5):
    """
    Post-process HoVer-Net predictions to get instance segmentation
    Args:
        np_pred: Nuclear pixel prediction (H, W, 2) - [background, nuclei]
        hv_pred: HoVer prediction (H, W, 2) - [h_dist, v_dist]
        threshold: Threshold for nuclear pixel
    Returns:
        instance_map: Instance segmentation map
    """
    # Get binary mask
    binary_mask = (np_pred[..., 1] > threshold).astype(np.uint8)
    
    # Get horizontal and vertical distances
    h_dist = hv_pred[..., 0]
    v_dist = hv_pred[..., 1]
    
    # Compute distance from center
    dist_map = np.sqrt(h_dist**2 + v_dist**2)
    dist_map = 1 - dist_map  # Invert so peaks are at centers
    dist_map[binary_mask == 0] = 0
    
    # Find local maxima as markers
    from skimage.feature import peak_local_max
    from scipy.ndimage import label as scipy_label
    
    coordinates = peak_local_max(dist_map, min_distance=5, threshold_abs=0.1)
    markers = np.zeros_like(dist_map, dtype=bool)
    markers[tuple(coordinates.T)] = True
    markers = scipy_label(markers)[0]
    
    # Watershed segmentation
    instance_map = watershed(-dist_map, markers, mask=binary_mask)
    
    return instance_map


# ============================================================================
# PFF-Net Architecture
# ============================================================================

class PFFNet(nn.Module):
    """
    PFF-Net: Panoptic Feature Fusion Network
    Reference: Liu et al. "Panoptic feature fusion net: a novel instance segmentation 
    paradigm for biomedical and biological images" (IEEE TIP 2021)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._make_layer(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self._make_layer(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self._make_layer(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._make_layer(base_channels * 8, base_channels * 16)
        
        # Semantic segmentation branch
        self.up_sem_1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec_sem_1 = self._make_layer(base_channels * 16, base_channels * 8)
        
        self.up_sem_2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec_sem_2 = self._make_layer(base_channels * 8, base_channels * 4)
        
        self.up_sem_3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec_sem_3 = self._make_layer(base_channels * 4, base_channels * 2)
        
        self.up_sem_4 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec_sem_4 = self._make_layer(base_channels * 2, base_channels)
        
        self.out_sem = nn.Conv2d(base_channels, 2, 1)  # Binary segmentation
        
        # Instance contour branch
        self.up_cont_1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec_cont_1 = self._make_layer(base_channels * 16, base_channels * 8)
        
        self.up_cont_2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec_cont_2 = self._make_layer(base_channels * 8, base_channels * 4)
        
        self.up_cont_3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec_cont_3 = self._make_layer(base_channels * 4, base_channels * 2)
        
        self.up_cont_4 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec_cont_4 = self._make_layer(base_channels * 2, base_channels)
        
        self.out_cont = nn.Conv2d(base_channels, 1, 1)  # Contour prediction
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Semantic branch
        sem = self.up_sem_1(b)
        sem = torch.cat([sem, e4], dim=1)
        sem = self.dec_sem_1(sem)
        
        sem = self.up_sem_2(sem)
        sem = torch.cat([sem, e3], dim=1)
        sem = self.dec_sem_2(sem)
        
        sem = self.up_sem_3(sem)
        sem = torch.cat([sem, e2], dim=1)
        sem = self.dec_sem_3(sem)
        
        sem = self.up_sem_4(sem)
        sem = torch.cat([sem, e1], dim=1)
        sem = self.dec_sem_4(sem)
        
        sem_out = self.out_sem(sem)
        
        # Contour branch
        cont = self.up_cont_1(b)
        cont = torch.cat([cont, e4], dim=1)
        cont = self.dec_cont_1(cont)
        
        cont = self.up_cont_2(cont)
        cont = torch.cat([cont, e3], dim=1)
        cont = self.dec_cont_2(cont)
        
        cont = self.up_cont_3(cont)
        cont = torch.cat([cont, e2], dim=1)
        cont = self.dec_cont_3(cont)
        
        cont = self.up_cont_4(cont)
        cont = torch.cat([cont, e1], dim=1)
        cont = self.dec_cont_4(cont)
        
        cont_out = self.out_cont(cont)
        
        return {'semantic': sem_out, 'contour': cont_out}


def pffnet_post_process(sem_pred, cont_pred, threshold_sem=0.5, threshold_cont=0.5):
    """
    Post-process PFF-Net predictions to get instance segmentation
    Args:
        sem_pred: Semantic prediction (H, W, 2)
        cont_pred: Contour prediction (H, W)
        threshold_sem: Threshold for semantic segmentation
        threshold_cont: Threshold for contour detection
    Returns:
        instance_map: Instance segmentation map
    """
    # Get binary masks
    binary_mask = (sem_pred[..., 1] > threshold_sem).astype(np.uint8)
    contour_mask = (cont_pred > threshold_cont).astype(np.uint8)
    
    # Remove contours from semantic mask
    nuclei_interior = binary_mask * (1 - contour_mask)
    
    # Label connected components
    instance_map = scipy_label(nuclei_interior)[0]
    
    # Expand instances to fill gaps using watershed
    if instance_map.max() > 0:
        # Use distance transform for watershed
        distance = distance_transform_edt(binary_mask)
        instance_map = watershed(-distance, instance_map, mask=binary_mask)
    
    return instance_map


# ============================================================================
# Training Functions
# ============================================================================

class HoVerNetLoss(nn.Module):
    """Combined loss for HoVer-Net"""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # Target should contain 'np' (semantic) and 'hv' (hover)
        np_loss = self.ce_loss(pred['np'], target['np'].long())
        hv_loss = self.mse_loss(pred['hv'], target['hv'])
        
        return np_loss + hv_loss


class PFFNetLoss(nn.Module):
    """Combined loss for PFF-Net"""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # Target should contain 'semantic' and 'contour'
        sem_loss = self.ce_loss(pred['semantic'], target['semantic'].long())
        cont_loss = self.bce_loss(pred['contour'], target['contour'])
        
        return sem_loss + cont_loss


def prepare_hovernet_targets(structures):
    """
    Prepare targets for HoVer-Net from nuclei structures
    Args:
        structures: Batch of nuclei structures (B, 3, H, W)
    Returns:
        targets: Dictionary with 'np' and 'hv' targets
    """
    # structures: [semantic, h_dist, v_dist]
    semantic = structures[:, 0:1, :, :]  # (B, 1, H, W)
    h_dist = structures[:, 1:2, :, :]
    v_dist = structures[:, 2:3, :, :]
    
    # Create 2-class semantic target (background, nuclei)
    np_target = torch.zeros(semantic.shape[0], 2, semantic.shape[2], semantic.shape[3], 
                           device=semantic.device)
    np_target[:, 0] = 1 - semantic.squeeze(1)  # Background
    np_target[:, 1] = semantic.squeeze(1)      # Nuclei
    
    # Get class indices
    np_target_idx = torch.argmax(np_target, dim=1)
    
    # HV target
    hv_target = torch.cat([h_dist, v_dist], dim=1)
    
    return {'np': np_target_idx, 'hv': hv_target}


def prepare_pffnet_targets(structures, instances):
    """
    Prepare targets for PFF-Net from nuclei structures
    Args:
        structures: Batch of nuclei structures (B, 3, H, W)
        instances: Batch of instance maps (B, H, W)
    Returns:
        targets: Dictionary with 'semantic' and 'contour' targets
    """
    semantic = structures[:, 0, :, :]  # (B, H, W)
    
    # Create contour targets
    contour = torch.zeros_like(semantic)
    for b in range(instances.shape[0]):
        inst_map = instances[b].cpu().numpy()
        # Compute contours using gradient
        grad_x = np.abs(cv2.Sobel(inst_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(inst_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
        contour_map = (grad_x + grad_y) > 0
        contour[b] = torch.from_numpy(contour_map.astype(np.float32)).to(semantic.device)
    
    return {'semantic': semantic, 'contour': contour.unsqueeze(1)}


def train_epoch(model, dataloader, criterion, optimizer, device, model_type='hovernet'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        structures = batch['structure'].to(device)
        instances = batch['instance']
        
        # Prepare targets based on model type
        if model_type == 'hovernet':
            targets = prepare_hovernet_targets(structures)
            targets = {k: v.to(device) for k, v in targets.items()}
        else:  # pffnet
            targets = prepare_pffnet_targets(structures, instances)
            targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_model(model, dataloader, device, model_type='hovernet'):
    """Evaluate model on dataset"""
    model.eval()
    
    pred_instances = []
    true_instances = []
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
        images = batch['image'].to(device)
        instances = batch['instance'].numpy()
        
        # Forward pass
        outputs = model(images)
        
        # Post-process predictions
        batch_size = images.shape[0]
        for i in range(batch_size):
            if model_type == 'hovernet':
                np_pred = torch.softmax(outputs['np'][i], dim=0).cpu().numpy().transpose(1, 2, 0)
                hv_pred = outputs['hv'][i].cpu().numpy().transpose(1, 2, 0)
                pred_inst = hovernet_post_process(np_pred, hv_pred)
            else:  # pffnet
                sem_pred = torch.softmax(outputs['semantic'][i], dim=0).cpu().numpy().transpose(1, 2, 0)
                cont_pred = torch.sigmoid(outputs['contour'][i, 0]).cpu().numpy()
                pred_inst = pffnet_post_process(sem_pred, cont_pred)
            
            pred_instances.append(pred_inst)
            true_instances.append(instances[i])
    
    # Compute metrics
    metrics = evaluate_segmentation(pred_instances, true_instances)
    
    return metrics


# ============================================================================
# Full Training Pipeline
# ============================================================================

def train_segmentation_model(model_type='hovernet', dataset_percentage=1.0, 
                             epochs=100, batch_size=4, lr=1e-4,
                             data_root='dataset/MoNuSeg', device='cuda'):
    """
    Complete training pipeline for nuclei segmentation
    Args:
        model_type: 'hovernet' or 'pffnet'
        dataset_percentage: 0.1, 0.2, 0.5, or 1.0
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        data_root: Root directory of dataset
        device: Training device
    Returns:
        model: Trained model
        results: Dictionary with training history and final metrics
    """
    from nuclei_data_prep import prepare_datasets
    
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} with {int(dataset_percentage*100)}% labeled data")
    print(f"{'='*70}\n")
    
    # Prepare datasets
    datasets, dataloaders = prepare_datasets(
        root_dir=data_root,
        label_percentages=[dataset_percentage],
        batch_size=batch_size
    )
    
    train_loader = dataloaders[f'train_{int(dataset_percentage*100)}pct']
    test_loader = dataloaders['test']
    
    # Initialize model
    if model_type == 'hovernet':
        model = HoVerNet().to(device)
        criterion = HoVerNetLoss()
    else:
        model = PFFNet().to(device)
        criterion = PFFNetLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=10)
    
    # Training loop
    best_dice = 0
    train_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, model_type)
        train_losses.append(train_loss)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            metrics = evaluate_model(model, test_loader, device, model_type)
            print(f"Dice: {metrics['Dice']:.4f} ± {metrics['Dice_std']:.4f}")
            print(f"AJI: {metrics['AJI']:.4f} ± {metrics['AJI_std']:.4f}")
            
            # Save best model
            if metrics['Dice'] > best_dice:
                best_dice = metrics['Dice']
                torch.save(model.state_dict(), 
                          f"{model_type}_{int(dataset_percentage*100)}pct_best.pth")
        
        scheduler.step(train_loss)
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(f"{model_type}_{int(dataset_percentage*100)}pct_best.pth"))
    final_metrics = evaluate_model(model, test_loader, device, model_type)
    
    print(f"\nFinal Dice: {final_metrics['Dice']:.4f} ± {final_metrics['Dice_std']:.4f}")
    print(f"Final AJI: {final_metrics['AJI']:.4f} ± {final_metrics['AJI_std']:.4f}")
    
    results = {
        'train_losses': train_losses,
        'final_metrics': final_metrics,
        'best_dice': best_dice
    }
    
    return model, results


# ============================================================================
# Reproduce Paper Results
# ============================================================================

def reproduce_paper_results(data_root='dataset/MoNuSeg', device='cuda'):
    """
    Reproduce all results from the paper (Table 1 and Table 2)
    """
    percentages = [0.1, 0.2, 0.5, 1.0]
    models = ['hovernet', 'pffnet']
    
    all_results = {}
    
    for model_type in models:
        print(f"\n{'#'*70}")
        print(f"# Training {model_type.upper()}")
        print(f"{'#'*70}\n")
        
        model_results = {}
        
        for pct in percentages:
            model, results = train_segmentation_model(
                model_type=model_type,
                dataset_percentage=pct,
                epochs=100,
                batch_size=4,
                lr=1e-4 if model_type == 'hovernet' else 2e-5,
                data_root=data_root,
                device=device
            )
            
            model_results[f'{int(pct*100)}pct'] = results['final_metrics']
        
        all_results[model_type] = model_results
    
    # Display results in table format
    print(f"\n{'='*70}")
    print("FINAL RESULTS (Reproducing Paper Tables)")
    print(f"{'='*70}\n")
    
    # Table 1: HoVer-Net Results
    print("\nTable 1: Effectiveness of Data Augmentation with Hover-Net")
    print("="*70)
    print(f"{'Training Data':<25} {'MoNuSeg':<25} {'Kumar':<25}")
    print(f"{'':25} {'Dice':<12} {'AJI':<12} {'Dice':<12} {'AJI':<12}")
    print("-"*70)
    
    if 'hovernet' in all_results:
        for pct in percentages:
            key = f'{int(pct*100)}pct'
            metrics = all_results['hovernet'][key]
            print(f"{key + ' labeled':<25} "
                  f"{metrics['Dice']:.4f}      {metrics['AJI']:.4f}      "
                  f"{'N/A':<12} {'N/A':<12}")  # Kumar results would go here
    
    # Table 2: PFF-Net Results
    print("\n\nTable 2: Generalization of Data Augmentation with PFF-Net")
    print("="*70)
    print(f"{'Training Data':<25} {'MoNuSeg':<25} {'Kumar':<25}")
    print(f"{'':25} {'Dice':<12} {'AJI':<12} {'Dice':<12} {'AJI':<12}")
    print("-"*70)
    
    if 'pffnet' in all_results:
        for pct in percentages:
            key = f'{int(pct*100)}pct'
            metrics = all_results['pffnet'][key]
            print(f"{key + ' labeled':<25} "
                  f"{metrics['Dice']:.4f}      {metrics['AJI']:.4f}      "
                  f"{'N/A':<12} {'N/A':<12}")  # Kumar results would go here
    
    print("\n" + "="*70)
    print("Note: Kumar dataset results require separate training with Kumar data")
    print("="*70)
    
    return all_results


# ============================================================================
# Dataset Statistics and Visualization
# ============================================================================

def analyze_dataset(data_root='dataset/MoNuSeg'):
    """
    Analyze and visualize dataset statistics
    """
    from nuclei_data_prep import NucleiDataset, get_val_transforms
    
    print(f"\n{'='*70}")
    print("Dataset Analysis")
    print(f"{'='*70}\n")
    
    # Load datasets
    train_dataset = NucleiDataset(
        root_dir=data_root,
        split='train',
        patch_size=256,
        stride=128,
        transform=get_val_transforms(),
        load_structure=True
    )
    
    test_dataset = NucleiDataset(
        root_dir=data_root,
        split='test',
        patch_size=256,
        stride=128,
        transform=get_val_transforms(),
        load_structure=True
    )
    
    print(f"Training patches: {len(train_dataset)}")
    print(f"Test patches: {len(test_dataset)}")
    
    # Analyze nuclei statistics
    train_nuclei_counts = []
    train_nuclei_areas = []
    
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        instance = sample['instance']
        
        num_nuclei = instance.max()
        train_nuclei_counts.append(num_nuclei)
        
        for nid in range(1, num_nuclei + 1):
            area = (instance == nid).sum()
            train_nuclei_areas.append(area)
    
    print(f"\nTraining set statistics:")
    print(f"  Average nuclei per patch: {np.mean(train_nuclei_counts):.2f} ± {np.std(train_nuclei_counts):.2f}")
    print(f"  Average nucleus area: {np.mean(train_nuclei_areas):.2f} ± {np.std(train_nuclei_areas):.2f} pixels")
    print(f"  Total nuclei: {sum(train_nuclei_counts)}")
    
    # Visualize distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(train_nuclei_counts, bins=30, edgecolor='black')
    axes[0].set_xlabel('Number of nuclei per patch')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Nuclei Count')
    
    axes[1].hist(train_nuclei_areas, bins=50, edgecolor='black')
    axes[1].set_xlabel('Nucleus area (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Nucleus Area')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to 'dataset_statistics.png'")
    
    # Sample visualization
    print(f"\nVisualizing sample predictions...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(3):
        sample = train_dataset[i]
        
        # Image
        image = sample['image'].permute(1, 2, 0).numpy()
        image = np.clip(image * 0.229 + 0.485, 0, 1)
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        # Semantic
        semantic = sample['structure'][0].numpy()
        axes[i, 1].imshow(semantic, cmap='gray')
        axes[i, 1].set_title('Semantic')
        axes[i, 1].axis('off')
        
        # Distance transforms
        h_dist = sample['structure'][1].numpy()
        v_dist = sample['structure'][2].numpy()
        
        im = axes[i, 2].imshow(h_dist, cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 2].set_title('H-Distance')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(v_dist, cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 3].set_title('V-Distance')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print(f"Saved sample visualization to 'dataset_samples.png'")


def compare_augmentation_results(base_results, augmented_results):
    """
    Compare results between baseline and augmented datasets
    Args:
        base_results: Results from baseline (non-augmented) training
        augmented_results: Results from augmented training
    """
    print(f"\n{'='*70}")
    print("Comparison: Baseline vs Augmented")
    print(f"{'='*70}\n")
    
    percentages = [10, 20, 50, 100]
    
    print(f"{'Dataset':<25} {'Baseline Dice':<20} {'Augmented Dice':<20} {'Improvement':<15}")
    print("-" * 80)
    
    for pct in percentages:
        key = f'{pct}pct'
        base_dice = base_results[key]['Dice']
        aug_dice = augmented_results[key]['Dice']
        improvement = aug_dice - base_dice
        
        print(f"{key + ' labeled':<25} "
              f"{base_dice:.4f}              "
              f"{aug_dice:.4f}              "
              f"{improvement:+.4f} ({improvement/base_dice*100:+.2f}%)")
    
    print("\n")
    print(f"{'Dataset':<25} {'Baseline AJI':<20} {'Augmented AJI':<20} {'Improvement':<15}")
    print("-" * 80)
    
    for pct in percentages:
        key = f'{pct}pct'
        base_aji = base_results[key]['AJI']
        aug_aji = augmented_results[key]['AJI']
        improvement = aug_aji - base_aji
        
        print(f"{key + ' labeled':<25} "
              f"{base_aji:.4f}              "
              f"{aug_aji:.4f}              "
              f"{improvement:+.4f} ({improvement/base_aji*100:+.2f}%)")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train nuclei segmentation models')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'analyze', 'reproduce'],
                       help='Execution mode')
    parser.add_argument('--model', type=str, default='hovernet',
                       choices=['hovernet', 'pffnet'],
                       help='Model architecture')
    parser.add_argument('--data_root', type=str, default='dataset/MoNuSeg',
                       help='Dataset root directory')
    parser.add_argument('--percentage', type=float, default=1.0,
                       help='Percentage of labeled data to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Training device')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        # Analyze dataset
        analyze_dataset(args.data_root)
    
    elif args.mode == 'train':
        # Train single model
        model, results = train_segmentation_model(
            model_type=args.model,
            dataset_percentage=args.percentage,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            data_root=args.data_root,
            device=args.device
        )
        
        print("\nTraining completed!")
        print(f"Best model saved to {args.model}_{int(args.percentage*100)}pct_best.pth")
    
    elif args.mode == 'reproduce':
        # Reproduce all paper results
        all_results = reproduce_paper_results(
            data_root=args.data_root,
            device=args.device
        )
        
        # Save results
        import json
        with open('paper_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\nAll results saved to 'paper_results.json'")
    
    else:
        print(f"Unknown mode: {args.mode}")