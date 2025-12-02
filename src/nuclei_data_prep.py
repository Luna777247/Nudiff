import os
import cv2
import numpy as np
import scipy.io as sio
from scipy import ndimage
from skimage import morphology
from skimage.segmentation import watershed
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# Nuclei Structure Computation
# ============================================================================

def compute_distance_transform(mask):
    """
    Compute horizontal and vertical distance transforms for a nucleus mask
    Using bounding box normalization (matches original NuDiff paper)
    Args:
        mask: Binary mask of a single nucleus (H, W)
    Returns:
        h_dist, v_dist: Normalized horizontal and vertical distance maps
    """
    # Get nucleus coordinates and center
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.zeros_like(mask, dtype=np.float32), np.zeros_like(mask, dtype=np.float32)
    
    center_y, center_x = coords.mean(axis=0)
    
    # Get bounding box for normalization (like original script)
    y_min, y_max = coords[:, 0].min(), coords[:, 0].max() + 1
    x_min, x_max = coords[:, 1].min(), coords[:, 1].max() + 1
    max_distance = max(y_max - y_min, x_max - x_min)
    
    # Initialize distance maps
    h_dist = np.zeros_like(mask, dtype=np.float32)
    v_dist = np.zeros_like(mask, dtype=np.float32)
    
    # Compute distances for each pixel in the nucleus
    for y, x in coords:
        h_dist[y, x] = (x - center_x) / max_distance if max_distance > 0 else 0
        v_dist[y, x] = (y - center_y) / max_distance if max_distance > 0 else 0
    
    return h_dist, v_dist


def instance_to_structure(instance_map):
    """
    Convert instance segmentation map to nuclei structure (3-channel)
    Matches original NuDiff paper format
    Args:
        instance_map: Instance segmentation map (H, W) where each nucleus has unique ID
    Returns:
        structure: 3-channel nuclei structure [semantic, v_dist, h_dist]
                   semantic: -1 (background), 1 (nuclei)
                   v_dist, h_dist: normalized distances in [-1, 1]
    """
    h, w = instance_map.shape
    
    # Initialize structure maps
    # semantic: -1 (background), 1 (nuclei) - matches original NuDiff
    semantic = np.ones((h, w), dtype=np.float32) * -1  # Initialize all to -1 (background)
    semantic[instance_map > 0] = 1  # Set nuclei to 1
    
    # Distance maps initialized to 0
    h_dist_map = np.zeros((h, w), dtype=np.float32)
    v_dist_map = np.zeros((h, w), dtype=np.float32)
    
    # Get unique nucleus IDs
    nucleus_ids = np.unique(instance_map)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]  # Exclude background
    
    # Compute distance transform for each nucleus
    for nid in nucleus_ids:
        mask = (instance_map == nid)
        h_dist, v_dist = compute_distance_transform(mask)
        h_dist_map[mask] = h_dist[mask]
        v_dist_map[mask] = v_dist[mask]
    
    # Stack into 3-channel structure (semantic, v_dist, h_dist) like original
    structure = np.stack([semantic, v_dist_map, h_dist_map], axis=-1)
    
    return structure


def structure_to_instance(structure, threshold=0.0):
    """
    Convert nuclei structure back to instance map using watershed
    Args:
        structure: 3-channel nuclei structure [semantic, v_dist, h_dist]
        threshold: Threshold for semantic segmentation (0 for -1/1 encoding)
    Returns:
        instance_map: Instance segmentation map
    """
    semantic = structure[..., 0]
    v_dist = structure[..., 1]
    h_dist = structure[..., 2]
    
    # Threshold semantic map (threshold at 0 for -1/1 encoding)
    binary = (semantic > threshold).astype(np.uint8)
    
    # Compute distance from center using h_dist and v_dist
    dist_from_center = np.sqrt(h_dist**2 + v_dist**2)
    
    # Find local maxima as markers (nucleus centers)
    local_max = morphology.local_maxima(1 - dist_from_center * binary)
    markers = ndimage.label(local_max)[0]
    
    # Watershed segmentation
    instance_map = watershed(dist_from_center, markers, mask=binary)
    
    return instance_map


# ============================================================================
# MAT File Processing for MoNuSeg Dataset
# ============================================================================

def load_mat_annotation(mat_path):
    """
    Load .mat annotation file from MoNuSeg dataset
    Args:
        mat_path: Path to .mat file
    Returns:
        instance_map: Instance segmentation map
    """
    mat_data = sio.loadmat(mat_path)
    
    # MoNuSeg .mat files contain 'inst_map' key
    if 'inst_map' in mat_data:
        instance_map = mat_data['inst_map']
    else:
        raise ValueError(f"No 'inst_map' found in {mat_path}")
    
    return instance_map


# ============================================================================
# Dataset Class
# ============================================================================

class NucleiDataset(Dataset):
    """Dataset for nuclei segmentation with diffusion augmentation"""
    
    def __init__(self, root_dir, split='train', patch_size=256, stride=128, 
                 transform=None, load_structure=True):
        """
        Args:
            root_dir: Root directory (e.g., 'dataset/MoNuSeg')
            split: 'train', 'test', or 'valid'
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            transform: Albumentations transform
            load_structure: If True, load/compute nuclei structure
        """
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.load_structure = load_structure
        
        # Get image and label paths
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                   if f.endswith('.png') | f.endswith('.tif')])
        
        # Extract patches
        self.patches = []
        self._extract_patches()
    
    def _extract_patches(self):
        """Extract patches from all images"""
        print(f"Extracting patches from {len(self.image_files)} images...")
        
        for img_file in self.image_files:
            # Load image
            img_path = os.path.join(self.image_dir, img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load annotation
            label_file = img_file.replace('.png', '.mat')
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                instance_map = load_mat_annotation(label_path)
            else:
                print(f"Warning: No annotation found for {img_file}")
                continue
            
            # Extract patches
            h, w = image.shape[:2]
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    # Extract patch
                    img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    inst_patch = instance_map[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Only keep patches with nuclei
                    if inst_patch.max() > 0:
                        self.patches.append({
                            'image': img_patch,
                            'instance': inst_patch,
                            'source': img_file
                        })
        
        print(f"Extracted {len(self.patches)} patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_data = self.patches[idx]
        
        image = patch_data['image'].copy()
        instance = patch_data['instance'].copy()
        
        # Compute nuclei structure
        if self.load_structure:
            structure = instance_to_structure(instance)
        else:
            structure = None
        
        # Apply transforms
        if self.transform:
            if structure is not None:
                transformed = self.transform(image=image, mask=structure)
                image = transformed['image']
                structure = transformed['mask']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if structure is not None:
                structure = torch.from_numpy(structure).permute(2, 0, 1).float()
        
        if structure is not None:
            return {'image': image, 'structure': structure, 'instance': instance}
        else:
            return {'image': image, 'instance': instance}


# ============================================================================
# Patch Selection with K-means (as described in paper)
# ============================================================================

def select_representative_patches(dataset, num_clusters=6, samples_per_cluster=None, 
                                  feature_extractor=None):
    """
    Select representative patches using K-means clustering on features
    Args:
        dataset: NucleiDataset
        num_clusters: Number of clusters
        samples_per_cluster: Number of samples to select per cluster
        feature_extractor: Pre-trained model for feature extraction (ResNet50)
    Returns:
        selected_indices: Indices of selected patches
    """
    from torchvision import models
    
    print("Extracting features for patch selection...")
    
    # Use pre-trained ResNet50 if not provided
    if feature_extractor is None:
        feature_extractor = models.resnet50(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
        feature_extractor.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = feature_extractor.to(device)
    
    # Extract features
    features = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            
            # Normalize for ResNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            image = (image - mean) / std
            
            feat = feature_extractor(image)
            features.append(feat.cpu().numpy().flatten())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} patches")
    
    features = np.array(features)
    
    # K-means clustering
    print("Performing K-means clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Select samples closest to cluster centers
    selected_indices = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_features = features[cluster_indices]
        center = kmeans.cluster_centers_[cluster_id]
        
        # Compute distances to center
        distances = np.linalg.norm(cluster_features - center, axis=1)
        
        # Select closest samples
        if samples_per_cluster is None:
            # Select only the closest sample
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)
        else:
            # Select top-k closest samples
            k = min(samples_per_cluster, len(cluster_indices))
            closest_k_idx = cluster_indices[np.argsort(distances)[:k]]
            selected_indices.extend(closest_k_idx)
    
    print(f"Selected {len(selected_indices)} patches from {num_clusters} clusters")
    
    return selected_indices


def create_subset_dataset(dataset, percentage=0.1, num_clusters=6):
    """
    Create subset dataset with specified percentage of data
    Args:
        dataset: Full NucleiDataset
        percentage: Percentage of data to keep (0.1, 0.2, 0.5, 1.0)
        num_clusters: Number of clusters for K-means selection
    Returns:
        subset_dataset: Subset of the original dataset
    """
    if percentage == 1.0:
        return dataset
    
    # Calculate number of samples to select
    total_samples = len(dataset)
    num_samples = int(total_samples * percentage)
    samples_per_cluster = num_samples // num_clusters
    
    # Select representative patches
    selected_indices = select_representative_patches(
        dataset, 
        num_clusters=num_clusters,
        samples_per_cluster=samples_per_cluster
    )
    
    # Create subset
    class SubsetDataset(Dataset):
        def __init__(self, parent_dataset, indices):
            self.parent = parent_dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.parent[self.indices[idx]]
    
    return SubsetDataset(dataset, selected_indices)


# ============================================================================
# Data Augmentation Transforms
# ============================================================================

def get_train_transforms(image_size=256):
    """Training transforms with augmentation"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MedianBlur(blur_limit=3, p=1),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256):
    """Validation transforms without augmentation"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Data Preparation Pipeline
# ============================================================================

def prepare_datasets(root_dir='dataset/MoNuSeg', patch_size=256, stride=128, 
                     label_percentages=[0.1, 0.2, 0.5, 1.0], batch_size=4):
    """
    Prepare all datasets as described in the paper
    Args:
        root_dir: Root directory of MoNuSeg dataset
        patch_size: Size of patches
        stride: Stride for patch extraction
        label_percentages: List of label percentages to create subsets
        batch_size: Batch size for DataLoader
    Returns:
        datasets: Dictionary containing all datasets and dataloaders
    """
    # Create full training dataset
    print("Creating full training dataset...")
    train_transform = get_train_transforms(patch_size)
    full_train_dataset = NucleiDataset(
        root_dir=root_dir,
        split='train',
        patch_size=patch_size,
        stride=stride,
        transform=train_transform,
        load_structure=True
    )
    
    # Create test dataset
    print("Creating test dataset...")
    val_transform = get_val_transforms(patch_size)
    test_dataset = NucleiDataset(
        root_dir=root_dir,
        split='test',
        patch_size=patch_size,
        stride=stride,
        transform=val_transform,
        load_structure=True
    )
    
    # Create subset datasets
    datasets = {'test': test_dataset}
    dataloaders = {}
    
    for pct in label_percentages:
        print(f"\n{'='*60}")
        print(f"Creating {int(pct*100)}% labeled subset...")
        
        if pct == 1.0:
            subset = full_train_dataset
        else:
            subset = create_subset_dataset(full_train_dataset, percentage=pct)
        
        datasets[f'train_{int(pct*100)}pct'] = subset
        
        # Create DataLoader
        dataloaders[f'train_{int(pct*100)}pct'] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    # Create test DataLoader
    dataloaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"Training subsets: {[k for k in datasets.keys() if k.startswith('train')]}")
    print(f"Test dataset: {len(test_dataset)} patches")
    
    return datasets, dataloaders


# ============================================================================
# Save Patches to Disk
# ============================================================================

def save_patches_to_disk(dataset, output_dir, subset_name):
    """
    Save patches to disk in project format:
    output_dir/
        images/
        structures/
        instances/
    
    Args:
        dataset: NucleiDataset object
        output_dir: Root output directory (e.g., 'monuseg/patches256x256_128')
        subset_name: Name of subset (e.g., 'train_10pct', 'test')
    """
    subset_dir = os.path.join(output_dir, subset_name)
    
    # Create directories
    images_dir = os.path.join(subset_dir, 'images')
    structures_dir = os.path.join(subset_dir, 'structures')
    instances_dir = os.path.join(subset_dir, 'instances')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(structures_dir, exist_ok=True)
    os.makedirs(instances_dir, exist_ok=True)
    
    print(f"\nSaving {len(dataset)} patches to {subset_dir}...")
    
    # Check if this is a SubsetDataset or regular NucleiDataset
    is_subset = hasattr(dataset, 'parent') and hasattr(dataset, 'indices')
    
    # Save each patch
    for idx in range(len(dataset)):
        if (idx + 1) % 100 == 0:
            print(f"  Saved {idx + 1}/{len(dataset)} patches")
        
        # Get raw patch data (without transforms)
        if is_subset:
            # For SubsetDataset, get the actual index from parent dataset
            actual_idx = dataset.indices[idx]
            patch_data = dataset.parent.patches[actual_idx]
        else:
            # For regular NucleiDataset
            patch_data = dataset.patches[idx]
        
        image = patch_data['image']
        instance = patch_data['instance']
        
        # Compute structure
        structure = instance_to_structure(instance)
        
        # Generate filename
        filename = f"patch_{idx:05d}"
        
        # Save image as PNG
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(images_dir, f"{filename}.png"), image_bgr)
        
        # Save structure as NPY (3-channel float32)
        np.save(os.path.join(structures_dir, f"{filename}.npy"), structure.astype(np.float32))
        
        # Save instance as NPY (int32)
        np.save(os.path.join(instances_dir, f"{filename}.npy"), instance.astype(np.int32))
    
    print(f"  ✓ Saved {len(dataset)} patches")
    print(f"  Images: {images_dir}")
    print(f"  Structures: {structures_dir}")
    print(f"  Instances: {instances_dir}")


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_sample(sample, save_path=None):
    """Visualize a sample with image, structure, and instance map"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Image
    image = sample['image'].permute(1, 2, 0).numpy()
    image = (image * 0.229) + 0.485  # Denormalize
    image = np.clip(image, 0, 1)
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Structure components [semantic, v_dist, h_dist]
    structure = sample['structure'].permute(1, 2, 0).numpy()
    
    # Semantic: -1 (background), 1 (nuclei)
    im1 = axes[1].imshow(structure[..., 0], cmap='gray', vmin=-1, vmax=1)
    axes[1].set_title('Semantic ([-1, 1])')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Vertical distance: [-1, 1], use diverging colormap
    im2 = axes[2].imshow(structure[..., 1], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Vertical Distance ([-1, 1])')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Horizontal distance: [-1, 1], use diverging colormap
    im3 = axes[3].imshow(structure[..., 2], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[3].set_title('Horizontal Distance ([-1, 1])')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Instance map
    if isinstance(sample['instance'], torch.Tensor):
        instance = sample['instance'].numpy()
    else:
        instance = sample['instance']
    axes[4].imshow(instance, cmap='tab20')
    axes[4].set_title('Instance Map')
    axes[4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare nuclei datasets with patches')
    parser.add_argument('--root_dir', type=str, default='dataset/MoNuSeg',
                        help='Root directory of MoNuSeg dataset')
    parser.add_argument('--output_dir', type=str, default='monuseg/patches256x256_128',
                        help='Output directory for saved patches')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of patches')
    parser.add_argument('--stride', type=int, default=128,
                        help='Stride for patch extraction')
    parser.add_argument('--label_percentages', type=float, nargs='+', 
                        default=[0.1, 0.2, 0.5, 1.0],
                        help='Label percentages for subsets')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for DataLoader')
    parser.add_argument('--save_to_disk', action='store_true',
                        help='Save patches to disk (default: False)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample patches (default: False)')
    
    args = parser.parse_args()
    
    # Prepare datasets
    datasets, dataloaders = prepare_datasets(
        root_dir=args.root_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        label_percentages=args.label_percentages,
        batch_size=args.batch_size
    )
    
    # Save patches to disk if requested
    if args.save_to_disk:
        print(f"\n{'='*60}")
        print("Saving patches to disk...")
        print(f"Output directory: {args.output_dir}")
        
        for subset_name, dataset in datasets.items():
            save_patches_to_disk(dataset, args.output_dir, subset_name)
        
        print(f"\n{'='*60}")
        print("All patches saved successfully!")
        print(f"\nTo use for training:")
        print(f"  Structure synthesis: --data_dir {args.output_dir}/train_10pct/structures")
        print(f"  Image synthesis: --data_dir {args.output_dir}/train_10pct")
    
    # Visualize some samples if requested
    if args.visualize:
        print("\nVisualizing samples...")
        train_10pct = datasets['train_10pct']
        
        for i in range(min(3, len(train_10pct))):
            sample = train_10pct[i]
            visualize_sample(sample, save_path=f'sample_{i}.png')
        
        print("Saved sample visualizations: sample_0.png, sample_1.png, sample_2.png")
    
    print("\nDone!")