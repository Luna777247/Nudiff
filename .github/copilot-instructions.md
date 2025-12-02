# AI Coding Instructions for NuDiff Project

## Project Overview
NuDiff implements diffusion-based data augmentation for nuclei image segmentation (MICCAI 2023). It uses a two-stage synthesis pipeline: (1) **unconditional diffusion** generates nuclei structures (semantic + distance transforms), then (2) **conditional diffusion** generates realistic histopathology images given structures. Synthetic augmented data trains Hover-Net for improved segmentation with limited labeled data (achieves full-supervision results with only 10% labeled data).

## Architecture Essentials

**Core Pipeline**: Instance maps → Structures → Unconditional diffusion → Synthetic structures → Conditional diffusion → Synthetic images → HoverNet training

**Three Main Components**:
1. **nudiff/struct_syn** - Unconditional U-Net diffusion model synthesizing nuclei structures (3-channel: semantic binary map + normalized vertical/horizontal distance transforms from nucleus centroid)
2. **nudiff/image_syn** - Conditional U-Net diffusion with SPADE modules accepting structure guidance; uses classifier-free guidance during sampling (w parameter controls structure adherence)
3. **hover_net** - Instance segmentation network using horizontal/vertical distance maps for localization; trained on synthetic+real augmented data

**Data Format Invariants**:
- Instance maps: numpy arrays shape (H, W) with instance IDs 1-N (0=background), loaded from .npy files as 'inst_map' (not .mat)
- Structures: shape (H, W, 3) floats; [semantic, v_dist, h_dist] where distances are normalized by nucleus bounding box
- Patches: 4D numpy (B, H, W, C) stored as .npy files from `hover_net/extract_patches.py` with combined [image, semantic, type] channels

## Structure Creation & Reconstruction

**Creating Structures from Instance Maps** (`nuclei_data_prep.py`):
```python
# Semantic map: binary mask where nuclei exist
semantic = (instance_map > 0).astype(np.uint8)

# Horizontal/vertical distance transforms from nucleus centroids
from nudiff.image_syn.utils.datasets import get_hv
hv_map = get_hv(instance_map)  # Returns (H, W, 2) with [v_dist, h_dist]

# Combine into 3-channel structure
structure = np.stack([semantic, hv_map[..., 0], hv_map[..., 1]], axis=-1)
```

**Reconstructing Instance Maps from Structures** (updated logic from `generate_synthetic.py`):
```python
from scipy.ndimage import maximum_filter

def structure_to_instance(structure):
    semantic, v_dist, h_dist = structure[..., 0], structure[..., 1], structure[..., 2]
    
    # Threshold semantic at 0.5 for binary mask
    binary = (semantic > 0.5).astype(np.uint8)
    
    # Distance from center (higher values = farther from center)
    dist_map = np.sqrt(v_dist**2 + h_dist**2)
    
    # Find local maxima as nucleus markers using maximum_filter
    local_max = (dist_map == maximum_filter(dist_map, size=5))
    markers = ndimage.label(local_max * binary)[0]
    
    # Watershed segmentation
    instance_map = watershed(dist_map, markers, mask=binary)
    return instance_map
```

## Critical Workflows

**Full Pipeline (see `nudiff_paper_implementation_monuseg.ipynb`)**:
1. Dataset preparation: download MoNuSeg (30 train + 14 test images) or Kumar datasets
2. K-means patch clustering: extract 256×256 patches (stride 128), cluster by ResNet50 features to select representative images for 10%/20%/50%/100% subsets
3. Structure computation: extract semantic + distance transforms from instance maps
4. Unconditional training: `python scripts/struct_syn/struct_train.py --data_dir=structures/path --image_size=1000` (single GPU or distributed)
5. Structure sampling: generate synthetic structures with guidance scale control
6. Conditional training: `python scripts/image_syn/train.py --data_dir=image_structure_pairs/path` (uses CycleTrainLoop for image-structure consistency)
7. Image synthesis: sample paired synthetic images from structures with classifier-free guidance
8. HoverNet training: standard supervised training on augmented dataset; visualize with `--view=train` flag

**Quick Testing**: Notebooks isolate each stage - can execute cells independently to debug specific components (subset creation, structure quality, synthesis quality, segmentation metrics).

**Distributed Training**: Both diffusion scripts support multi-GPU via environment variables (OMP_NUM_THREADS=4 for optimal throughput); use `dist_util.setup_dist()` and `dist_util.dev()` for device/rank handling.

## Key Conventions & Patterns

**Config Management**:
- `hover_net/config.py`: Central Config class with dataset paths, shape info (aug_shape, act_shape, out_shape), model mode (original/fast), type_classification flag
- Image size must match diffusion checkpoint: 1000px for full images, 256px for patches
- Batch size/microbatch tuning in script args affects memory; reduce batch_size if OOM

**Dataset Abstraction** (`hover_net/dataset.py`):
- Implement `__AbstractDataset` for new datasets with `load_img()` and `load_ann()` methods; Kumar, CoNSeP, CPM17 provided
- Multi-channel annotation: numpy arrays shape (H, W, C) where C=1 for instance only or C≥2 for [instance, type, ...]

**Training Loops**:
- `struct_syn/train_util.py::TrainLoop`: Handles EMA, schedule sampling, visualization of denoising steps
- `image_syn/src/run_desc.py::CycleTrainLoop`: Trains both conditional and unconditional models jointly via random condition dropping
- Sampling uses `respace.SpacedDiffusion` with configurable steps (e.g., 50-250 steps trades speed for quality)

**Metrics & Evaluation**:
- Instance segmentation: AJI (Aggregated Jaccard Index) and Dice computed by `hover_net/compute_stats.py`
- Visualize predictions via `hover_net/run_infer.py` or notebook cells; overlays predictions on images
- Instance ID assignment via watershed on distance transforms (not direct network output)

**Environment Setup**:
- Thread pinning critical for data loading: set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS to match CPU cores (typically 4)
- CUDA synchronization points: diffusion sampling and HoverNet inference can be bottlenecked by GPU-CPU transfer; batch multiple samples
- Checkpoints: saved as torch.pt files with model state_dict and optimizer state; loading requires matching model architecture

**Dependencies & Fixes**:
- Use `albumentations` for augmentations, but fix deprecated transforms: `A.Flip()` → `A.HorizontalFlip()` and `A.VerticalFlip()`
- Ensure `scipy.ndimage` and `skimage.segmentation.watershed` for reconstruction
- For reconstruction, prefer `maximum_filter` over `morphology.local_maxima` for better nucleus detection (improves from ~37 to ~233 nuclei)

## Code Patterns to Follow

When implementing new features:
- **Adding new diffusion configuration**: Edit `nudiff/struct_syn/script_util.py::model_and_diffusion_defaults()` and pass args via `add_dict_to_argparser()`
- **Custom data loading**: Extend `nudiff/struct_syn/datasets.py::MaskDataset` or `nudiff/image_syn/utils/datasets.py::ImageDataset` to override `_load_image()`
- **New augmentation**: Apply to batch_loader in training loop via `dataset.load_data()` which applies random_flip/rotate by default; extend for domain-specific transforms
- **Evaluation**: Use `hover_net/metrics/stats_utils.py` which handles matched instance pairs and AJI computation; respects semantic type when type_classification=True

## Example Commands

```bash
# Setup and validation
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"

# Extract patches (required before any training)
cd hover_net
python extract_patches.py  # See __main__ block for hardcoded paths; edit dataset_info dict

# Train unconditional diffusion (structure synthesis)
python scripts/struct_syn/struct_train.py \
  --data_dir dataset/structures \
  --image_size 1000 \
  --batch_size 2 \
  --lr 1e-4 \
  --save_interval 500 \
  --log_interval 100

# Sample structures (modify guidance_scale in notebook to control diversity)
python scripts/struct_syn/struct_sample.py \
  --model_path checkpoints/struct_final.pt \
  --num_samples 100 \
  --image_size 1000

# Train conditional diffusion (image synthesis given structures)
python scripts/image_syn/train.py \
  --data_dir dataset/image_structure_pairs \
  --image_size 1000 \
  --batch_size 2 \
  --lr 1e-4 \
  --dropout_rate 0.1  # Controls unconditional vs conditional mixing

# Train HoverNet segmentation
cd hover_net
python run_train.py --gpu=0  # Uses Config paths; check config.py before running
```

## Critical Implementation Details

1. **Structure → Instance Map conversion**: Use `maximum_filter(dist_map, size=5)` for local maxima detection instead of `morphology.local_maxima`; watershed algorithm applied to distance transform peaks; no additional training needed. See `analyze_structure_detail.py` or `generate_synthetic.py`.
2. **Guidance-free sampling**: At inference, set guidance_scale=1.0 for unconditional, >1.0 for structured generation. HoverNet inference uses deterministic instance map from structure.
3. **Memory bottleneck**: Image diffusion with 1000×1000 images requires careful batch sizing; patch-based training uses 256×256 subsampled versions. Check microbatch parameter if CUDA OOM.
4. **Checkpoint compatibility**: Models trained with fp16_util conversion work with standard state_dict loading; mismatch in in_channels or num_res_blocks causes loading failure.
5. **Dataset subset reproducibility**: K-means clustering uses random_state=42; subset selection deterministic from cluster centers for reproducible train/val splits.</content>
<parameter name="filePath">d:\project\Nudiff\.github\copilot-instructions.md