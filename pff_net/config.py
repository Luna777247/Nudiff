import torch  # Import PyTorch để sử dụng device

# Cấu hình mô hình
BACKBONE = 'resnet50'  # Backbone sử dụng: 'resnet50' hoặc 'resnet101'
NUM_CLASSES = 2  # Số classes: Background + nuclei (điều chỉnh cho multi-class)
INPUT_SIZE = (512, 512)  # Kích thước input image
BATCH_SIZE = 4  # Batch size cho training
LEARNING_RATE = 1e-4  # Learning rate
EPOCHS = 100  # Số epochs training

# Trọng số cho các loss components
LOSS_WEIGHTS = {
    'detection': 1.0,  # Trọng số loss detection (bbox)
    'mask': 1.0,  # Trọng số loss mask
    'semantic_main': 1.0,  # Trọng số loss semantic chính
    'semantic_instance': 1.0,  # Trọng số loss semantic từ instance branch
    'consistency': 0.1,  # Trọng số loss consistency giữa hai semantic heads
    'quality': 0.1  # Trọng số loss mask quality
}

# Đường dẫn dữ liệu (điều chỉnh theo dataset của bạn)
# NOTE: These defaults are overridden by train.py and evaluate.py command line args
DATA_ROOT = '../dataset'  # Thư mục gốc dataset
TRAIN_IMAGES = f'{DATA_ROOT}/MoNuSeg/train/images'  # images (lowercase)
TRAIN_MASKS = f'{DATA_ROOT}/MoNuSeg/train/labels'  # labels (not Masks)
VAL_IMAGES = f'{DATA_ROOT}/MoNuSeg/test/images'  # test as validation
VAL_MASKS = f'{DATA_ROOT}/MoNuSeg/test/labels'  # test labels

# Device sử dụng (GPU nếu có, ngược lại CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Các cấu hình khác
NUM_WORKERS = 4  # Số workers cho DataLoader
SAVE_DIR = './checkpoints'  # Thư mục lưu checkpoints