# Import các thư viện cần thiết
import os  # Để làm việc với file paths
import torch  # PyTorch
from torch.utils.data import Dataset  # Base class cho custom dataset
from PIL import Image  # Xử lý images
import numpy as np  # Numerical operations
from torchvision import transforms  # Image transformations
import cv2  # OpenCV cho image processing

# Lớp dataset cho MoNuSeg (nuclei segmentation)
class MoNuSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir  # Thư mục chứa images
        self.mask_dir = mask_dir  # Thư mục chứa masks
        self.transform = transform  # Transformations áp dụng
        self.images = os.listdir(image_dir)  # List tên files images
        self.masks = os.listdir(mask_dir)  # List tên files masks

    def __len__(self):
        return len(self.images)  # Trả về số lượng samples

    def __getitem__(self, idx):
        # Lấy path của image và mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load image và mask
        image = Image.open(img_path).convert('RGB')  # RGB image
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        # Áp dụng transformations nếu có
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask thành tensor
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        # Cho instance segmentation, cần tạo instance masks
        # Đây là simplified; thực tế cần process mask để lấy instances
        instances = self.extract_instances(mask.numpy())  # Extract instances từ mask

        # Tạo target dictionary cho Mask R-CNN
        target = {
            'boxes': torch.tensor(instances['boxes'], dtype=torch.float32),  # Bounding boxes
            'labels': torch.tensor(instances['labels'], dtype=torch.int64),  # Class labels
            'masks': torch.tensor(instances['masks'], dtype=torch.uint8),  # Instance masks
            'image_id': torch.tensor([idx]),  # ID của image
            'area': torch.tensor(instances['area'], dtype=torch.float32),  # Diện tích instances
            'iscrowd': torch.zeros(len(instances['boxes']), dtype=torch.int64)  # Không có crowd instances
        }

        return image, target  # Trả về image và target

    def extract_instances(self, mask):
        # Extract instances sử dụng connected components (đơn giản hóa)
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))  # Tìm connected components
        instances = {
            'boxes': [],  # Bounding boxes
            'labels': [],  # Labels
            'masks': [],  # Instance masks
            'area': []  # Diện tích
        }
        for i in range(1, num_labels):  # Bỏ qua background (label 0)
            instance_mask = (labels == i).astype(np.uint8)  # Mask cho instance i
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Tìm contours
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])  # Bounding box
                instances['boxes'].append([x, y, x+w, y+h])  # Thêm box
                instances['labels'].append(1)  # Class nuclei
                instances['masks'].append(instance_mask)  # Thêm mask
                instances['area'].append(cv2.contourArea(contours[0]))  # Diện tích
        return instances  # Trả về dictionary instances


# Hàm tạo transformations
def get_transform(train):
    transforms_list = []  # List transformations
    transforms_list.append(transforms.ToTensor())  # Convert PIL to tensor
    if train:  # Nếu training, thêm augmentations
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))  # Random horizontal flip
    return transforms.Compose(transforms_list)  # Compose transformations