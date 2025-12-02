# Import các thư viện cần thiết
import torch  # PyTorch
from model import PFFNet  # Model PFFNet
from config import *  # Config
from PIL import Image  # Xử lý images
from torchvision import transforms  # Image transforms
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numerical operations

# Hàm load model từ checkpoint
def load_model(checkpoint_path, device):
    model = PFFNet(NUM_CLASSES, BACKBONE).to(device)  # Khởi tạo model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # Load weights
    model.eval()  # Set to evaluation mode
    return model  # Trả về model

# Hàm preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Load và convert RGB
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])
    return transform(image).unsqueeze(0)  # Thêm batch dimension

# Hàm inference
def infer(model, image_tensor, device):
    with torch.no_grad():  # Không tính gradient
        image_tensor = image_tensor.to(device)  # Move to device
        outputs = model(image_tensor)  # Forward pass
        return outputs  # Trả về outputs

# Hàm visualize kết quả
def visualize_results(image, outputs):
    semantic_logits = outputs['semantic_logits']  # Semantic predictions
    instance_output = outputs['instance_output']  # Instance outputs

    # Semantic segmentation
    semantic_pred = torch.argmax(semantic_logits, dim=1).squeeze().cpu().numpy()  # Argmax để lấy class

    # Instance masks
    masks = instance_output[0]['masks'].cpu().numpy()  # Instance masks
    boxes = instance_output[0]['boxes'].cpu().numpy()  # Bounding boxes
    scores = instance_output[0]['scores'].cpu().numpy()  # Confidence scores

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Tạo figure với 2 subplots
    ax[0].imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())  # Hiển thị original image
    ax[0].set_title('Original Image')  # Title

    ax[1].imshow(semantic_pred, cmap='gray')  # Hiển thị semantic segmentation
    ax[1].set_title('Semantic Segmentation')  # Title

    for mask in masks:  # Overlay instance masks
        ax[1].imshow(mask, alpha=0.5, cmap='Blues')  # Alpha blending

    plt.show()  # Hiển thị plot

# Main function
if __name__ == "__main__":
    device = DEVICE  # Device
    model = load_model('./checkpoints/pffnet_epoch_100.pth', device)  # Load model (điều chỉnh path)

    image_path = '../dataset/MoNuSeg/test/Images/test_image.png'  # Path image test (điều chỉnh)
    image_tensor = preprocess_image(image_path)  # Preprocess

    outputs = infer(model, image_tensor, device)  # Inference
    visualize_results(image_tensor, outputs)  # Visualize