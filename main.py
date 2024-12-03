import cv2
import torch
from torchvision import transforms
from PIL import Image
import os

model = torch.load("model.pth")
model.eval()

# Transformation pipeline for images
transform = transforms.Compose([
    transforms.Resize((720, 1280)),  # Adjust based on video resolution
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Load video
cap = cv2.VideoCapture('input_video.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output directory for frames
os.makedirs('output_frames', exist_ok=True)

# Process each frame
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(frame_pil).unsqueeze(0)

    # Apply style transfer
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Save styled frame
    cv2.imwrite(f'output_frames/frame_{i:04d}.jpg', output_image)

cap.release()

# Combine frames back into video
os.system("ffmpeg -r {fps} -i output_frames/frame_%04d.jpg -vcodec libx264 output_video.mp4")
