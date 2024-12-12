import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur as torch_gaussian_blur

def estimate_deep_flow(frame1, frame2):
    """Estimate optical flow using RAFT or similar deep learning model"""
    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    # TODO: Implement RAFT or similar deep flow estimation
    # For now, fallback to Farneback
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray,
        None, 0.5, 5, 15, 3, 7, 1.5, 0
    )
    return torch.from_numpy(flow.transpose(2, 0, 1))

def estimate_optical_flow(frame1, frame2, method='farneback'):
    """Estimate optical flow using cv2's implementations"""
    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    if method == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray,
            None, 0.5, 5, 15, 3, 7, 1.5, 0
        )
    elif method == 'dualtvl1':
        # DualTVL1 optical flow (more accurate but slower)
        flow_calculator = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = flow_calculator.calc(frame1_gray, frame2_gray, None)
    
    return torch.from_numpy(flow.transpose(2, 0, 1))

def warp_image(image, flow):
    """Warp image using cv2's remap"""
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(flow):
        flow = flow.cpu().numpy()
    
    h, w = flow.shape[1:3]
    
    # Create meshgrid
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Add flow to coordinates
    flow = flow.transpose(1, 2, 0)
    new_x = x + flow[..., 0]
    new_y = y + flow[..., 1]
    
    # Perform warping using cv2.remap
    warped = cv2.remap(image.transpose(1, 2, 0), 
                      new_x, 
                      new_y, 
                      cv2.INTER_LINEAR, 
                      borderMode=cv2.BORDER_REPLICATE)
    
    return torch.from_numpy(warped.transpose(2, 0, 1))

def compute_flow_gradients(flow):
    """Compute spatial gradients of the optical flow"""
    # Compute gradients in x and y directions
    dx = torch.zeros_like(flow)
    dy = torch.zeros_like(flow)
    
    dx[..., 1:] = flow[..., 1:] - flow[..., :-1]  # gradient in x direction
    dy[..., 1:, :] = flow[..., 1:, :] - flow[..., :-1, :]  # gradient in y direction
    
    # Compute gradient magnitude
    gradient_magnitude = torch.sqrt(dx.pow(2).sum(dim=0) + dy.pow(2).sum(dim=0))
    return gradient_magnitude

def gaussian_blur(tensor, kernel_size=5, sigma=0.8):
    """Apply Gaussian blur using cv2"""
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().numpy()
    
    # Ensure proper shape for cv2
    if len(tensor.shape) == 2:
        blurred = cv2.GaussianBlur(tensor, (kernel_size, kernel_size), sigma)
    else:
        blurred = cv2.GaussianBlur(tensor.transpose(1, 2, 0), 
                                  (kernel_size, kernel_size), 
                                  sigma)
        blurred = blurred.transpose(2, 0, 1)
    
    return torch.from_numpy(blurred)

def compute_consistency_mask(flow_forward, flow_backward, threshold=0.01):
    """Improved consistency mask computation following reference implementation"""
    if not isinstance(flow_forward, torch.Tensor):
        flow_forward = torch.from_numpy(flow_forward).float()
    if not isinstance(flow_backward, torch.Tensor):
        flow_backward = torch.from_numpy(flow_backward).float()
    
    # Ensure flow tensors are on the same device
    device = flow_forward.device
    flow_backward = flow_backward.to(device)
    
    # Warp backward flow with forward flow
    warped_backward = warp_image(flow_backward.unsqueeze(0), flow_forward).squeeze(0)
    
    # Calculate consistency error
    consistency_error = torch.norm(flow_forward + warped_backward, dim=0)
    
    # Detect motion boundaries
    flow_gradients = compute_flow_gradients(flow_forward)
    motion_boundary_mask = (flow_gradients > threshold)
    
    # Combine consistency and motion boundary masks
    mask = ((consistency_error < threshold) & ~motion_boundary_mask).float()
    
    # Apply Gaussian smoothing to mask
    mask = gaussian_blur(mask, kernel_size=5, sigma=0.8)
    return mask.clamp(0, 1)

def spatial_gradient(flow):
    """Compute spatial gradients using cv2 Sobel"""
    if torch.is_tensor(flow):
        flow = flow.cpu().numpy()
    
    # Calculate gradients using cv2.Sobel
    grad_x = cv2.Sobel(flow, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(flow, cv2.CV_32F, 0, 1, ksize=3)
    
    return torch.from_numpy(np.sqrt(grad_x**2 + grad_y**2))

def temporal_consistency_loss(current_frame, warped_prev_frame, mask):
    """Computes temporal consistency loss between current and warped previous frame"""
    if not isinstance(current_frame, torch.Tensor):
        current_frame = torch.from_numpy(current_frame).float()
    if not isinstance(warped_prev_frame, torch.Tensor):
        warped_prev_frame = torch.from_numpy(warped_prev_frame).float()
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask).float()
    
    # Ensure all tensors are on the same device
    device = current_frame.device
    warped_prev_frame = warped_prev_frame.to(device)
    mask = mask.to(device)
    
    return torch.mean(mask * torch.abs(current_frame - warped_prev_frame))

def compute_flow_weights(flow_forward, flow_backward):
    """Compute flow reliability weights similar to consistencyChecker.cpp"""
    if not isinstance(flow_forward, torch.Tensor):
        flow_forward = torch.from_numpy(flow_forward).float()
    if not isinstance(flow_backward, torch.Tensor):
        flow_backward = torch.from_numpy(flow_backward).float()
    
    # Compute motion boundaries
    flow_gradients = spatial_gradient(flow_forward)
    motion_boundaries = torch.norm(flow_gradients, dim=0) > 0.01
    
    # Compute consistency check
    consistency_mask = compute_consistency_mask(flow_forward, flow_backward)
    
    # Combine masks and apply smoothing
    weights = (~motion_boundaries & consistency_mask).float()
    weights = gaussian_blur(weights, kernel_size=5, sigma=0.8)
    return weights.clamp(0, 1) 

def get_temporal_weight(pass_idx):
    """Compute temporal consistency weight based on pass number"""
    base_weight = 5e2  # Base temporal weight
    return base_weight * (1 + pass_idx / 10)  # Increase weight in later passes

def blend_neighboring_frames(current_frame, prev_stylized, next_stylized, blend_weight, flow_weights=None):
    """Blend current frame with neighboring stylized frames using optical flow"""
    device = current_frame.device
    blended = current_frame.clone()
    
    if prev_stylized is not None:
        # Compute forward flow and weights
        flow_forward = estimate_optical_flow(
            current_frame.cpu().numpy(), 
            prev_stylized.cpu().numpy()
        ).to(device)
        
        if flow_weights is None:
            flow_weights = compute_flow_weights(
                flow_forward,
                estimate_optical_flow(
                    prev_stylized.cpu().numpy(),
                    current_frame.cpu().numpy()
                ).to(device)
            )
        
        # Warp and blend previous frame
        warped_prev = warp_image(prev_stylized.unsqueeze(0), flow_forward)
        blended = blended * (1 - blend_weight * flow_weights) + \
                 warped_prev.squeeze(0) * (blend_weight * flow_weights)
    
    if next_stylized is not None:
        # Compute backward flow and weights
        flow_backward = estimate_optical_flow(
            current_frame.cpu().numpy(),
            next_stylized.cpu().numpy()
        ).to(device)
        
        if flow_weights is None:
            flow_weights = compute_flow_weights(
                flow_backward,
                estimate_optical_flow(
                    next_stylized.cpu().numpy(),
                    current_frame.cpu().numpy()
                ).to(device)
            )
        
        # Warp and blend next frame
        warped_next = warp_image(next_stylized.unsqueeze(0), flow_backward)
        blended = blended * (1 - blend_weight * flow_weights) + \
                 warped_next.squeeze(0) * (blend_weight * flow_weights)
    
    return blended

def compute_blend_weight(pass_idx, num_passes):
    """Compute adaptive blending weight based on pass number"""
    base_weight = 1.0
    if pass_idx < num_passes // 3:
        return base_weight * 0.5  # Less blending in early passes
    elif pass_idx > 2 * num_passes // 3:
        return base_weight * 0.8  # More blending in later passes
    return base_weight
