import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image  # Still used for tensor conversion helper
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import ffmpeg
import cv2

# Optimization parameters
MAX_TRUE_SHIFT = 5.0  # Maximum absolute value for the random ground truth shifts (for comparison plot)
LEARNING_RATE = 0.1
NUM_ITERATIONS = 100  # Iterations to find shifts for each frame (reduced from 300)
SEED = 42  # for reproducibility

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- RowTemporalShiftInterp Module ---
class RowTemporalShiftInterp(nn.Module):
    """
    A PyTorch module that shifts each row of an input image tensor horizontally
    and interpolates it temporally based on a learnable parameter.
    Works for grayscale or RGB.

    The amount of horizontal shift for each row and the temporal interpolation
    weight are determined by learnable parameters internal to the module.

    Args:
        height (int): The expected height of the input image tensors.

    Input Shape:
        (B, C, H, W), where B is batch size, C is number of channels, H is height, W is width.

    Output Shape:
        (B, C, H, W), same shape as the input.
    """
    def __init__(self, height: int):
        super().__init__()
        if not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")
        self.height = height
        # Learnable parameters for horizontal shift (one per row)
        self.row_shifts = nn.Parameter(torch.zeros(height))
        # Learnable parameters for temporal shift (one per row)
        self.row_temporal_shifts = nn.Parameter(torch.zeros(height//2))
        self.row_temporal_shifts.data.clamp_(-1.0, 1.0) # Initialize within the valid range

    def forward(self, frame_stack: torch.Tensor) -> torch.Tensor:
        B, _, _, H, W = frame_stack.shape
        device = frame_stack.device
        dtype = frame_stack.dtype

        if H != self.height:
            raise ValueError(f"Input tensor height {H} does not match module's expected height {self.height}")

        # 1. Horizontal Shift
        shifts = self.row_shifts.to(device)
        base_y = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype) if H > 0 else torch.zeros(0, device=device, dtype=dtype)
        base_x = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype) if W > 0 else torch.zeros(0, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing='ij')
        norm_shifts = shifts * (2.0 / W) if W > 0 else torch.zeros_like(shifts)
        shifted_norm_x = grid_x - norm_shifts.view(H, 1)
        grid = torch.stack((shifted_norm_x, grid_y), dim=-1).unsqueeze(0).expand(B, H, W, 2)

        def horizontal_shift(frame):
            return  F.grid_sample(frame, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        horizontally_shifted_prev = horizontal_shift(frame_stack[:, :, 0, :, :])
        horizontally_shifted_current = horizontal_shift(frame_stack[:, :, 1, :, :])
        horizontally_shifted_next = horizontal_shift(frame_stack[:, :, 2, :, :])

        # 2. Per-Row Temporal Interpolation
        alpha_per_row = torch.zeros(H, device=device, dtype=dtype)
        # even lines are unchanged (0), odd lines are shifted
        alpha_per_row[1::2] = self.row_temporal_shifts.to(device)
        alpha_per_row = alpha_per_row.view(B, 1, H, 1) 

        # Interpolate between the horizontally shifted previous, current and next frames
        output = torch.where(
        alpha_per_row <= 0,
        (1 + alpha_per_row) * horizontally_shifted_current - alpha_per_row * horizontally_shifted_prev,
        (1 - alpha_per_row) * horizontally_shifted_current + alpha_per_row * horizontally_shifted_next
    )        
        return output

    def extra_repr(self) -> str:
        return f'height={self.height}'

# --- Total Variation Loss Function (Handles multi-channel input correctly) ---
def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the anisotropic Total Variation loss for a batch of images."""
    dv = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    tv = torch.sum(dv)
    return tv

# --- Helper function for NumPy frame to Tensor conversion ---
def frame_to_tensor(frame_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts a NumPy frame (H, W, C) uint8 to a Tensor (1, C, H, W) float32 [0,1]."""
    img_pil = Image.fromarray(frame_np)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img_pil).unsqueeze(0)  # Add batch dimension (B=1)
    return img_tensor.to(device)

# --- Helper function for Tensor to NumPy frame conversion ---
def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Converts a Tensor (1, C, H, W) float32 [0,1] back to NumPy frame (H, W, C) uint8."""
    img = tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def frame_to_tensor_cv2(frame_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts a NumPy frame (H, W, C) uint8 to a Tensor (1, C, H, W) float32 [0,1]."""
    frame_tensor = torch.from_numpy(frame_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return frame_tensor.to(device)

def tensor_to_frame_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Converts a Tensor (1, C, H, W) float32 [0,1] back to NumPy frame (H, W, C) uint8."""
    img = (tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img

def timecode_to_seconds(timecode_str):
    """Converts a 'HH:MM:SS' timecode string to seconds."""
    parts = list(map(int, timecode_str.split(':')))
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def optimize_frame_shifts_temporal_spatial(frame_stack, 
                                           height, 
                                           device, 
                                           learning_rate=LEARNING_RATE, 
                                           num_iterations=NUM_ITERATIONS):
    """Optimize horizontal and per-row temporal alpha."""
    frame_shifter = RowTemporalShiftInterp(height=height).to(device)
    with torch.no_grad():
        frame_shifter.row_shifts.data = torch.randn_like(frame_shifter.row_shifts) * 0.01
        frame_shifter.row_temporal_shifts.data = torch.randn_like(frame_shifter.row_temporal_shifts) * 0.2
        frame_shifter.row_temporal_shifts.data.clamp_(-1.0, 1.0)

    frame_optimizer = optim.Adam(frame_shifter.parameters(), lr=learning_rate)
    frame_losses = []
    for i in range(num_iterations):
        frame_optimizer.zero_grad()
        output = frame_shifter(frame_stack)
        loss = total_variation_loss(output)
        loss.backward()
        frame_optimizer.step()
        frame_losses.append(loss.item())
    return frame_shifter, frame_losses

# --- Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restoration')
    parser.add_argument('--inputvideo', type=str, default='ORIGINAL-MGCAA0035917--AG_JAQ_01_EXT.mkv')
    parser.add_argument('--outputvideo', type=str, default='output.mp4')
    parser.add_argument('--startTime', type=str, default='00:02:00', help='start timecode')
    parser.add_argument('--nbFrames', type=int, default=100, help='number of frames to process')
    opt = parser.parse_args()

    INPUT_VIDEO_PATH = opt.inputvideo
    OUTPUT_VIDEO_PATH = opt.outputvideo
    START_TIME = opt.startTime
    NB_FRAMES = opt.nbFrames

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or other codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    start_frame = 0
    if START_TIME:
        start_seconds = timecode_to_seconds(START_TIME)
        start_frame = int(start_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting from frame: {start_frame} (time: {START_TIME})")
        
    frames_processed = 0
    frame_buffer = []
    while frames_processed < NB_FRAMES:
        ret, frame_np = cap.read()
        if not ret:
            break
        frame_buffer.append(frame_to_tensor_cv2(frame_np, device))
        frames_processed += 1
    cap.release()

    for current_i in range(len(frame_buffer)):
        prev_i = max(0, current_i - 1)
        next_i = min(len(frame_buffer) - 1, current_i + 1)
        
        prev_frame_tensor = frame_buffer[prev_i]
        current_frame_tensor = frame_buffer[current_i]
        next_frame_tensor = frame_buffer[next_i]

        frame_stack = torch.stack([prev_frame_tensor, current_frame_tensor, next_frame_tensor], dim=2) # Shape (B, C, 3, H, W)

        optimized_shifter, frame_losses = optimize_frame_shifts_temporal_spatial(
            frame_stack,
            height=height,
            device=device
        )

        with torch.no_grad():
            output_frame_tensor = optimized_shifter(frame_stack)

        output_frame_np = tensor_to_frame(output_frame_tensor)
        out.write(output_frame_np)
        if (current_i % 100) == 0:
            print(f"Processed frame {current_i}. \
                    Final TV loss: {frame_losses[-1]:.4f}, \
                    Temporal Alpha (min/max): \
                    {optimized_shifter.row_temporal_shifts.min().item():.3f}/{optimized_shifter.row_temporal_shifts.max().item():.3f}")
            
    out.release()
    print(f"\nFinished processing. {frames_processed} frames written to {OUTPUT_VIDEO_PATH}")
