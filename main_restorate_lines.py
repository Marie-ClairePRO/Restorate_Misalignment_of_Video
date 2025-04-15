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
from math import inf

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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

        # Learnable parameters for temporal shift (one per odd row)
        self.row_temporal_shifts = nn.Parameter(torch.zeros(height//2))
        self.row_temporal_shifts.data.clamp_(0.0, 1.0) # Initialize within the valid range

    def forward(self, frame_stack: torch.Tensor) -> torch.Tensor:
        B, _, _, H, W = frame_stack.shape
        device = frame_stack.device
        dtype = frame_stack.dtype

        if H != self.height:
            raise ValueError(f"Input tensor height {H} does not match module's expected height {self.height}")

        # Horizontal Shift
        shifts = self.row_shifts.to(device)
        base_y = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype) if H > 0 else torch.zeros(0, device=device, dtype=dtype)
        base_x = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype) if W > 0 else torch.zeros(0, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing='ij')
        norm_shifts = shifts * (2.0 / W) if W > 0 else torch.zeros_like(shifts)
        shifted_norm_x = grid_x - norm_shifts.view(H, 1)
        grid = torch.stack((shifted_norm_x, grid_y), dim=-1).unsqueeze(0).expand(B, H, W, 2)
        
        prev = frame_stack[:, :, 0, :, :]
        curr = frame_stack[:, :, 1, :, :]

        # Per-Row Temporal Interpolation
        alpha_per_row = torch.zeros(H, device=device, dtype=dtype)
        # even lines are unchanged (0), odd lines are shifted
        alpha_per_row[1::2] = self.row_temporal_shifts.to(device)
        alpha_per_row = alpha_per_row.view(B, 1, H, 1) 

        # Interpolate between the horizontally shifted previous and current frames
        frame = (1 - alpha_per_row) * curr + alpha_per_row * prev

        output = F.grid_sample(frame, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return output

    def extra_repr(self) -> str:
        return f'height={self.height}'

# --- Total Variation Loss Function (Handles multi-channel input correctly) ---
def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes the anisotropic Total Variation loss for a batch of images."""
    dv = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    tv = torch.sum(dv)
    return tv

def frame_to_tensor_PIL(frame_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts a NumPy frame (H, W, C) uint8 to a Tensor (1, C, H, W) float32 [0,1]."""
    img_pil = Image.fromarray(frame_np)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img_pil).unsqueeze(0)  # Add batch dimension (B=1)
    return img_tensor.to(device)

def tensor_to_frame_PIL(tensor: torch.Tensor) -> np.ndarray:
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
                                           learning_rate, 
                                           num_iterations,
                                           max_shift):
    """Optimize horizontal and per-row temporal alpha."""
    frame_shifter = RowTemporalShiftInterp(height=height).to(device)
    with torch.no_grad():
        frame_shifter.row_shifts.data = torch.randn_like(frame_shifter.row_shifts) * 0.01
        frame_shifter.row_shifts.data.clamp_(-max_shift, max_shift)
        frame_shifter.row_temporal_shifts.data = torch.randn_like(frame_shifter.row_temporal_shifts) * 0.2
        frame_shifter.row_temporal_shifts.data.clamp_(0., 1.0)

    frame_optimizer = optim.Adam(frame_shifter.parameters(), lr=learning_rate)
    frame_losses = []
    for i in range(num_iterations):
        frame_optimizer.zero_grad()
        output = frame_shifter(frame_stack)
        loss = total_variation_loss(output)
        loss.backward()
        frame_optimizer.step()
        with torch.no_grad():
            frame_shifter.row_temporal_shifts.data.clamp_(0.0, 1.0)
            frame_shifter.row_shifts.data.clamp_(-max_shift, max_shift)
        frame_losses.append(loss.item())
    return frame_shifter, frame_losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restoration')
    parser.add_argument('--inputVideo', type=str, default='input/video.mkv')
    parser.add_argument('--outputVideo', type=str, default='outputs/output.mp4')
    parser.add_argument('--startTime', type=str, default='00:00:00', help='start timecode in format hh:mm:ss')
    parser.add_argument('--nbFrames', type=float, default=inf, help='number of frames to process')
    parser.add_argument('--max_shift', type=int, default=5, help='Maximum absolute value shifts')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations of optimization')
    opt = parser.parse_args()

    INPUT_VIDEO_PATH = opt.inputVideo
    OUTPUT_VIDEO_PATH = opt.outputVideo
    START_TIME = opt.startTime
    NB_FRAMES = opt.nbFrames
    
    # Optimization parameters
    LEARNING_RATE = opt.learning_rate
    NUM_ITERATIONS = opt.num_iterations 
    MAX_SHIFT = opt.max_shift

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
    prev_frame_tensor = None

    while frames_processed < NB_FRAMES:
        ret, frame_np = cap.read()
        if not ret:
            break

        current_frame_tensor = frame_to_tensor_cv2(frame_np, device)

        if prev_frame_tensor is None:
            prev_frame_tensor = current_frame_tensor.clone()

        frame_stack = torch.stack([prev_frame_tensor, current_frame_tensor], dim=2) # Shape (B, C, 2, H, W)

        optimized_shifter, frame_losses = optimize_frame_shifts_temporal_spatial(
            frame_stack,
            height=height,
            device=device,
            learning_rate=LEARNING_RATE,
            num_iterations=NUM_ITERATIONS,
            max_shift=MAX_SHIFT
        )

        with torch.no_grad():
            output_frame_tensor = optimized_shifter(frame_stack)

        output_frame_np = tensor_to_frame_cv2(output_frame_tensor)
        out.write(output_frame_np)
        
        prev_frame_tensor = current_frame_tensor.clone()
        frames_processed += 1
        if (frames_processed % 100) == 0:
            min_temp_shift = optimized_shifter.row_temporal_shifts.min().item()
            max_temp_shift = optimized_shifter.row_temporal_shifts.max().item()
            min_spa_shift = optimized_shifter.row_shifts.min().item()
            max_spa_shift = optimized_shifter.row_shifts.max().item()

            print(f"Processed frame {frames_processed}. \
                    Final TV loss: {frame_losses[-1]:.4f}, \
                    Spatial Alpha (min/max): {min_spa_shift:.2f}/{max_spa_shift:.2f} \
                    Temporal Alpha (min/max): {min_temp_shift:.2f}/{max_temp_shift:.2f} ")
            
    out.release()
    print(f"\nFinished processing. {frames_processed} frames written to {OUTPUT_VIDEO_PATH}")
