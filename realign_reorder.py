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

# --- RowShiftInterp Module (Handles multi-channel input correctly) ---
class RowShiftInterp(nn.Module):
    """
    A PyTorch module that shifts each row of an input image tensor horizontally
    using bilinear interpolation for subpixel accuracy. Works for grayscale or RGB.

    The amount of shift for each row is determined by learnable parameters
    internal to the module. Shifts can be fractional. Sampling outside the
    original image boundaries results in zeros.

    Args:
        height (int): The expected height of the input image tensors. This is
                      needed to initialize the learnable shift parameters correctly.

    Input Shape:
        (B, C, H, W), where B is batch size, C is number of channels (1 for gray, 3 for RGB),
        H is height, and W is width. The input H must match the `height`
        provided during initialization.

    Output Shape:
        (B, C, H, W), same shape as the input.
    """

    def __init__(self, height: int):
        super().__init__()
        if not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")
        self.height = height
        # Learnable parameters, one shift value per row.
        self.row_shifts = nn.Parameter(torch.zeros(height))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the row-wise horizontal shifts using bilinear interpolation."""
        B, _, H, W = x.shape
        device = x.device
        dtype = x.dtype
        if H != self.height:
            raise ValueError(f"Input tensor height {H} does not match module's expected height {self.height}")
        shifts = self.row_shifts.to(device)
        base_y = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype) if H > 0 else torch.zeros(
            0, device=device, dtype=dtype)
        base_x = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype) if W > 0 else torch.zeros(
            0, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing='ij')
        norm_shifts = shifts * (2.0 / W) if W > 0 else torch.zeros_like(shifts)
        shifted_norm_x = grid_x - norm_shifts.view(H, 1)
        grid = torch.stack((shifted_norm_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).expand(B, H, W, 2)
        output = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
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

# --- New function for optimizing shifts for a specific frame ---
def optimize_frame_shifts(frame_tensor, height, device, learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS):
    """Optimize row shifts for a specific frame using total variation minimization."""
    frame_shifter = RowShiftInterp(height=height).to(device)
    with torch.no_grad():
        frame_shifter.row_shifts.data = torch.randn_like(frame_shifter.row_shifts) * 0.01
    frame_optimizer = optim.Adam(frame_shifter.parameters(), lr=learning_rate)
    frame_losses = []
    for i in range(num_iterations):
        frame_optimizer.zero_grad()
        output = frame_shifter(frame_tensor)
        loss = total_variation_loss(output)
        loss.backward()
        frame_optimizer.step()
        frame_losses.append(loss.item())
    return frame_shifter, frame_losses

# --- Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RRN Secam ')
    parser.add_argument('--gpuNum', default=0, type=int, help='GPU Number')
    parser.add_argument('--inputvideo', type=str, default='ORIGINAL-MGCAA0035917--AG_JAQ_01_EXT.mkv')
    parser.add_argument('--outputvideo', type=str, default='dejittered.mp4')
    parser.add_argument('--fieldsPerFrame', type=int, default=2, help='one or two fields per frame')
    parser.add_argument('--startTime', type=str, default='00:01:30', help='start timecode')
    parser.add_argument('--nbFrames', type=int, default=100, help='number of processed frames')
    parser.add_argument('--outputQuality', type=int, default=17, help='MPEG 4 quality level')
    parser.add_argument('--outputRatio', type=str, default='4:3', help='video output ratio')
    parser.add_argument('--noAlign', action="store_true", help='discard alignement process')
    opt = parser.parse_args()

    INPUT_VIDEO_PATH = opt.inputvideo
    OUTPUT_VIDEO_PATH = opt.outputvideo
    FIELDS_PER_FRAME = opt.fieldsPerFrame
    START_TIME = opt.startTime

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
    previous_frame_tensor = None

    while frames_processed < opt.nbFrames:
        ret, frame_np = cap.read()
        if not ret:
            break

        current_frame_tensor = frame_to_tensor_cv2(frame_np, device)

        mix_and_shift_frame_tensor = current_frame_tensor.clone()
        
        if previous_frame_tensor is not None and FIELDS_PER_FRAME == 2:
            mix_and_shift_frame_tensor[:, :, 1::2, :] = previous_frame_tensor[:, :, 1::2, :]

        #define previous frame for next iteration
        previous_frame_tensor = current_frame_tensor.detach()

        if not opt.noAlign:
            optimized_shifter, frame_losses = optimize_frame_shifts(
                mix_and_shift_frame_tensor,
                height=height,
                device=device
            )

            with torch.no_grad():
                mix_and_shift_frame_tensor = optimized_shifter(mix_and_shift_frame_tensor)

        output_frame_np = tensor_to_frame(mix_and_shift_frame_tensor)
        out.write(output_frame_np)
        frames_processed += 1
        if (frames_processed % 100) == 0:
            print(f"Processed frame {frames_processed}." + (""
                  if opt.noAlign else f" Final TV loss: {frame_losses[-1]:.4f}" ))

    cap.release()
    out.release()
    print(f"\nFinished processing. {frames_processed} frames written to {OUTPUT_VIDEO_PATH}")