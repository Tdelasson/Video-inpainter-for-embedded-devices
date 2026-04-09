import os
import cv2
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vgg16, VGG16_Weights

from training_pipeline.dataset import YouTubeVOSDataset
from model_architecture.video_inpainter import VideoInpainter


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.vgg(x), self.vgg(y))


# --- CONFIG ---
TARGET_RES = (256, 256)
BATCH_SIZE = 4  # Nu kan du køre flere sekvenser ad gangen!
NUM_ITERATIONS = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET & DATALOADER ---
root_dir = os.getcwd()
train_path = os.path.join(root_dir, "training_data", "train")
dataset = YouTubeVOSDataset(root_dir=train_path, seq_len=5)

# DataLoaderen sørger for at loade data i baggrunden
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # Bruger 4 CPU kerner til at pre-loade data
    drop_last=True
)

# --- MODEL SETUP ---
IN_CHANNELS = 5 * 3  # seq_len * RGB
model = VideoInpainter(in_channels=IN_CHANNELS, base_channels=128, num_layers=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS, eta_min=1e-6)

perceptual_criterion = PerceptualLoss().to(device)
l1_criterion = torch.nn.L1Loss()

print(f"Starting training on {device}...")

def train():
    # --- TRAINING LOOP ---
    current_iter = 0
    while current_iter < NUM_ITERATIONS:
        for batch_data in train_loader:
            if current_iter >= NUM_ITERATIONS: break

            optimizer.zero_grad()

            # Preprocessing af hele batchen (B, Seq, H, W, C) -> (B, Seq, C, H, W)
            batch_data = batch_data.float() / 255.0
            batch_data = batch_data.permute(0, 1, 4, 2, 3).to(device)
            B, S, C, H, W = batch_data.shape

            # Split i input (de første frames) og target (den sidste frame)
            inputs = batch_data.reshape(B, -1, 256, 256)  # (B, (S-1)*C, 256, 256)
            target = batch_data[:, -1]  # (B, C, 256, 256)

            # Model forward
            output, _ = model(inputs.unsqueeze(1))
            output = output.squeeze(1)  # Fjern seq dim igen

            # Loss beregning
            l1_loss = l1_criterion(output, target)
            vgg_loss = perceptual_criterion(output, target)
            total_loss = l1_loss + (0.1 * vgg_loss)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if current_iter % 10 == 0:
                print(f"Iter {current_iter} | Total: {total_loss.item():.4f} | L1: {l1_loss.item():.4f}")

            # Gem et eksempel indimellem
            if current_iter % 500 == 0:
                out_img = (output[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(f"output_iter_{current_iter}.png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

            current_iter += 1

    print("Training Done!")

if __name__ == '__main__':
    train()