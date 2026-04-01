from torchinfo import summary
from .. import VideoInpainter
from .. import unet_cell


model = VideoInpainter(in_channels=10, base_channels=32, num_layers=3)
summary(model, input_size=(1, 10, 520, 520)) # Batch, Channels, H, W