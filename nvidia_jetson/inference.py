from model_architecture.video_inpainter import VideoInpainter
import cv2
import torch
import os
import numpy as np


def run_model(video_inpainter: VideoInpainter, input_frames):
    images_tensors = []

    for current_frame in input_frames:
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # Convert to Tensor and Normalize to [0, 1]
        t = torch.from_numpy(frame_rgb).float() / 255.0

        # Change shape from (H, W, C) to (C, H, W)
        t = t.permute(2, 0, 1)

        images_tensors.append(t)

    if not images_tensors:
        print("Error: No frames found to stack!")
        return

    # Now we stack them and add Batch dimension at the start: (1, Seq, C, H, W)
    input_tensor = torch.stack(images_tensors).unsqueeze(0)

    print(f"Input Shape: {input_tensor.shape}")

    video_inpainter.eval()
    with torch.no_grad():
        output, hidden = video_inpainter(input_tensor)

    print(output.shape)
    print(hidden.shape)

    output_frame = output[0, 0].cpu()

    # Transform to NumPy Image format
    # (C, H, W) -> (H, W, C)
    output_frame = output_frame.permute(1, 2, 0).numpy()

    # Scale to 0-255 and change to BGR for OpenCV
    output_frame = (output_frame * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

    # Write the actual JPEG file
    #cv2.imwrite("output_frame.jpg", frame_bgr)
    cv2.imshow("Model Output", frame_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # Run the summary to see the GFLOPs!
    # from torchinfo import summary
    # summary(video_inpainter, input_data=input_tensor)


root_dir=r"C:\Users\tobpu\Documents\aau\Semester 6\training_data\train"
jpeg_path = os.path.join(root_dir, "JPEGImages")
specific_jpeg_path = os.path.join(jpeg_path, "00a23ccf53", "00000.jpg")

frames = []
jpeg_img = cv2.imread(specific_jpeg_path)
frames.append(jpeg_img)

model = VideoInpainter(in_channels=3, base_channels=32, num_layers=3, kernel_size=3, stride=1, padding=1)

run_model(video_inpainter = model, input_frames=frames)







