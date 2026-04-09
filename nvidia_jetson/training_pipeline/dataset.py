import os
import cv2
from torch.utils.data import Dataset
import json
import numpy as np


class YouTubeVOSDataset(Dataset):
    def __init__(self, root_dir, seq_len: int = 5):
        self.jpeg_path = os.path.join(root_dir, "JPEGImages")
        self.seq_len = seq_len

        with open(os.path.join(root_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)['videos']

        self.video_list = list(self.meta.keys())

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        specific_jpeg_path = os.path.join(self.jpeg_path, video_id)
        all_frames = sorted([f for f in os.listdir(specific_jpeg_path) if f.endswith('.jpg')])

        if len(all_frames) < self.seq_len:
            return self.__getitem__(np.random.randint(0, len(self.video_list)))

        selected_frames = all_frames[:self.seq_len]

        rgb_frames = []
        for frame_name in selected_frames:
            frame_path = os.path.join(specific_jpeg_path, frame_name)
            jpeg_img = cv2.imread(frame_path)
            if jpeg_img is not None:
                jpeg_img = cv2.resize(jpeg_img, (256, 256))
                rgb_img = cv2.cvtColor(jpeg_img, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb_img)

        if len(rgb_frames) < self.seq_len:
            return self.__getitem__(np.random.randint(0, len(self.video_list)))

        return np.array(rgb_frames)  # Nu er alle (5, 256, 256, 3)