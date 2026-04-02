from dataset import YouTubeVOSDataset
import cv2

# We have 3471 files
number_of_seq = 50


dataset = YouTubeVOSDataset(root_dir=r"C:\Users\tobpu\Documents\aau\Semester 6\training_data\train", seq_len=5)

rgb_data = []

for i in range(0, number_of_seq):
    rgb_data.append(dataset.load_data())

for i in range(0, len(rgb_data)):
    for h in range(0, len(rgb_data[i])):
        cv2.imshow('image', rgb_data[i][h])
        cv2.waitKey(0)
