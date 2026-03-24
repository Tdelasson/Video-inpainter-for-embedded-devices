
import customtkinter as ctk
import cv2
import os

from components.header_content import Header
from components.text import BodyText
from components.theme import Theme
from PIL import Image

class MainPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        #Placeholder using downloaded video instead of live video footage from camera
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_path, "assets", "video_placeholder.mp4")

        self.video = cv2.VideoCapture(video_path)

        self.grid_columnconfigure((0, 1), weight=1)


        self.left_frame = ctk.CTkFrame(self, fg_color=Theme.TP)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20)

        self.title_left = BodyText(self.left_frame, text="Input")
        self.title_left.grid(row=0, column=0)

        self.btn = ctk.CTkButton(self.left_frame, text="Start", width=80)
        self.btn.grid(row=0, column=1, padx=(50,0))

        #Right Column
        self.title_right = BodyText(self, text="Input")
        self.title_right.grid(row=0, column=1, pady=10)


        #Display
        self.display_left = ctk.CTkLabel(self, text="")
        self.display_left.grid(row=2, column=0, padx=10)

        self.display_right = ctk.CTkLabel(self, text="")
        self.display_right.grid(row=2, column=1, padx=10)

        #Footer
        self.desc_left = BodyText(self, text="Input")
        self.desc_left.grid(row=3, column=0, pady=10)

        self.desc_right = BodyText(self, text="Input")
        self.desc_right.grid(row=3, column=1, pady=10)

        self.update_frame()
    
    def update_frame(self):
        ret, frame = self.cap.read() #Tries to read frame

        if ret:
            frame = cv2.resize(frame, (500, 300))
            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_img)
            ctk_img = ctk.CTkImage(light_image=pil_img, size=(500, 300))

            self.display_left.configure(image=ctk_img)
            self.display_right.configure(image=ctk_img)
            self.display_left.image = ctk_img
            self.display_right.image =ctk_img

        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0) #Replay video when ending

        self.after(30, self.update_frame)
            