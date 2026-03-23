import customtkinter as ctk

class BodyText(ctk.CTkLabel):
    def __init__(self, master, text,font_type="Calibri", fontsize = 32, wraplength=500):
        super().__init__(
            master, text=text, font=(font_type, fontsize),
            justify="left",
            wraplength=wraplength,
            anchor="w"
        )
