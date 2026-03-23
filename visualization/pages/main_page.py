import customtkinter as ctk
from components.header_content import Header
from components.text import BodyText

class MainPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_columnconfigure(0, weight=1)

        description = ("Main Page")
        self.content = BodyText(self, text=description)
        self.content.grid(row=0, column=0, pady=20, padx=20, sticky="w")

    