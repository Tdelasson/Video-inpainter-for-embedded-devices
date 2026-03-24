import customtkinter as ctk
from components.theme import Theme

class HeaderButton(ctk.CTkButton):
    def __init__(self, master, text, command):
        super().__init__(
            master,
            text=text,
            command=command,
            width=118,
            height=24,
            fg_color=Theme.WHITE,
            text_color=Theme.BLUE,
            hover_color=Theme.HOVER,
            corner_radius=5
        )

class Header(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=Theme.BLUE, corner_radius=0)
        
        #Stretching of header
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 0)
        self.grid_columnconfigure(2, weight = 1)

        # self.nav_buttons = 

        #Title label
        self.label = ctk.CTkLabel(self, text="                modelName", font=(Theme.FONT_T,48), text_color=Theme.WHITE)
        self.label.grid(row=0, column=1, padx=(70,10), pady=20)

        #Buttons
        self.btn_group = ctk.CTkFrame(self, fg_color=Theme.TP)
        self.btn_group.grid(row=0, column=2, sticky="e", padx=30)

        self.home_page_btn = HeaderButton(self.btn_group,"MainPage",lambda: controller.show_frame("MainPage"))
        self.home_page_btn.pack(side="left", expand=True, anchor="w",padx=15)

        self.guide_btn = HeaderButton(self.btn_group,"Guide",lambda: controller.show_frame("GuidePage"))
        self.guide_btn.pack(side="left", expand=True, anchor="center", padx=15)

        self.about_us_btn = HeaderButton(self.btn_group,"About Us",lambda: controller.show_frame("AboutUs"))
        self.about_us_btn.pack(side="left", expand=True, anchor="e", padx=15)