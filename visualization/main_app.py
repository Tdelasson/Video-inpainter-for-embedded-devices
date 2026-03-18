import customtkinter as ctk 

class App(ctk.CTk):
    def __init__(self):
         super().__init__()
         self.title("Hello World")
         self.geometry("400x200")
         label = ctk.CTkLabel(self, text="Hello, Jetson", font=("Calibri",24))
         label.pack(pady=20)

class MainPage(ctk.CTk):
    def __init__(self, parent):
        super().__init__(parent)
        label = ctk.CTkLabel(self, text="Main Menu", font=("Calibri",24))
        label.pack(pady=20)
         
if __name__ == "__main__":
    app = App()
    app.mainloop()