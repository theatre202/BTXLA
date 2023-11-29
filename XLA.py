import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý Ảnh")

        # Đặt kích thước cửa sổ và vị trí trung tâm
        window_width = 800
        window_height = 550
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        self.image_path = None
        self.kernel_size = 5
        self.gamma = 1.5  # Giá trị gamma mặc định
        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("Lọc trung vị")  # Giá trị mặc định

        self.create_widgets()

    def apply_noise_reduction(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (self.kernel_size, self.kernel_size), 0)

        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Xử lý (Giảm nhiễu)', blurred_image)
        cv2.resizeWindow('Ảnh Gốc', 800, 600)  # Thay đổi kích thước cửa sổ hiển thị
        cv2.resizeWindow('Ảnh Xử lý (Giảm nhiễu)', 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_contrast(self):
        # Đọc ảnh
        image = cv2.imread(self.image_path)

        # Chuyển đổi sang ảnh xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Áp dụng phương pháp CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # Hiển thị ảnh gốc và ảnh được tăng cường độ tương phản
        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Xử lý (Tăng cường độ tương phản)', enhanced_image)
        cv2.resizeWindow('Ảnh Gốc', 800, 600)
        cv2.resizeWindow('Ảnh Xử lý (Tăng cường độ tương phản)', 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_stretch(self):
        image = cv2.imread(self.image_path)

        # Chuyển đổi sang ảnh xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Áp dụng phương pháp power-law transformation (gamma correction)
        stretched_image = gray_image / float(np.max(gray_image)) * self.gamma * 255.0
        stretched_image = np.uint8(stretched_image)

        # Hiển thị ảnh gốc và ảnh được dãn
        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Xử lý (Tăng cường độ tương phản)', stretched_image)
        cv2.resizeWindow('Ảnh Gốc', 800, 600)
        cv2.resizeWindow('Ảnh Xử lý (Tăng cường độ tương phản)', 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_max_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        max_filtered_image = cv2.dilate(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))

        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Lọc Max', max_filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_min_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_filtered_image = cv2.erode(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))

        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Lọc Min', min_filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_midpoint_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        max_filtered_image = cv2.dilate(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))
        min_filtered_image = cv2.erode(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))
        midpoint_filtered_image = (max_filtered_image + min_filtered_image) // 2

        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Lọc Trung điểm', midpoint_filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_median_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.medianBlur(gray_image, self.kernel_size)

        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Lọc Trung vị', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_Mean_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.blur(gray_image, (self.kernel_size, self.kernel_size))

        cv2.imshow('Ảnh Gốc', gray_image)
        cv2.imshow('Ảnh Lọc Trung bình', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_widgets(self):
        # Nút để chọn ảnh
        select_image_button = tk.Button(self.root, text="Chọn Ảnh", command=self.select_image)
        select_image_button.pack(pady=10)

        # Dropdown để chọn thuật toán
        algorithms = ["Lọc trung vị", "Giảm nhiễu", "Tăng cường độ tương phản", "Dãn ảnh", "Lọc Max và Min", "Lọc Trung điểm"]
        algorithm_menu = tk.OptionMenu(self.root, self.algorithm_var, *algorithms)
        algorithm_menu.pack(pady=10)

        # Nút để thực hiện xử lý ảnh
        process_button = tk.Button(self.root, text="Xử lý Ảnh", command=self.process_image)
        process_button.pack(pady=10)

    def select_image(self):
        # Hiển thị hộp thoại để chọn ảnh
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path

    def process_image(self):
        if self.image_path:
            algorithm = self.algorithm_var.get()

            if algorithm == "Lọc trung vị":
                self.apply_median_filter()
            elif algorithm == "Giảm nhiễu":
                self.apply_noise_reduction()
            elif algorithm == "Tăng cường độ tương phản":
                self.apply_contrast()
            elif algorithm == "Dãn ảnh":
                self.apply_stretch()
            elif algorithm == "Lọc Max và Min":
                self.apply_max_filter()
            elif algorithm == "Lọc Min":
                self.apply_min_filter()
            elif algorithm == "Lọc Trung điểm":
                self.apply_midpoint_filter()
            elif algorithm == "Lọc Trung bình":
                self.apply_Mean_filter()

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
