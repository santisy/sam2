import os
import hashlib
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import tkinter as tk
from tkinter import filedialog, Button
from PIL import ImageTk, ImageDraw
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
os.environ["SAM2_REPO_ROOT"] = str(repo_root)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))  # put it first, before other paths

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

class SAM2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Interactive Segmentation")
        
        self.image = None
        self.image_array = None
        self.image_path = None
        self.image_hash = None
        self.photo = None
        self.scale = 1.0
        
        self.canvas = tk.Canvas(root, width=800, height=600, bg='gray')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5, pady=5)
        Button(btn_frame, text="Quit", command=root.quit).pack(side=tk.LEFT, padx=5, pady=5)
        
        if len(sys.argv) > 1:
            self.load_image(sys.argv[1])
    
    def load_image(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="Select Image")
            if not path:
                return
        
        self.image_path = path
        self.image_name = os.path.basename(path).split(".")[0]
        self.image = Image.open(path).convert("RGB")
        self.image_array = np.array(self.image)
        
        with open(path, "rb") as f:
            self.image_hash = hashlib.md5(f.read()).hexdigest()[:6]
        
        predictor.set_image(self.image_array)
        self.display_image(self.image)
    
    def display_image(self, img):
        canvas_width = 800
        canvas_height = 600
        self.scale = min(canvas_width / img.width, canvas_height / img.height)
        display_width = int(img.width * self.scale)
        display_height = int(img.height * self.scale)
        
        display_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def on_click(self, event):
        if self.image is None:
            return
        
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        
        masks, _, _ = predictor.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]))
        mask = masks[0]
        
        overlay = self.image.copy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        colored_mask = Image.new("RGB", self.image.size, (0, 255, 0))
        overlay = Image.composite(colored_mask, overlay, mask_img)
        overlay = Image.blend(self.image, overlay, 0.5)
        
        os.makedirs("output", exist_ok=True)
        output_path = f"output/{self.image_name}-{self.image_hash}-{x}x{y}-mask.png"
        mask_img.save(output_path)
        print(f"Saved: {output_path}")
        
        self.display_image(overlay)

root = tk.Tk()
app = SAM2GUI(root)
root.mainloop()