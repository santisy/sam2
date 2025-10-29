import os
import hashlib
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import tkinter as tk
from tkinter import filedialog, Button
from PIL import ImageTk
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
        
        # Internal state for mask composition
        self.image = None
        self.image_array = None
        self.image_path = None
        self.image_hash = None
        self.photo = None
        self.scale = 1.0
        self.current_hold_points = []
        self.current_hold_mask = None
        self.combined_mask = None
        self.inference_job = None
        self._needs_inference = False
        self.sample_min_dist = 12  # pixels on original image to avoid oversampling
        self.hold_active = False
        self.last_sampled_point = None
        self.inference_delay_ms = 80
        self.save_index = 0
        self.saved_mask_paths = []
        
        self.canvas = tk.Canvas(root, width=800, height=600, bg='gray')
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5, pady=5)
        Button(btn_frame, text="Clear Mask", command=self.reset_mask).pack(side=tk.LEFT, padx=5, pady=5)
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
        
        self._clear_inference_state()

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
    
    def on_press(self, event):
        if self.image is None:
            return
        
        if self.combined_mask is not None or self.current_hold_mask is not None:
            self._clear_hold_state(
                clear_combined=True,
                reset_save_index=False,
                delete_saved="none",
                show_image=True,
            )
        
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        self.hold_active = True
        self.current_hold_points = []
        self.current_hold_mask = None
        self.last_sampled_point = None
        self._needs_inference = False
        self._queue_point(x, y, force=True)
    
    def on_drag(self, event):
        if not self.hold_active or self.image is None:
            return
        
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        self._queue_point(x, y)
    
    def on_release(self, event):
        if not self.hold_active or self.image is None:
            return
        
        self.hold_active = False
        self._cancel_inference_job()
        self._run_inference()
        
        if self.current_hold_mask is not None:
            if self.combined_mask is None:
                self.combined_mask = self.current_hold_mask.copy()
            else:
                self.combined_mask = np.logical_or(self.combined_mask, self.current_hold_mask)
            self._save_mask(self.combined_mask)

        self._clear_hold_state(clear_combined=False, show_image=False)
        self._update_overlay()
    
    def _queue_point(self, x, y, force=False):
        if not self.current_hold_points or force:
            self.current_hold_points.append([x, y])
            self.last_sampled_point = (x, y)
            self._schedule_inference()
            return
        
        last_x, last_y = self.last_sampled_point
        if (x - last_x) ** 2 + (y - last_y) ** 2 >= self.sample_min_dist ** 2:
            self.current_hold_points.append([x, y])
            self.last_sampled_point = (x, y)
            self._schedule_inference()
    
    def _schedule_inference(self):
        self._needs_inference = True
        if self.inference_job is None:
            self.inference_job = self.root.after(self.inference_delay_ms, self._run_inference)
    
    def _run_inference(self):
        self._cancel_inference_job()
        if not self._needs_inference or self.image is None or not self.current_hold_points:
            self._needs_inference = False
            return
        
        self._needs_inference = False
        point_coords = np.array(self.current_hold_points, dtype=np.float32)
        point_labels = np.ones(len(point_coords), dtype=np.int32)
        
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        self.current_hold_mask = masks[0].astype(bool)
        self._update_overlay()
    
    def _get_display_mask(self):
        mask = None
        if self.combined_mask is not None:
            mask = self.combined_mask
        if self.current_hold_mask is not None:
            mask = self.current_hold_mask if mask is None else np.logical_or(mask, self.current_hold_mask)
        return mask
    
    def _update_overlay(self):
        mask = self._get_display_mask()
        if mask is None:
            if self.image is not None:
                self.display_image(self.image)
            return
        
        mask_img = Image.fromarray((mask.astype(np.uint8)) * 255)
        colored_mask = Image.new("RGB", self.image.size, (0, 255, 0))
        overlay = Image.composite(colored_mask, self.image, mask_img)
        overlay = Image.blend(self.image, overlay, 0.5)
        self.display_image(overlay)
    
    def _save_mask(self, mask):
        if mask is None:
            return
        os.makedirs("output", exist_ok=True)
        self.save_index += 1
        mask_img = Image.fromarray((mask.astype(np.uint8)) * 255)
        output_path = f"output/{self.image_name}-{self.image_hash}-mask-{self.save_index:03d}.png"
        mask_img.save(output_path)
        print(f"Saved: {output_path}")
        self.saved_mask_paths.append(output_path)
    
    def reset_mask(self):
        if self.image is None:
            return
        self._clear_hold_state(
            clear_combined=True,
            reset_save_index=True,
            delete_saved="last",
            show_image=True,
        )
    
    def _clear_inference_state(self):
        self._clear_hold_state(
            clear_combined=True,
            reset_save_index=True,
            delete_saved="none",
            show_image=False,
        )
        self.saved_mask_paths = []
        self.save_index = 0

    def _cancel_inference_job(self):
        if self.inference_job is not None:
            self.root.after_cancel(self.inference_job)
            self.inference_job = None
    
    def _clear_hold_state(
        self,
        *,
        clear_combined,
        reset_save_index=False,
        delete_saved="none",
        show_image=True,
    ):
        if delete_saved not in {"none", "last", "all"}:
            raise ValueError(f"Unknown delete_saved mode: {delete_saved}")
        self._cancel_inference_job()
        self.current_hold_points = []
        self.current_hold_mask = None
        self.last_sampled_point = None
        self._needs_inference = False
        self.hold_active = False
        if clear_combined:
            self.combined_mask = None
        if delete_saved != "none":
            self._delete_saved_masks(mode=delete_saved)
        if reset_save_index or delete_saved != "none":
            self.save_index = len(self.saved_mask_paths)
        if show_image and self.image is not None:
            if clear_combined:
                self.display_image(self.image)
            else:
                self._update_overlay()

    def _delete_saved_masks(self, *, mode):
        if not self.saved_mask_paths:
            return
        if mode == "last":
            paths = [self.saved_mask_paths.pop()]
        elif mode == "all":
            paths = []
            while self.saved_mask_paths:
                paths.append(self.saved_mask_paths.pop())
        else:
            return
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                print(f"Warning: failed to delete {path}: {exc}")

root = tk.Tk()
app = SAM2GUI(root)
root.mainloop()
