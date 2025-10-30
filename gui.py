import os
import csv
import hashlib
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.font as tkfont
from PIL import ImageTk
import sys
from pathlib import Path

try:
    import ttkbootstrap as tb
except ImportError:
    tb = None

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
        self.bg_color = "#f4f6fb"
        self.canvas_bg = "#1f2937"
        self.accent_color = "#2563eb"
        self.accent_hover = "#1d4ed8"
        self.button_text_color = "#ffffff"
        self.help_bg = "#2563eb"
        self.root.configure(bg=self.bg_color)
        self.font_family = self._choose_font_family()
        self.base_font = tkfont.Font(family=self.font_family, size=12)
        self.button_font = tkfont.Font(family=self.font_family, size=12, weight="bold")
        self.entry_font = tkfont.Font(family=self.font_family, size=12)
        self.root.option_add("*Font", self.base_font)
        
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
        self.prompt_var = tk.StringVar()
        self.output_dir = Path("output")
        self.csv_path = self.output_dir / "annotations.csv"
        
        self.canvas = tk.Canvas(root, width=800, height=600, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        btn_frame = tk.Frame(root, bg=self.bg_color)
        btn_frame.pack(pady=10)
        self._create_button(btn_frame, "Load Image", self.load_image).pack(side=tk.LEFT, padx=8)
        self._create_button(btn_frame, "Clear Mask", self.reset_mask).pack(side=tk.LEFT, padx=8)
        self._create_button(btn_frame, "Quit", root.quit).pack(side=tk.LEFT, padx=8)
        
        prompt_frame = tk.Frame(root, bg=self.bg_color)
        prompt_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        tk.Label(prompt_frame, text="Prompt:", font=self.base_font, bg=self.bg_color).pack(side=tk.LEFT, padx=(2, 4))
        self.prompt_entry = tk.Entry(prompt_frame, textvariable=self.prompt_var, width=60, font=self.entry_font, relief=tk.FLAT, bg="white")
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5, ipady=6, ipadx=6)
        self._create_help_icon(prompt_frame)
        self._create_button(prompt_frame, "Save Mask", self.save_current_mask).pack(side=tk.LEFT, padx=8)
        
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
    
    def save_current_mask(self):
        if self.image is None:
            print("No image loaded. Load an image before saving a mask.")
            return
        if self.combined_mask is None:
            print("No mask to save. Draw on the image first.")
            return
        output_path = self._save_mask(self.combined_mask)
        if output_path is None:
            print("Unable to save mask.")
            return
        prompt_text = self.prompt_var.get().strip()
        self._record_annotation(output_path, prompt_text)
        print(f"Saved: {output_path}")
    
    def _save_mask(self, mask):
        if mask is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_index += 1
        mask_img = Image.fromarray((mask.astype(np.uint8)) * 255)
        mask_filename = f"{self.image_name}-{self.image_hash}-mask-{self.save_index:03d}.png"
        output_path = self.output_dir / mask_filename
        mask_img.save(str(output_path))
        self.saved_mask_paths.append(str(output_path))
        return output_path
    
    def _record_annotation(self, mask_path, prompt_text):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not csv_exists:
                writer.writerow(["image_filename", "mask_filename", "prompt"])
            image_name = os.path.basename(self.image_path) if self.image_path else ""
            mask_name = os.path.basename(mask_path)
            writer.writerow([image_name, mask_name, prompt_text])
    
    def _create_button(self, parent, text, command):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.accent_color,
            fg=self.button_text_color,
            activebackground=self.accent_hover,
            activeforeground=self.button_text_color,
            relief=tk.FLAT,
            font=self.button_font,
            padx=16,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
        )
        button.bind("<Enter>", lambda _event, btn=button: btn.configure(bg=self.accent_hover))
        button.bind("<Leave>", lambda _event, btn=button: btn.configure(bg=self.accent_color))
        return button
    
    def _create_help_icon(self, parent):
        canvas = tk.Canvas(parent, width=30, height=30, bg=self.bg_color, highlightthickness=0, bd=0)
        canvas.pack(side=tk.LEFT, padx=(0, 6))
        radius = 12
        center = 15
        canvas.create_oval(
            center - radius,
            center - radius,
            center + radius,
            center + radius,
            fill=self.help_bg,
            outline=self.help_bg,
        )
        canvas.create_text(
            center,
            center,
            text="?",
            fill=self.button_text_color,
            font=self.button_font,
        )
        canvas.bind("<Button-1>", lambda _event: self._show_prompt_help())
        canvas.bind("<Enter>", lambda _event: canvas.configure(cursor="hand2"))
        canvas.bind("<Leave>", lambda _event: canvas.configure(cursor=""))
        self.prompt_help = canvas
    
    def _show_prompt_help(self):
        message = (
            "The prompt is a descriptive sentence that identifies the location and intent of the mask. "
            "Use it to explain what the segmented region represents without relying on the mask itself."
        )
        messagebox.showinfo("Prompt guidance", message)
    
    def _choose_font_family(self):
        preferred = [
            "Inter",
            "Roboto",
            "Segoe UI",
            "SF Pro Display",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
        ]
        available = {name.lower(): name for name in tkfont.families()}
        for candidate in preferred:
            key = candidate.lower()
            if key in available:
                return available[key]
        try:
            return tkfont.nametofont("TkDefaultFont").actual("family")
        except Exception:
            return "Arial"
    
    def reset_mask(self):
        if self.image is None:
            return
        self._clear_hold_state(
            clear_combined=True,
            reset_save_index=False,
            delete_saved="none",
            show_image=True,
        )
        self.prompt_var.set("")
    
    def _clear_inference_state(self):
        self._clear_hold_state(
            clear_combined=True,
            reset_save_index=True,
            delete_saved="none",
            show_image=False,
        )
        self.saved_mask_paths = []
        self.save_index = 0
        self.prompt_var.set("")

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

def _create_root():
    if tb is not None:
        try:
            return tb.Window(themename="flatly")
        except Exception:
            pass
    return tk.Tk()


root = _create_root()
app = SAM2GUI(root)
root.mainloop()
