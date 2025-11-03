import os
import csv
import hashlib
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import tkinter as tk
from tkinter import filedialog
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

def _parse_cli_args():
    image_path = None
    output_dir = None
    input_dir = None
    args = iter(sys.argv[1:])
    for arg in args:
        if arg.startswith("--output="):
            output_dir = arg.split("=", 1)[1]
        elif arg in {"--output", "-o"}:
            try:
                output_dir = next(args)
            except StopIteration:
                print("Warning: --output flag provided without a path. Ignoring.")
        elif arg.startswith("--input="):
            input_dir = arg.split("=", 1)[1]
        elif arg in {"--input", "-i"}:
            try:
                input_dir = next(args)
            except StopIteration:
                print("Warning: --input flag provided without a path. Ignoring.")
        elif image_path is None:
            candidate = Path(arg).expanduser()
            if candidate.exists() and candidate.is_dir() and input_dir is None:
                input_dir = str(candidate)
            else:
                image_path = arg
        elif output_dir is None:
            output_dir = arg
        elif input_dir is None:
            input_dir = arg
    return image_path, output_dir, input_dir

CLI_IMAGE_PATH, CLI_OUTPUT_DIR, CLI_INPUT_DIR = _parse_cli_args()

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
        self.crop_base_color = "#059669"
        self.crop_hover_color = "#047857"
        self.neutral_color = "#6b7280"
        self.neutral_hover = "#4b5563"
        self.danger_color = "#dc2626"
        self.danger_hover = "#b91c1c"
        self.root.configure(bg=self.bg_color)
        self.font_family = self._choose_font_family()
        self.base_font = tkfont.Font(family=self.font_family, size=12)
        self.button_font = tkfont.Font(family=self.font_family, size=12, weight="bold")
        self.group_title_font = tkfont.Font(family=self.font_family, size=11, weight="bold")
        self.root.option_add("*Font", self.base_font)
        self.palette = {
            "image": ("#4b5563", "#374151"),
            "segment": (self.accent_color, self.accent_hover),
            "crop": (self.crop_base_color, self.crop_hover_color),
            "neutral": (self.neutral_color, self.neutral_hover),
            "danger": (self.danger_color, self.danger_hover),
        }
        
        # Internal state for mask composition
        self.image = None
        self.image_array = None
        self.image_path = None
        self.image_name = None
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
        self.output_dir = Path("output")
        self.csv_path = self.output_dir / "annotations.csv"
        self.mode = "mask"
        self.segmented_image = None
        self.crop_rect_id = None
        self.crop_start_img = None
        self.crop_box_img = None
        self.crop_index = 0
        self.crop_active = False
        self.crop_selection_ready = False
        self.crop_move_active = False
        self.crop_move_offset = (0, 0)
        self.crop_side = 0
        self.display_reference_image = None
        self.input_dir = None
        self.image_files = []
        self.current_file_index = -1
        
        if CLI_OUTPUT_DIR:
            self.set_output_dir(CLI_OUTPUT_DIR)

        toolbar = tk.Frame(root, bg=self.bg_color)
        toolbar.pack(padx=12, pady=12, fill=tk.X)

        image_group = self._create_button_group(toolbar, "Image")
        self._create_button(image_group, "Load Image", self.load_image, palette="image").pack(side=tk.LEFT, padx=4, pady=2)
        self._create_button(image_group, "Next Image", self.load_next_image, palette="image").pack(side=tk.LEFT, padx=4, pady=2)

        segment_group = self._create_button_group(toolbar, "Segmentation")
        self._create_button(segment_group, "Clear Mask", self.reset_mask, palette="segment").pack(side=tk.LEFT, padx=4, pady=2)
        self._create_button(segment_group, "Commit Segment", self.commit_current_segment, palette="segment").pack(side=tk.LEFT, padx=4, pady=2)
        self._create_button(segment_group, "Save Mask", self.save_current_mask, palette="segment").pack(side=tk.LEFT, padx=4, pady=2)
        self._create_button(segment_group, "Segment Image", self.segment_image, palette="segment").pack(side=tk.LEFT, padx=4, pady=2)

        crop_group = self._create_button_group(toolbar, "Cropping")
        self._create_button(crop_group, "Start Crop", self.start_crop_mode, palette="crop").pack(side=tk.LEFT, padx=4, pady=2)
        self._create_button(crop_group, "Commit Crop", self.commit_crop_selection, palette="crop").pack(side=tk.LEFT, padx=4, pady=2)
        self._create_button(crop_group, "Cancel Crop", self.cancel_crop_selection, palette="neutral").pack(side=tk.LEFT, padx=4, pady=2)

        system_group = self._create_button_group(toolbar, "System")
        self._create_button(system_group, "Quit", root.quit, palette="danger").pack(side=tk.LEFT, padx=4, pady=2)

        self.canvas = tk.Canvas(root, width=800, height=600, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        if CLI_INPUT_DIR:
            self.set_input_dir(CLI_INPUT_DIR)
        if CLI_IMAGE_PATH:
            self._load_image_from_path(CLI_IMAGE_PATH, update_index=True)
    
    def load_image(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="Select Image")
            if not path:
                return
        self._load_image_from_path(path, update_index=True)
    
    def _load_image_from_path(self, path, *, update_index):
        try:
            img_path = Path(path).expanduser()
        except Exception as exc:
            print(f"Invalid image path '{path}': {exc}")
            return False
        try:
            resolved_path = img_path.resolve()
        except Exception:
            resolved_path = img_path
        if not resolved_path.exists():
            print(f"Image not found: {resolved_path}")
            return False
        try:
            pil_image = Image.open(resolved_path).convert("RGB")
        except (OSError, ValueError) as exc:
            print(f"Failed to load image {resolved_path}: {exc}")
            return False
        self.image_path = str(resolved_path)
        self.image_name = resolved_path.stem
        self.image = pil_image
        self.image_array = np.array(pil_image)
        
        self._clear_inference_state()

        try:
            with resolved_path.open("rb") as f:
                self.image_hash = hashlib.md5(f.read()).hexdigest()[:6]
        except OSError as exc:
            print(f"Warning: unable to hash image {resolved_path}: {exc}")
            self.image_hash = "000000"
        
        predictor.set_image(self.image_array)
        self.display_image(self.image)
        
        if update_index:
            if self.image_files:
                try:
                    self.current_file_index = self.image_files.index(resolved_path)
                except ValueError:
                    self.current_file_index = -1
            else:
                self.current_file_index = -1
        position_msg = ""
        if self.image_files and self.current_file_index is not None and self.current_file_index >= 0:
            position_msg = f" ({self.current_file_index + 1}/{len(self.image_files)})"
        print(f"Loaded image{position_msg}: {self.image_path}")
        return True
    
    def display_image(self, img):
        canvas_width = 800
        canvas_height = 600
        self.scale = min(canvas_width / img.width, canvas_height / img.height)
        display_width = int(img.width * self.scale)
        display_height = int(img.height * self.scale)
        self.display_reference_image = img
        self.display_width = display_width
        self.display_height = display_height
        self._clear_crop_overlay()

        display_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_img)
        
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def set_output_dir(self, path):
        if not path:
            return
        new_dir = Path(path).expanduser()
        if not new_dir.is_absolute():
            new_dir = (Path.cwd() / new_dir).resolve()
        else:
            new_dir = new_dir.resolve()
        try:
            new_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"Failed to prepare output directory '{new_dir}': {exc}")
            return
        self.output_dir = new_dir
        self.csv_path = self.output_dir / "annotations.csv"
        print(f"Output directory set to {self.output_dir}")
    
    def set_input_dir(self, path):
        if not path:
            return
        new_dir = Path(path).expanduser()
        try:
            new_dir = new_dir.resolve()
        except Exception:
            pass
        if not new_dir.exists() or not new_dir.is_dir():
            print(f"Input directory does not exist: {new_dir}")
            return
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        files = []
        for entry in sorted(new_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in supported_exts:
                try:
                    files.append(entry.resolve())
                except Exception:
                    files.append(entry)
        self.input_dir = new_dir
        self.image_files = files
        self.current_file_index = -1
        if not files:
            print(f"Input directory '{new_dir}' contains no supported images.")
            return
        print(f"Input directory set to {new_dir} ({len(files)} images).")
        for idx, candidate in enumerate(files):
            if not candidate.exists():
                continue
            if self._load_image_from_path(candidate, update_index=True):
                return
            print(f"Skipping unreadable image: {candidate}")
        print("Unable to load any images from the input directory.")
    
    def load_next_image(self):
        if not self.image_files:
            print("No input directory configured. Use --input to provide a folder.")
            return
        start_index = self.current_file_index if self.current_file_index is not None else -1
        next_index = start_index + 1
        total = len(self.image_files)
        while next_index < total:
            candidate = self.image_files[next_index]
            if not candidate.exists():
                print(f"Skipping missing file: {candidate}")
                next_index += 1
                continue
            if self._load_image_from_path(candidate, update_index=True):
                return
            print(f"Skipping unreadable image: {candidate}")
            next_index += 1
        print("Reached end of input directory.")
    
    def on_press(self, event):
        if self.image is None:
            return
        
        if self.mode == "crop":
            if self.crop_selection_ready and self._cropped_point_inside(event.x, event.y):
                self._begin_move_crop(event)
            else:
                self._start_crop(event)
            return
        
        if self.current_hold_mask is not None:
            self.current_hold_mask = None
            self._update_overlay()
        
        self._cancel_inference_job()
        
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        self.hold_active = True
        self.current_hold_points = []
        self.last_sampled_point = None
        self._needs_inference = False
        self._queue_point(x, y, force=True)
    
    def on_drag(self, event):
        if self.image is None:
            return
        if self.mode == "crop":
            if self.crop_move_active:
                self._move_crop(event)
            elif self.crop_active:
                self._update_crop_rect(event)
            return
        if not self.hold_active:
            return
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        self._queue_point(x, y)
    
    def on_release(self, event):
        if self.image is None:
            return
        
        if self.mode == "crop":
            if self.crop_move_active:
                self._move_crop(event)
                self._finish_crop_move()
            else:
                self._update_crop_rect(event)
                self._finish_crop_drag()
            return
        
        if not self.hold_active:
            return
        
        self.hold_active = False
        self._cancel_inference_job()
        self._run_inference()
        self.current_hold_points = []
        self.last_sampled_point = None
        self._needs_inference = False
    
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
    
    def commit_current_segment(self):
        if self.image is None:
            print("No image loaded. Load an image before committing.")
            return
        if self.current_hold_mask is None:
            print("No pending segment to commit. Draw and release on the image first.")
            return
        if self.combined_mask is None:
            self.combined_mask = self.current_hold_mask.copy()
        else:
            self.combined_mask = np.logical_or(self.combined_mask, self.current_hold_mask)
        self.current_hold_mask = None
        self.current_hold_points = []
        self.last_sampled_point = None
        self._needs_inference = False
        self._cancel_inference_job()
        self._update_overlay()
        print("Segment committed.")
    
    def segment_image(self):
        if self.image is None:
            print("No image loaded. Load an image before segmenting.")
            return
        mask = self._get_display_mask()
        if mask is None:
            print("No mask available. Draw or commit a mask before segmenting.")
            return
        mask = mask.astype(bool)
        if mask.shape[:2] != self.image_array.shape[:2]:
            print("Mask shape does not match the image. Unable to segment.")
            return
        segmented_array = self.image_array.copy()
        segmented_array[~mask] = 0
        self.segmented_image = Image.fromarray(segmented_array)
        self._exit_crop_mode(clear_segmented=False)
        self.display_image(self.segmented_image)
        print("Segmented image ready. Click 'Start Crop' to draw and commit square crops.")
    
    def start_crop_mode(self):
        if self.segmented_image is None:
            print("Segment the image before cropping.")
            return
        self._enter_crop_mode()
        self.display_image(self.segmented_image)
        print("Crop mode enabled. Drag to select a square crop; drag inside the outline to reposition; use 'Commit Crop' when ready.")
    
    def _enter_crop_mode(self):
        self.mode = "crop"
        self._cancel_inference_job()
        self.hold_active = False
        self.current_hold_points = []
        self.current_hold_mask = None
        self.last_sampled_point = None
        self._needs_inference = False
        self._clear_crop_overlay()
    
    def _exit_crop_mode(self, *, clear_segmented=True):
        if self.mode == "crop":
            self.mode = "mask"
        self._clear_crop_overlay()
        if clear_segmented:
            self.segmented_image = None
    
    def _start_crop(self, event):
        if self.segmented_image is None:
            print("Segment the image before cropping.")
            return
        self._clear_crop_overlay()
        self.crop_active = True
        self.crop_selection_ready = False
        self.crop_move_active = False
        self.crop_move_offset = (0, 0)
        self.crop_side = 0
        self.crop_start_img = self._to_image_coords(event.x, event.y)
        x_disp, y_disp = self._to_canvas_coords(*self.crop_start_img)
        self.crop_rect_id = self.canvas.create_rectangle(
            x_disp,
            y_disp,
            x_disp,
            y_disp,
            outline=self.crop_base_color,
            width=2,
        )
        self.crop_box_img = None
    
    def _cropped_point_inside(self, x, y):
        if not self.crop_selection_ready or not self.crop_box_img:
            return False
        ix, iy = self._to_image_coords(x, y)
        left, top, right, bottom = self.crop_box_img
        return left <= ix < right and top <= iy < bottom
    
    def _begin_move_crop(self, event):
        if not self.crop_selection_ready or not self.crop_box_img:
            return
        self.crop_move_active = True
        self.crop_active = False
        ix, iy = self._to_image_coords(event.x, event.y)
        left, top, right, bottom = self.crop_box_img
        offset_x = max(0, min(ix - left, right - left))
        offset_y = max(0, min(iy - top, bottom - top))
        self.crop_move_offset = (offset_x, offset_y)
        self.crop_start_img = None
        if self.crop_side == 0:
            self.crop_side = right - left
    
    def _update_crop_rect(self, event):
        if not self.crop_active or self.segmented_image is None or self.crop_start_img is None:
            return
        self.crop_selection_ready = False
        base_img = self.segmented_image
        width, height = base_img.size
        current = self._to_image_coords(event.x, event.y)
        x0, y0 = self.crop_start_img
        dx = current[0] - x0
        dy = current[1] - y0
        if dx == 0 and dy == 0:
            side = 0
            sx = 1
            sy = 1
        else:
            sx = 1 if dx >= 0 else -1
            sy = 1 if dy >= 0 else -1
            max_side = max(abs(dx), abs(dy))
            available_x = (width - 1 - x0) if sx == 1 else x0
            available_y = (height - 1 - y0) if sy == 1 else y0
            side = min(max_side, available_x, available_y)
        x1 = x0 + sx * side
        y1 = y0 + sy * side
        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1) + 1
        bottom = max(y0, y1) + 1
        right = min(right, width)
        bottom = min(bottom, height)
        if right <= left or bottom <= top:
            self.crop_box_img = None
            return
        self.crop_box_img = (left, top, right, bottom)
        self.crop_side = right - left
        self.crop_selection_ready = True
        self._redraw_crop_rect()
    
    def _finish_crop_drag(self):
        if not self.crop_active:
            return
        self.crop_active = False
        if not self.crop_selection_ready:
            print("Crop selection cleared. Drag to define a square region.")
            self._clear_crop_overlay()
            return
        print("Crop selection ready. Use 'Commit Crop' to save or drag again to adjust.")
    
    def _move_crop(self, event):
        if not self.crop_move_active or not self.crop_box_img or self.segmented_image is None:
            return
        width, height = self.segmented_image.size
        ix, iy = self._to_image_coords(event.x, event.y)
        offset_x, offset_y = self.crop_move_offset
        box_width = self.crop_side if self.crop_side > 0 else (self.crop_box_img[2] - self.crop_box_img[0])
        if box_width <= 0:
            return
        new_left = ix - offset_x
        new_top = iy - offset_y
        max_left = max(0, width - box_width)
        max_top = max(0, height - box_width)
        new_left = int(np.clip(new_left, 0, max_left))
        new_top = int(np.clip(new_top, 0, max_top))
        new_right = new_left + box_width
        new_bottom = new_top + box_width
        self.crop_box_img = (new_left, new_top, new_right, new_bottom)
        self.crop_selection_ready = True
        self._redraw_crop_rect()
    
    def _finish_crop_move(self):
        if not self.crop_move_active:
            return
        self.crop_move_active = False
        self.crop_move_offset = (0, 0)
        self.crop_start_img = None
        self.crop_side = self.crop_box_img[2] - self.crop_box_img[0] if self.crop_box_img else 0
        print("Crop position updated. Use 'Commit Crop' to save or drag again to adjust.")
    
    def _redraw_crop_rect(self):
        if not self.crop_box_img:
            return
        left, top, right, bottom = self.crop_box_img
        x_disp_start, y_disp_start = self._to_canvas_coords(left, top)
        x_disp_end, y_disp_end = self._to_canvas_coords(right, bottom)
        if x_disp_end == x_disp_start:
            x_disp_end += 1
        if y_disp_end == y_disp_start:
            y_disp_end += 1
        if self.crop_rect_id is None:
            self.crop_rect_id = self.canvas.create_rectangle(
                x_disp_start,
                y_disp_start,
                x_disp_end,
                y_disp_end,
                outline=self.crop_base_color,
                width=2,
            )
        else:
            self.canvas.coords(
                self.crop_rect_id,
                x_disp_start,
                y_disp_start,
                x_disp_end,
                y_disp_end,
            )
    
    def commit_crop_selection(self):
        if self.mode != "crop":
            print("Enable crop mode before committing a crop.")
            return
        self.crop_move_active = False
        if self.segmented_image is None:
            print("Segment the image before committing a crop.")
            return
        if not self.crop_selection_ready or not self.crop_box_img:
            print("No crop selection ready. Drag a square region before committing.")
            return
        if not self.image_name or not self.image_hash:
            print("Load an image before committing a crop.")
            return
        left, top, right, bottom = self.crop_box_img
        if right - left <= 0 or bottom - top <= 0:
            print("Crop selection is empty. Adjust the region before committing.")
            return
        crop = self.segmented_image.crop((left, top, right, bottom))
        resized = crop.resize((512, 512), Image.Resampling.LANCZOS)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_index += 1
        crop_filename = f"{self.image_name}-{self.image_hash}-crop-{self.crop_index:03d}.png"
        crop_path = self.output_dir / crop_filename
        resized.save(str(crop_path))
        print(f"Saved crop: {crop_path}")
        self._clear_crop_overlay()
    
    def cancel_crop_selection(self):
        if self.mode != "crop":
            print("Crop mode is not active.")
            return
        if not self.crop_box_img and not self.crop_active:
            print("No crop selection to cancel.")
            return
        self._clear_crop_overlay()
        print("Crop selection cleared.")
    
    def _clear_crop_overlay(self):
        if self.crop_rect_id is not None:
            try:
                self.canvas.delete(self.crop_rect_id)
            except tk.TclError:
                pass
        self.crop_rect_id = None
        self.crop_start_img = None
        self.crop_box_img = None
        self.crop_active = False
        self.crop_selection_ready = False
        self.crop_move_active = False
        self.crop_move_offset = (0, 0)
        self.crop_side = 0
    
    def _to_image_coords(self, x, y):
        reference = self.display_reference_image if self.display_reference_image is not None else self.image
        if reference is None or self.scale == 0:
            return (0, 0)
        ix = int(np.clip(x / self.scale, 0, reference.width - 1))
        iy = int(np.clip(y / self.scale, 0, reference.height - 1))
        return ix, iy
    
    def _to_canvas_coords(self, ix, iy):
        x = int(ix * self.scale)
        y = int(iy * self.scale)
        if hasattr(self, "display_width"):
            x = min(max(x, 0), self.display_width)
        if hasattr(self, "display_height"):
            y = min(max(y, 0), self.display_height)
        return x, y
    
    def save_current_mask(self):
        if self.image is None:
            print("No image loaded. Load an image before saving a mask.")
            return
        mask = self._get_display_mask()
        if mask is None:
            print("No mask to save. Draw on the image first.")
            return
        output_path = self._save_mask(mask.astype(bool))
        if output_path is None:
            print("Unable to save mask.")
            return
        self._record_annotation(output_path, "")
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
    
    def _create_button(self, parent, text, command, *, palette="segment"):
        bg_color, hover_color = self.palette.get(palette, self.palette["segment"])
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=self.button_text_color,
            activebackground=hover_color,
            activeforeground=self.button_text_color,
            relief=tk.FLAT,
            font=self.button_font,
            padx=16,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
        )
        button.bind("<Enter>", lambda _event, btn=button, color=hover_color: btn.configure(bg=color))
        button.bind("<Leave>", lambda _event, btn=button, color=bg_color: btn.configure(bg=color))
        return button
    
    def _create_button_group(self, parent, title):
        wrapper = tk.Frame(parent, bg=self.bg_color)
        wrapper.pack(side=tk.LEFT, padx=10)
        label = tk.Label(
            wrapper,
            text=title,
            bg=self.bg_color,
            fg="#1f2937",
            font=self.group_title_font,
        )
        label.pack(anchor="w")
        group = tk.Frame(wrapper, bg=self.bg_color)
        group.pack()
        return group
    
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
        self._exit_crop_mode()
        self._clear_hold_state(
            clear_combined=True,
            reset_save_index=False,
            delete_saved="none",
            show_image=True,
        )
    
    def _clear_inference_state(self):
        self._exit_crop_mode()
        self._clear_hold_state(
            clear_combined=True,
            reset_save_index=True,
            delete_saved="none",
            show_image=False,
        )
        self.saved_mask_paths = []
        self.save_index = 0
        self.crop_index = 0

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
