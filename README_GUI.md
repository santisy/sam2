# SAM2 GUI Tool

This desktop tool lets you interactively segment images with the SAM2 model and capture annotations for later use.

## Requirements

- Python 3.8+
- Project dependencies installed (see main project README for details)
- A SAM2 checkpoint available at `./checkpoints/sam2.1_hiera_large.pt` (https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

## Styling

The GUI ships with a brighter layout, larger controls, and an in-app prompt helper. For an extra-polished look, install [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap); the interface will automatically adopt its *flatly* theme when available.

```bash
pip install ttkbootstrap
```

## Launching the GUI

```bash
python3 gui.py
```

You can optionally pass an image path to open it immediately:

```bash
python3 gui.py path/to/image.jpg
```

## Workflow

1. **Load Image** – Click *Load Image* or pass a path on start-up.
2. **Segment** – Left-click and drag to sketch positive strokes; the mask updates as you release the mouse. **If a single click oversegments the target, hold and drag across the desired area to add more prompt locations and tighten the mask.**
3. **Refine** – Use additional strokes and releases until the overlay looks right. *Clear Mask* removes the in-memory mask so you can restart from scratch without touching saved files.
4. **Describe & Save** – Enter a descriptive prompt (use the ? tooltip for guidance) and click *Save Mask*. The GUI keeps the current mask so you can continue refining or save additional versions if needed.
5. **Quit** – Close the window or click *Quit* when finished.

### What the prompt should capture

The prompt is a descriptive sentence that identifies the location and intent of the mask. Use it to explain what the segmented region represents without relying on the mask itself.

## Outputs

All outputs are saved inside the `output/` directory adjacent to `gui.py`:

- Binary mask images named `{image-name}-{image-hash}-mask-###.png`
- `annotations.csv` containing the columns:
  - `image_filename` – the loaded image file name
  - `mask_filename` – the mask file name that was just written
  - `prompt` – the text you entered before saving

Each time you click *Save Mask*, a new row is appended to `annotations.csv` and the mask image is written. Clearing the mask does not delete saved files; remove them manually if you need to discard an export.
