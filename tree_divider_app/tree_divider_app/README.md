# Tree Divider (Python / Kivy)

Point the camera at a tree, tap its **base** and **top**, and the trunk is split into
equal numbered parts (100 by default). Tap anywhere along the trunk to get an exact
reading such as **11.64 / 100**. **Save** writes the picture with the ruler drawn on it.

## Run it on your computer (Windows, Mac or Linux)

**Easiest (Windows):** install Python from python.org (tick "Add python.exe to PATH"),
double-click `Setup (Windows).bat` once, then `Tree Divider (Windows).bat` to start.
**Mac/Linux:** run `./setup_mac_linux.sh` once, then `./run_mac_linux.sh`.

Or by hand:

```bash
python -m venv venv
# Windows:  venv\Scripts\activate      Mac/Linux:  source venv/bin/activate
pip install -r requirements.txt
python main.py
```

No webcam? Tap **Photo** and open a picture of a tree.

## How to use

**Opening a photo:** tap **Photo**. You get your system's normal "Open" window
(Windows / Mac / Linux) or the phone's Gallery / Files picker (Android), starting in
your Pictures folder. Phone photos are automatically turned the right way up.

| Action | Phone | Computer |
|---|---|---|
| Set the **Base**, then the **Top** | tap | click |
| Add a blue **reading** (e.g. `11.64`) | tap near the trunk | click near the trunk |
| Move a dot | drag it | drag it |
| Remove a reading | double-tap it | double-click it |
| **Zoom in / out** | pinch with two fingers | mouse wheel (zooms around the pointer) |
| **Move around** when zoomed | drag with one finger | drag, or right-drag |
| Zoom buttons | **+**, **-**, **Fit** on the right | same |

Zoom in to place the base, top and readings precisely. As you zoom, more numbers
appear (every 5, then every 2, then every single part), and the ruler stays locked
to the tree.

Bottom bar: **Freeze/Live** pauses the camera (**Camera** returns to it from a
photo), **Photo**, **- 100 +** number of parts, **Save**, **Reset**.

## Tree height in metres

Stand a **banner** (any upright reference object of known height, 1 m by default) at
the base of the tree, in the same photo.

1. Tap the **base** and **top** of the tree.
2. Tap the **top of the banner**. This first blue dot is the banner; it is labelled
   `Banner 1 m`.
3. The tree height appears in the top-left corner:

```
tree height = banner height x parts / banner reading
            = 1 m x 100 / 11.53 = 8.67 m
```

(Not reading / 100: 11.53 / 100 = 0.1153 is the banner's *share* of the tree.)

Any further blue dot shows its own height above the base, e.g. `45.19` ->
`3.92 m above base` (= 45.19 / 11.53 x 1 m), useful for the first branch.

**Different banner?** Tap the tree-height box and enter its height (e.g. 1.5).
The app remembers it.

**Accuracy:** the result assumes the banner and the tree are the same distance from
the camera and the phone is held upright (see Accuracy below). Tilting the phone up
makes the upper parts look shorter, so the tree height comes out too low.

## readings.txt: every reading in one text file

Each **Save** adds one line to `readings.txt` in the same folder as the pictures
(Android: `Documents/TreeDivider`). The save window shows its full path.

```
Tree Divider readings (tab-separated: opens in Excel or any text editor)
Date        Time      Picture                                   Parts  Banner reading  Banner height (m)  Tree height (m)  Other readings
2026-10-10  13:33:52  tree-divider-height-8_67m-reading-11_53-...jpg  100  11.53  1.00  8.67  45.19 = 3.92 m
```

The saved picture also shows the heights, with the tree height in a strip below it.

## Machine vision check (automatic)

Each time you add, move or remove a blue dot (or change the number of parts), the app
draws the ruler onto the picture exactly as **Save** would, then machine vision reads it
back. The result appears under each blue banner:

| Label | Meaning |
|---|---|
| `Vision: reading...` | working (takes a moment; the app stays usable) |
| `Vision 11.53` + green tick | banner text read by OCR **and** confirmed by measuring the dot |
| `Vision 11.54 (measured)` | Tesseract isn't installed, so only the measurement is shown |
| `Vision: banner 8.17, measured 5.17 !` (orange) | OCR and measurement disagree: check it |
| `Vision: dot not found` | the dot is hidden by something of the same colour |

**Opening a picture that already has a ruler** (for example one you saved earlier):
the app spots it and offers **Load ruler**, bringing back the base, top and readings so
you can check or adjust them. It even works out how many parts the picture was saved
with (banner value / dot position), and the loaded reading matches the banner exactly.

Needs `pip install -r requirements-vision.txt` plus the Tesseract program (see below).
Without OpenCV the check is simply switched off; without Tesseract it measures only.
On Android, add `opencv` and `numpy` to `requirements` in `buildozer.spec` to enable
the measurement (Tesseract OCR isn't available there).

## Where Save puts pictures

After saving, a window shows the **full path** and an **Open folder** button.
If saving isn't possible (no base/top yet, or an error), the window says why.

| Device | Folder |
|---|---|
| Windows | your real Pictures folder, as Windows reports it (OneDrive included), then `TreeDivider` |
| Mac | `~/Pictures/TreeDivider` |
| Linux | your desktop's Pictures folder (`xdg-user-dir PICTURES`), then `TreeDivider` |
| Android | the shared `Pictures/TreeDivider` folder, so pictures appear in the Gallery / Photos app |

If that folder can't be written, the picture goes to the app's own data folder and
the window says so. Files are named like `tree-divider-reading-11_64-20261010-114047.jpg`.

## How the app is built

```
main.py                      entry point
tree_divider/
  ruler.py                   LOGIC  - geometry: divisions, ticks, labels, readings
  viewport.py                LOGIC  - zoom & pan: picture pixels <-> screen positions
  session.py                 LOGIC  - app state: what each tap/drag does
  exporter.py                LOGIC  - Pillow: draws the ruler onto the full-size picture (JPEG)
  height.py                  LOGIC  - tree height from the banner; readings.txt format
  storage.py                 LOGIC  - where pictures are saved on each system; Open folder
  vision.py                  VISION - finds ruler, dots and banners; OCR; measurement
  vision_check.py            VISION - app checks: verify readings, find rulers in opened pictures
  auto_measure.py            ML     - zero-shot: Grounding DINO + SAM 2.1 find tree & banner
  ui/app.py                  FRONT  - Kivy screen: banner, zoom buttons, bottom bar, camera, photo loading
  ui/photo_view.py           FRONT  - draws picture + ruler; taps, drags, pinch, wheel, pan
  ui/picker.py               FRONT  - system file picker, with a built-in browser as backup
tools/extract_reading.py     VISION - command line: pictures or folders -> readings / CSV
tools/auto_measure.py        ML     - command line: automatic tree height from photos
tests/                       69 tests for the logic (no screen needed)

The base, top and readings are stored in **picture pixels**, not screen positions,
so zooming and panning never move them off the tree, and saving needs no conversion.
```

The logic files have no Kivy code, so the maths can be tested on its own and reused
with a different front end (for example Flask or PyQt).

**The reading formula** (`Ruler.reading`): the tapped point is projected onto the line
from base to top, so a tap slightly beside the trunk still reads correctly:

```
fraction = ((tap - base) · (top - base)) / |top - base|²     clamped to 0..1
reading  = fraction × parts
```

Run the tests with `pytest -q`.

**Picker order:** Windows/Linux use the system dialog via `tkinter`, then `plyer`; Mac uses
`plyer`, then `tkinter`; Android uses `plyer`. If none is available, the built-in
`PhotoBrowser` (folder shortcuts, preview, Open button) is used.

## Reading the banner value back out of a picture (machine vision)

`tools/extract_reading.py` finds the blue reading (e.g. **11.53**) in a saved picture.

```bash
pip install -r requirements-vision.txt
python tools/extract_reading.py tree.jpg                       # one picture
python tools/extract_reading.py tree.jpg --debug check.png     # + picture of what it found
python tools/extract_reading.py photos/ --csv readings.csv     # a whole folder -> spreadsheet
python tools/extract_reading.py photos/ --txt readings.txt     # + tree heights, app's text format
python tools/extract_reading.py photos/ --txt readings.txt --banner-m 1.5
```

For tree heights, the **lowest** blue dot in each picture is taken as the banner's top.

Re-saving a picture that already had a ruler burned in leaves two overlapping rulers;
the banner text is still read and confirmed, but the measured value can drift by about
1 part. Start from the original photo when you can.

If Tesseract is installed but not found, set `TESSERACT_CMD` to its full path
(e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`); the usual install folders
are checked automatically.

Install the Tesseract OCR program too: Windows - the installer from the
UB-Mannheim Tesseract page (tick "add to PATH"); Mac - `brew install tesseract`;
Linux - `sudo apt install tesseract-ocr`. Without it, the measurement still works.

**How it works**

1. **Colour segmentation (OpenCV, HSV):** yellow + pink pixels form the ruler;
   saturated blue pixels are the reading dot, connector and banner.
2. **Ruler line:** morphological closing joins the band across the tick marks; the
   longest thin component is the ruler; `cv2.fitLine` gives its direction.
3. **Base & Top:** the fattest spot at each end (distance transform) is the dot
   centre; the yellow end is the Base, the pink end the Top.
4. **Reading dot:** blue pixels lying on the ruler line; pieces split by ticks are merged.
5. **Measurement:** `reading = (dot - base)·(top - base) / |top - base|² × parts`.
6. **Banner OCR:** a morphological opening cuts the connector off the banner; the
   red channel separates white digits from blue; scaled up and read by Tesseract in
   single-line mode. The app always prints two decimals, so the digits alone fix the
   value even if the decimal point is lost.
7. **Final value:** the banner text if the measurement agrees (within 1% of the tree
   height); otherwise the measurement, marked **CHECK**.

**Tested** on 100 unseen pictures made by the app (random readings, tilts, 10-200
parts, picture sizes, JPEG quality 70-92): every reading found; measurement error
typically 0.05% of the tree height; banner read exactly 94% of the time, with
misreads caught by the measurement. On the real photo `...reading-11_53...jpg` it
returns **11.53** (measured 11.54).

`--parts` must match the app's setting when the picture was saved (default 100).

## Automatic measurement from a photo (zero-shot prototype)

`tools/auto_measure.py` measures a tree with **no tapping and no training**: give it a
photo of the whole tree with the banner standing at its base.

**In the app:** open a photo, click **Auto**, enter the banner's height and what it looks
like, and click **Measure**. The app places the base, top and banner dots for you;
check them (zoom in) and drag any that are off. Save works as usual.

**Many photos at once (Windows):** drag photos onto `Auto Measure - drop photos here (Windows).bat`.
Results (marked pictures + `auto-readings.txt`) go to `Pictures\TreeDivider\Auto`.
Edit the two settings at the top of that file (banner height, banner description).
Mac/Linux: `./auto_measure_mac_linux.sh photo.jpg`.

By hand:

```bash
pip install -r requirements-auto.txt
python tools/auto_measure.py photo.jpg --banner-words "white board"
python tools/auto_measure.py photos/ --out results/ --txt auto-readings.txt --banner-m 1
```

**How it works** (`tree_divider/auto_measure.py`):

1. **Grounding DINO** (open-vocabulary detector) finds the boxes of "tree" and the
   banner from those words alone.
2. **SAM 2.1** (Segment Anything) outlines each precisely inside its box.
3. Tree top = highest tree pixel; tree base = lowest point of the trunk; banner = top
   and bottom of its outline.
4. `tree height = banner height x tree pixels / banner pixels`

Both run with `onnxruntime` on an ordinary CPU (no PyTorch, no graphics card); they
download automatically (~335 MB) into `~/.tree_divider/models` on first use. Each photo
takes several seconds. Models: Grounding DINO (IDEA Research) and SAM 2.1 (Meta), both
Apache-2.0, in the ONNX versions published on the X-AnyLabeling GitHub releases.

**Describe your banner well**: `--banner-words` is what the detector looks for. On the
test photo "white board" scored 0.74 against 0.55 for "banner". Use what it looks like:
"orange board", "white sign", "poster"...

**It warns you when the photo can't give a valid answer**: banner not standing at the
tree's base, banner held close to the camera (looks far too big), tree cut off by the
photo's edge, or low detection confidence. Use `--out` to see what was found.

**Status (prototype):** the pipeline works end to end and the banner is measured to the
pixel. It has only been tried on one real photo (with the app's ruler digitally
removed), so its real-world accuracy is not known yet. Next step: run it on 20+ real
photos with the banner at the base and compare with the app's manual readings.

## Build an Android app (APK)

Buildozer runs on Linux (on Windows, use WSL with Ubuntu):

```bash
pip install buildozer cython
sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool pkg-config \
     zlib1g-dev libncurses-dev cmake libffi-dev libssl-dev
buildozer android debug          # first build takes a long time
```

The APK appears in `bin/`. Copy it to your phone and install it (allow "install unknown apps").

Notes for Android:
* Kivy's built-in camera works on many phones but can show the preview rotated on some.
  If that happens, the `camera4kivy` package is the more reliable camera for Android.
* **Photo** opens Android's own picker. If a phone's picker returns something the app
  can't read, it falls back to a built-in browser with shortcuts to Camera, Pictures
  and Downloads.
* iPhone builds are possible with `kivy-ios`, but need a Mac and Xcode.

## Accuracy

The parts are equal **on the screen**. If you tilt the phone up to fit a tall tree, the
upper parts look shorter than they really are. Stand further back and hold the phone
upright for readings closest to the real tree.
