"""
Convert a YOLO-format defect dataset into the MVTec-style layout expected by
`dataset.MVTecDataset`.

Input layout (user-defined):
    <yolo_root>/
        img001.jpg
        img001.txt    # each line: "<class_id> <cx> <cy> <w> <h>"  (normalized)
        img002.png
        img002.txt
        ...

All images under --yolo-root are assumed to be anomalous (defect) test samples
and MUST have a non-empty same-stem .txt label. Images without a label or with
an empty label are skipped with a warning.

Normal images (train/good and test/good) are supplied via separate directories
(--train-normal-dir, --test-normal-dir) since YOLO labels describe only
anomalies.

Output layout (MVTec-style):
    <out_root>/
        train/good/                      # copies of --train-normal-dir (if given)
        test/
            good/                        # copies of --test-normal-dir  (if given)
            <class_name>/                # images containing that defect class
        ground_truth/
            <class_name>/                # binary PNG masks, same stem as test img

Notes / limitations:
    * YOLO bboxes become rectangular white regions in the mask. They are NOT
      pixel-accurate defect contours, so pixel-level metrics (pixel AUROC,
      AUPRO) will be optimistic/distorted. Image-level AUROC is reliable.
    * If one image contains multiple classes, the image is placed under the
      folder of its first class, but the mask merges bboxes from ALL classes.
      Use --per-class-duplicate to instead copy the image once per class
      (each with a class-specific mask).

Usage example:
    python -m prepare_data.yolo_to_mvtec \
        --yolo-root /data/my_yolo/test \
        --out-root  /data/my_mvtec/category_a \
        --classes   scratch dent crack \
        --train-normal-dir /data/my_yolo/train_good
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG", ".JPEG", ".BMP"}


def find_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXTS:
            yield p


def parse_label(txt_path: Path):
    """Return list of (class_id, cx, cy, w, h) in normalized coords. Empty if missing."""
    if not txt_path.is_file():
        return []
    entries = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(x) for x in parts[1:5])
        entries.append((cls, cx, cy, w, h))
    return entries


def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = int(round((cx - w / 2) * W))
    y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W))
    y2 = int(round((cy + h / 2) * H))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


def build_mask(boxes, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def resolve_class_name(cls_id: int, class_names):
    if class_names and 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    return f"class_{cls_id}"


def copy_image(src: Path, dst_dir: Path, stem: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{stem}{src.suffix.lower()}"
    shutil.copy2(src, dst)
    return dst


def save_mask(mask: np.ndarray, dst_dir: Path, stem: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{stem}.png"
    Image.fromarray(mask, mode="L").save(dst)
    return dst


def convert(
    yolo_root: Path,
    out_root: Path,
    class_names=None,
    train_normal_dir: Path = None,
    test_normal_dir: Path = None,
    per_class_duplicate: bool = False,
    mask_suffix: str = "",
):
    test_dir = out_root / "test"
    gt_dir = out_root / "ground_truth"
    train_good_dir = out_root / "train" / "good"
    test_good_dir = test_dir / "good"

    n_skipped = n_def = n_masks = 0

    for img_path in find_images(yolo_root):
        txt_path = img_path.with_suffix(".txt")
        entries = parse_label(txt_path)
        stem = img_path.stem

        if not entries:
            print(f"[warn] skip {img_path} (missing or empty .txt)")
            n_skipped += 1
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        if per_class_duplicate:
            by_class = {}
            for cls, cx, cy, w, h in entries:
                by_class.setdefault(cls, []).append(yolo_to_xyxy(cx, cy, w, h, W, H))
            for cls, boxes in by_class.items():
                cname = resolve_class_name(cls, class_names)
                out_stem = f"{stem}_{cname}"
                copy_image(img_path, test_dir / cname, out_stem)
                mask = build_mask(boxes, W, H)
                save_mask(mask, gt_dir / cname, out_stem + mask_suffix)
                n_def += 1
                n_masks += 1
        else:
            first_cls = entries[0][0]
            cname = resolve_class_name(first_cls, class_names)
            boxes = [yolo_to_xyxy(cx, cy, w, h, W, H) for _, cx, cy, w, h in entries]
            copy_image(img_path, test_dir / cname, stem)
            mask = build_mask(boxes, W, H)
            save_mask(mask, gt_dir / cname, stem + mask_suffix)
            n_def += 1
            n_masks += 1

    if train_normal_dir is not None:
        train_good_dir.mkdir(parents=True, exist_ok=True)
        n_train = 0
        for img_path in find_images(train_normal_dir):
            shutil.copy2(img_path, train_good_dir / img_path.name)
            n_train += 1
        print(f"[train/good] copied {n_train} images from {train_normal_dir}")
    else:
        print("[train/good] skipped (no --train-normal-dir given)")

    n_test_good = 0
    if test_normal_dir is not None:
        test_good_dir.mkdir(parents=True, exist_ok=True)
        for img_path in find_images(test_normal_dir):
            shutil.copy2(img_path, test_good_dir / img_path.name)
            n_test_good += 1
        print(f"[test/good]  copied {n_test_good} images from {test_normal_dir}")
    else:
        print("[test/good]  skipped (no --test-normal-dir given)")

    print(f"[test/<cls>] {n_def} defect images, {n_masks} masks")
    if n_skipped:
        print(f"[skipped]    {n_skipped} images without valid labels")
    print(f"Done. Output: {out_root}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert YOLO-format defect data to MVTec-style layout."
    )
    p.add_argument("--yolo-root", type=Path, required=True,
                   help="Directory containing images and same-stem .txt labels.")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Output MVTec-style category root.")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Class names, indexed by YOLO class id. "
                        "If omitted, folders are named class_<id>.")
    p.add_argument("--train-normal-dir", type=Path, default=None,
                   help="Optional directory of normal images to populate train/good/.")
    p.add_argument("--test-normal-dir", type=Path, default=None,
                   help="Optional directory of normal images to populate test/good/.")
    p.add_argument("--per-class-duplicate", action="store_true",
                   help="If an image has multiple classes, duplicate it under each "
                        "class folder (each with its own mask).")
    p.add_argument("--mask-suffix", default="",
                   help="Suffix appended to mask filename stem "
                        "(e.g. '_mask' to match some MVTec variants).")
    return p.parse_args()


def main():
    args = parse_args()
    convert(
        yolo_root=args.yolo_root,
        out_root=args.out_root,
        class_names=args.classes,
        train_normal_dir=args.train_normal_dir,
        test_normal_dir=args.test_normal_dir,
        per_class_duplicate=args.per_class_duplicate,
        mask_suffix=args.mask_suffix,
    )


if __name__ == "__main__":
    main()
