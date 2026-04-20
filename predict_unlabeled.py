"""
对无标签图像数据集进行异常检测预测。

数据集只需是一个包含图像的目录（可嵌套子目录），不需要 good/defect 结构，也不需要 ground_truth。
脚本会输出每张图的异常分数、预测标签，并可选地保存热力图可视化。

用法：
    # 基本用法：指定 checkpoint、图像目录
    python predict_unlabeled.py \
        --checkpoint saved_results/xxx/model.pth \
        --data_path  /path/to/unlabeled_images \
        --encoder_name dinov2reg_vit_base_14

    # 手动指定阈值
    python predict_unlabeled.py \
        --checkpoint saved_results/xxx/model.pth \
        --data_path  /path/to/unlabeled_images \
        --threshold 0.35

    # 保存热力图（top-k 最异常的图像）
    python predict_unlabeled.py \
        --checkpoint saved_results/xxx/model.pth \
        --data_path  /path/to/unlabeled_images \
        --save_heatmap \
        --heatmap_topk 20

    # 保存全部图像的热力图
    python predict_unlabeled.py \
        --checkpoint saved_results/xxx/model.pth \
        --data_path  /path/to/unlabeled_images \
        --save_heatmap \
        --heatmap_topk 0
"""

import torch
import torch.nn as nn
import os
import argparse
import glob
import numpy as np
import pandas as pd
import cv2
from functools import partial
from PIL import Image
from torch.nn import functional as F

from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dataset import get_data_transforms, IMG_EXTENSIONS
from utils import cal_anomaly_maps, get_gaussian_kernel, min_max_norm, cvt2heatmap, show_cam_on_image


# ---------------------------------------------------------------------------
# Bounding box: 从 anomaly map 提取异常区域并画 YOLO 风格检测框
# ---------------------------------------------------------------------------
def extract_anomaly_bboxes(anomaly_map, bbox_threshold_ratio=0.5, min_area=100):
    """从 anomaly map 提取异常区域的 bounding boxes。

    Args:
        anomaly_map: (H, W) numpy array, 归一化到 [0, 1]。
        bbox_threshold_ratio: 异常图阈值比例，高于 max*ratio 的区域被视为异常。
        min_area: 最小连通区域面积，过滤噪声。

    Returns:
        list of (x, y, w, h, confidence)，confidence 为该区域内的最大异常值。
    """
    # 二值化
    thresh_val = anomaly_map.max() * bbox_threshold_ratio
    binary = (anomaly_map > thresh_val).astype(np.uint8) * 255

    # 形态学操作：闭运算填补小空洞，膨胀连接邻近区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 该区域内的最大异常值作为 confidence
        roi = anomaly_map[y : y + h, x : x + w]
        confidence = float(roi.max())
        bboxes.append((x, y, w, h, confidence))

    # 按 confidence 降序排列
    bboxes.sort(key=lambda b: b[4], reverse=True)
    return bboxes


def draw_bboxes(image, bboxes, color=(0, 0, 255), thickness=2):
    """在图像上画 YOLO 风格的 bounding boxes。

    Args:
        image: BGR numpy array (会被就地修改)。
        bboxes: list of (x, y, w, h, confidence)。
        color: 框颜色 (B, G, R)。
        thickness: 框线宽度。

    Returns:
        绘制了 bbox 的图像。
    """
    vis = image.copy()
    for x, y, w, h, conf in bboxes:
        # 框
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        # 标签背景 + 文字
        label = f"anomaly {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        # 标签放在框上方，如果上方空间不够就放框内顶部
        label_y = y - 4 if y - th - 4 > 0 else y + th + 4
        bg_y1 = label_y - th - 4
        bg_y2 = label_y + baseline
        cv2.rectangle(vis, (x, bg_y1), (x + tw + 4, bg_y2), color, -1)
        cv2.putText(vis, label, (x + 2, label_y - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Dataset: 从目录递归加载所有图像，无需标签
# ---------------------------------------------------------------------------
class UnlabeledImageDataset(torch.utils.data.Dataset):
    """加载目录下所有图像（递归搜索），不需要标签或 ground_truth。"""

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.img_paths = self._glob_all_images(root)
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {root}")
        self.img_paths.sort()
        print(f"[dataset] found {len(self.img_paths)} images in {root}")

    @staticmethod
    def _glob_all_images(directory):
        paths = []
        for ext in IMG_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        return paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_path


# ---------------------------------------------------------------------------
# 模型构建（与 predict.py / 训练脚本一致）
# ---------------------------------------------------------------------------
def build_model(encoder_name, device):
    patch_size = int(encoder_name.split("_")[-1])
    crop_size = (392 // patch_size) * patch_size

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unknown architecture in encoder_name: {encoder_name}")

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn_drop=0.0,
                attn=LinearAttention2,
            )
            for _ in range(8)
        ]
    )

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
    model = model.to(device)
    return model, crop_size


# ---------------------------------------------------------------------------
# 推理核心
# ---------------------------------------------------------------------------
def predict_unlabeled(
    model,
    dataloader,
    device,
    max_ratio=0.01,
    resize_mask=256,
    threshold=None,
    percentile=95.0,
    save_csv=None,
    save_heatmap=False,
    heatmap_dir=None,
    heatmap_topk=20,
    save_bbox=False,
    bbox_threshold_ratio=0.5,
    bbox_min_area=100,
):
    """对无标签数据集进行推理。

    Args:
        model: 训练好的模型。
        dataloader: 无标签数据集的 DataLoader (返回 img, img_path)。
        device: torch device。
        max_ratio: top-k ratio 计算 sample-level score (0 = global max)。
        resize_mask: 将 anomaly map resize 到此尺寸后再计分。
        threshold: 手动指定阈值。None 时用 percentile 自动计算。
        percentile: 自动阈值使用的百分位数（默认 95，即认为 top 5% 为异常）。
        save_csv: CSV 保存路径。
        save_heatmap: 是否保存热力图和/或 bbox。
        heatmap_dir: 热力图/bbox 保存目录。
        heatmap_topk: 只保存 top-k 最异常图像的可视化（0 = 全部保存）。
        save_bbox: 是否在图像上画 YOLO 风格异常检测框。
        bbox_threshold_ratio: bbox 二值化阈值 = max(anomaly_map) * ratio。
        bbox_min_area: 过滤面积小于此值的连通区域。

    Returns:
        results: list of dicts (img_path, score, pred)。
        threshold: 使用的阈值。
    """
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_paths = []
    all_scores = []
    # 如果需要热力图或 bbox，暂存 anomaly_map 和原图
    need_vis = save_heatmap or save_bbox
    all_anomaly_maps = [] if need_vis else None
    all_imgs = [] if need_vis else None

    with torch.no_grad():
        for img, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])

            if resize_mask is not None:
                anomaly_map = F.interpolate(
                    anomaly_map, size=resize_mask, mode="bilinear", align_corners=False
                )

            anomaly_map = gaussian_kernel(anomaly_map)

            # sample-level score
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                flat = anomaly_map.flatten(1)
                k = max(1, int(flat.shape[1] * max_ratio))
                sp_score = torch.sort(flat, dim=1, descending=True)[0][:, :k]
                sp_score = sp_score.mean(dim=1)

            all_paths.extend(list(img_path))
            all_scores.append(sp_score.cpu())

            if need_vis:
                all_anomaly_maps.append(anomaly_map.cpu())
                all_imgs.append(img.cpu())

    all_scores = torch.cat(all_scores).numpy()

    # 阈值
    if threshold is not None:
        thr_method = "manual"
    else:
        threshold = np.percentile(all_scores, percentile)
        thr_method = f"percentile_{percentile}"

    preds = (all_scores >= threshold).astype(int)

    # 构造结果
    results = []
    for path, score, pred in zip(all_paths, all_scores, preds):
        results.append(
            {
                "img_path": path,
                "score": float(score),
                "pred": int(pred),
                "pred_label": "anomaly" if pred == 1 else "normal",
            }
        )

    # 按分数降序排列
    results.sort(key=lambda r: r["score"], reverse=True)

    # 统计
    n_total = len(results)
    n_anomaly = sum(r["pred"] for r in results)
    n_normal = n_total - n_anomaly

    print(f"\n[predict] total: {n_total}  predicted anomaly: {n_anomaly}  predicted normal: {n_normal}")
    print(f"[predict] score distribution:")
    print(
        f"  all (n={n_total}): min={all_scores.min():.4f}  "
        f"mean={all_scores.mean():.4f}  max={all_scores.max():.4f}  "
        f"std={all_scores.std():.4f}"
    )
    print(f"[predict] threshold={threshold:.4f} ({thr_method})")

    # 打印 top-10 最异常
    print(f"\n[predict] top-10 most anomalous:")
    for i, r in enumerate(results[:10]):
        tag = "ANOMALY" if r["pred"] == 1 else "normal"
        print(f"  {i+1:>3d}. [{tag:>7s}] score={r['score']:.4f}  {r['img_path']}")

    # 保存 CSV
    if save_csv is not None:
        csv_dir = os.path.dirname(save_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(save_csv, index=False)
        print(f"\n[predict] saved CSV to {save_csv}")

    # 保存可视化（热力图 / bbox），各类型存到独立子文件夹
    if need_vis and heatmap_dir is not None:
        # 创建子文件夹
        dir_img = os.path.join(heatmap_dir, "img")
        dir_heatmap = os.path.join(heatmap_dir, "heatmap")
        dir_overlay = os.path.join(heatmap_dir, "overlay")
        dir_bbox = os.path.join(heatmap_dir, "bbox")

        if save_heatmap:
            os.makedirs(dir_img, exist_ok=True)
            os.makedirs(dir_heatmap, exist_ok=True)
            os.makedirs(dir_overlay, exist_ok=True)
        if save_bbox:
            os.makedirs(dir_bbox, exist_ok=True)

        # 合并所有 anomaly_map 和 img
        all_anomaly_maps = torch.cat(all_anomaly_maps, dim=0)  # (N, 1, H, W)
        all_imgs = torch.cat(all_imgs, dim=0)  # (N, 3, H, W)

        # 按分数排序的索引
        sorted_indices = np.argsort(all_scores)[::-1]
        if heatmap_topk > 0:
            sorted_indices = sorted_indices[:heatmap_topk]

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        vis_types = []
        if save_heatmap:
            vis_types.append("heatmap")
        if save_bbox:
            vis_types.append("bbox")
        print(f"\n[predict] saving {len(sorted_indices)} visualizations ({', '.join(vis_types)}) to {heatmap_dir}")

        for rank, idx in enumerate(sorted_indices):
            amap = all_anomaly_maps[idx, 0].numpy()  # (H, W)
            img_tensor = all_imgs[idx]  # (3, H, W)

            # 还原图像
            im = img_tensor.permute(1, 2, 0).numpy()
            im = im * std + mean
            im = np.clip(im * 255, 0, 255).astype("uint8")
            im_bgr = im[:, :, ::-1]

            # resize anomaly_map 到图像尺寸
            if amap.shape[0] != im.shape[0] or amap.shape[1] != im.shape[1]:
                amap = cv2.resize(amap, (im.shape[1], im.shape[0]))

            amap_norm = min_max_norm(amap)

            # 文件名
            orig_name = os.path.splitext(os.path.basename(all_paths[idx]))[0]
            tag = "anomaly" if preds[idx] == 1 else "normal"
            prefix = f"{rank+1:04d}_{all_scores[idx]:.4f}_{tag}_{orig_name}"

            if save_heatmap:
                heatmap = cvt2heatmap(amap_norm * 255)
                hm_on_img = show_cam_on_image(im_bgr, heatmap)
                cv2.imwrite(os.path.join(dir_img, f"{prefix}.png"), im_bgr)
                cv2.imwrite(os.path.join(dir_heatmap, f"{prefix}.png"), heatmap)
                cv2.imwrite(os.path.join(dir_overlay, f"{prefix}.png"), hm_on_img)

            if save_bbox:
                bboxes = extract_anomaly_bboxes(amap_norm, bbox_threshold_ratio, bbox_min_area)
                bbox_img = draw_bboxes(im_bgr, bboxes)
                cv2.imwrite(os.path.join(dir_bbox, f"{prefix}.png"), bbox_img)

        print(f"[predict] visualizations saved.")

    return results, threshold


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict_unlabeled] device: {device}")

    model, crop_size = build_model(args.encoder_name, device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[predict_unlabeled] loaded checkpoint: {args.checkpoint}")

    image_size = 448
    data_transform, _ = get_data_transforms(image_size, crop_size)

    dataset = UnlabeledImageDataset(root=args.data_path, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 可视化目录
    heatmap_dir = None
    if args.save_heatmap or args.save_bbox:
        heatmap_dir = args.heatmap_dir or os.path.join(
            os.path.dirname(args.save_csv) if args.save_csv else "./predictions",
            "heatmaps",
        )

    results, threshold = predict_unlabeled(
        model,
        dataloader,
        device,
        max_ratio=args.max_ratio,
        resize_mask=args.resize_mask,
        threshold=args.threshold,
        percentile=args.percentile,
        save_csv=args.save_csv,
        save_heatmap=args.save_heatmap,
        heatmap_dir=heatmap_dir,
        heatmap_topk=args.heatmap_topk,
        save_bbox=args.save_bbox,
        bbox_threshold_ratio=args.bbox_threshold_ratio,
        bbox_min_area=args.bbox_min_area,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict anomaly scores on an unlabeled image dataset."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model.pth"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Directory containing images (flat or nested, no labels needed)",
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="dinov2reg_vit_base_14",
        help="Must match the encoder used during training",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_ratio",
        type=float,
        default=0.01,
        help="Top-k ratio for sample-level score (0 = global max)",
    )
    parser.add_argument("--resize_mask", type=int, default=256)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold. If omitted, auto-select by percentile.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile for auto threshold (default 95: top 5%% are anomaly). "
        "Ignored if --threshold is given.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="./predictions/unlabeled_predictions.csv",
        help="CSV save path",
    )
    parser.add_argument(
        "--save_heatmap",
        action="store_true",
        help="Save anomaly heatmap visualizations",
    )
    parser.add_argument(
        "--heatmap_dir",
        type=str,
        default=None,
        help="Directory for heatmap images (default: alongside CSV)",
    )
    parser.add_argument(
        "--heatmap_topk",
        type=int,
        default=20,
        help="Only save heatmaps for top-k most anomalous images (0 = all)",
    )
    parser.add_argument(
        "--save_bbox",
        action="store_true",
        help="Draw YOLO-style bounding boxes around anomalous regions",
    )
    parser.add_argument(
        "--bbox_threshold_ratio",
        type=float,
        default=0.5,
        help="Bbox threshold = max(anomaly_map) * ratio (default 0.5). "
        "Lower = more sensitive (larger boxes), higher = stricter (smaller boxes).",
    )
    parser.add_argument(
        "--bbox_min_area",
        type=int,
        default=100,
        help="Minimum contour area in pixels to keep a bbox (filters noise)",
    )
    args = parser.parse_args()

    run(args)
