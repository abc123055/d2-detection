"""
独立推理脚本：加载训练好的模型权重，对测试集逐图输出 正常/异常 预测结果。

用法：
    python predict.py \
        --checkpoint saved_results/xxx/model.pth \
        --data_path  ../mvtec_anomaly_detection \
        --item       bottle \
        --encoder_name dinov2reg_vit_base_14 \
        --save_csv   results/bottle_predictions.csv

    也可以一次跑多个类别：
    python predict.py \
        --checkpoint saved_results/xxx/model.pth \
        --data_path  ../mvtec_anomaly_detection \
        --item       bottle carpet grid \
        --encoder_name dinov2reg_vit_base_14

    手动指定阈值（不自动选最优）：
    python predict.py \
        --checkpoint saved_results/xxx/model.pth \
        --data_path  ../mvtec_anomaly_detection \
        --item       bottle \
        --encoder_name dinov2reg_vit_base_14 \
        --threshold  0.35
"""

import torch
import torch.nn as nn
import os
import argparse
from functools import partial

from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dataset import MVTecDataset, get_data_transforms
from utils import predict_batch


def build_model(encoder_name, device):
    """构建模型结构（和训练脚本一致），返回未加载权重的模型。"""
    patch_size = int(encoder_name.split('_')[-1])
    crop_size = (392 // patch_size) * patch_size

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unknown architecture in encoder_name: {encoder_name}")

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn_drop=0., attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=target_layers, mask_neighbor_size=0,
                   fuse_layer_encoder=fuse_layer_encoder,
                   fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    return model, crop_size


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, crop_size = build_model(args.encoder_name, device)

    # 加载权重
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[predict] loaded checkpoint: {args.checkpoint}")

    image_size = 448
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    for item in args.item:
        print(f"\n{'='*50}")
        print(f"[predict] {item}")
        print(f"{'='*50}")

        test_path = os.path.join(args.data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 确定 CSV 保存路径
        if args.save_csv and len(args.item) == 1:
            csv_path = args.save_csv
        else:
            csv_path = os.path.join(args.save_dir, f'{item}_predictions.csv')

        results, threshold = predict_batch(
            model, test_dataloader, device,
            max_ratio=args.max_ratio,
            resize_mask=args.resize_mask,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            normal_percentile=args.normal_percentile,
            save_csv=csv_path,
        )

        # 打印错误样本
        errors = [r for r in results if not r['correct']]
        if errors:
            print(f"\n[predict] {item}: {len(errors)} errors:")
            for r in errors:
                tag = "误报(FP)" if r['label'] == 0 else "漏检(FN)"
                print(f"  {tag}  score={r['score']:.4f}  {r['img_path']}")
        else:
            print(f"\n[predict] {item}: all correct!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load trained model and predict per-image results.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model.pth')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Dataset root (same as training)')
    parser.add_argument('--item', type=str, nargs='+', required=True,
                        help='Category name(s), e.g. bottle carpet')
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14',
                        help='Must match the encoder used during training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_ratio', type=float, default=0.01,
                        help='Top-k ratio for sample-level score (0 = global max)')
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed threshold. If omitted, auto-select by --threshold_mode')
    parser.add_argument('--threshold_mode', type=str, default='f1', choices=['f1', 'normal'],
                        help='Auto threshold strategy when --threshold not given. '
                             'f1: best F1 on test data (overfits, for analysis only). '
                             'normal: percentile of normal-image scores (stable for deployment).')
    parser.add_argument('--normal_percentile', type=float, default=99.0,
                        help='Percentile for --threshold_mode=normal. '
                             '99 means at most 1%% of normal images become false alarms.')
    parser.add_argument('--save_csv', type=str, default=None,
                        help='CSV save path (only for single item). '
                             'For multiple items, auto-saves to --save_dir/<item>_predictions.csv')
    parser.add_argument('--save_dir', type=str, default='./predictions',
                        help='Directory for CSV files when running multiple items')
    args = parser.parse_args()

    run(args)
