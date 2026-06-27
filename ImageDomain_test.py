import os
import time

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset_code.dual_domain_dataset import ImageDomainDataset
from model.Image_Net import ImageRestorer
from utils.train_utils import calc_batch_psnr_ssim
from utils.visual_utils import plot_ct_comparison


def calc_batch_rmse(pred, gt):
    """
    计算 batch 内逐图 RMSE 的总和。
    返回 total_rmse, batch_size，便于和 PSNR/SSIM 一样做平均。
    """
    pred = pred.detach().float()
    gt = gt.detach().float()
    bs = pred.shape[0]
    mse_per_img = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    total_rmse = torch.sqrt(mse_per_img).sum().item()
    return total_rmse, bs


def sync_cuda_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============== 路径 =================
    test_input_dir = "./dataset/dataset_view18/test/input"
    test_gt_dir = "./dataset/dataset_view18/test/gt"

    exp_dir = "./outputs/image_domain_20260626"
    ckpt_path = os.path.join(exp_dir, "checkpoints", "model_best.pth")
    save_dir = os.path.join(exp_dir, "test_results")
    os.makedirs(save_dir, exist_ok=True)

    pred_dir = os.path.join(save_dir, "pred")
    compare_dir = os.path.join(save_dir, "compare")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "test_metrics.csv")

    # ============== 数据 =================
    test_dataset = ImageDomainDataset(
        input_dir=test_input_dir,
        gt_dir=test_gt_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ================ 模型 =================
    model = ImageRestorer(
        in_c=1,
        out_c=1,
        stage1_width=32,
        stage2_width=32,
        num_cab=6
    ).to(device)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("已加载最佳模型:", ckpt_path)
    print("测试输入目录:", test_input_dir)
    print("测试 GT 目录:", test_gt_dir)
    print("结果保存目录:", save_dir)

    # 预热一次，避免第一张图的 CUDA 初始化开销影响平均单图推理速度。
    with torch.no_grad():
        warmup_x = torch.zeros(1, 1, 256, 256, device=device)
        _ = model(warmup_x)
        sync_cuda_if_needed(device)

    with open(log_path, "w") as f:
        f.write("filename,psnr,ssim,rmse,inference_time_ms\n")

    # ============== 测试 =================
    model.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    rmse_sum = 0.0
    infer_time_sum = 0.0
    img_count = 0

    with torch.no_grad():
        for x, y, name in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync_cuda_if_needed(device)
            start_time = time.perf_counter()
            final_pred, stage1_pred = model(x)
            sync_cuda_if_needed(device)
            infer_time_ms = (time.perf_counter() - start_time) * 1000.0

            pred_for_metric = torch.clamp(final_pred, min=0.0, max=1.0)

            pred_np = pred_for_metric[0, 0].detach().cpu().numpy()
            pred_png = (pred_np * 255.0).clip(0, 255).astype(np.uint8)
            save_name = os.path.splitext(name[0])[0]

            Image.fromarray(pred_png).save(
                os.path.join(pred_dir, save_name + "_pred.png")
            )

            p_sum, s_sum, bs = calc_batch_psnr_ssim(pred_for_metric, y)
            r_sum, _ = calc_batch_rmse(pred_for_metric, y)

            psnr_sum += p_sum
            ssim_sum += s_sum
            rmse_sum += r_sum
            infer_time_sum += infer_time_ms
            img_count += bs

            with open(log_path, "a") as f:
                f.write(
                    f"{name[0]},"
                    f"{p_sum / bs:.4f},"
                    f"{s_sum / bs:.4f},"
                    f"{r_sum / bs:.6f},"
                    f"{infer_time_ms / bs:.4f}\n"
                )

            plot_ct_comparison(
                inp=x,
                gt=y,
                pred=pred_for_metric,
                save_path=os.path.join(compare_dir, f"{save_name}_compare.png")
            )

    avg_psnr = psnr_sum / img_count
    avg_ssim = ssim_sum / img_count
    avg_rmse = rmse_sum / img_count
    avg_infer_time_ms = infer_time_sum / img_count
    fps = 1000.0 / avg_infer_time_ms if avg_infer_time_ms > 0 else 0.0

    with open(log_path, "a") as f:
        f.write(
            f"AVERAGE,"
            f"{avg_psnr:.4f},"
            f"{avg_ssim:.4f},"
            f"{avg_rmse:.6f},"
            f"{avg_infer_time_ms:.4f}\n"
        )

    print("========== Test Result ==========")
    print(f"Test PSNR: {avg_psnr:.4f}")
    print(f"Test SSIM: {avg_ssim:.4f}")
    print(f"Test RMSE: {avg_rmse:.6f}")
    print(f"Avg inference time: {avg_infer_time_ms:.4f} ms/image")
    print(f"Inference speed: {fps:.2f} images/s")


if __name__ == "__main__":
    main()
