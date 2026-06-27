import matplotlib.pyplot as plt

# 绘制指标曲线
def plot_metrics_curve(train_losses, val_losses, val_psnrs, val_ssims, save_path, val_rmses=None):
    num_plots = 4 if val_rmses is not None else 3
    plt.figure(figsize=(6 * num_plots, 6))

    # 一行多列，第一个子图
    plt.subplot(1, num_plots, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.title("L1 Loss Evolution", fontsize=14, pad=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第二个子图
    plt.subplot(1, num_plots, 2)
    plt.plot(val_psnrs, label="PSNR", linewidth=2)
    plt.title("Validation PSNR", fontsize=14, pad=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第三个子图
    plt.subplot(1, num_plots, 3)
    plt.plot(val_ssims, label="SSIM", linewidth=2)
    plt.title("Validation SSIM", fontsize=14, pad=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if val_rmses is not None:
        # 第四个子图
        plt.subplot(1, num_plots, 4)
        plt.plot(val_rmses, label="RMSE", linewidth=2)
        plt.title("Validation RMSE", fontsize=14, pad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 调整子图间距
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=200)
    plt.close()


# 绘制CT对比图
def plot_ct_comparison(inp, gt, pred, save_path):
    inp_np = inp[0, 0].detach().cpu().numpy()
    gt_np = gt[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    images = [inp_np, pred_np, gt_np]
    titles = ["Input", "Prediction", "Ground Truth"]

    plt.figure(figsize=(18, 6))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(title, fontsize=14)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


