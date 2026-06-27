import os
import numpy as np
from tqdm import tqdm  
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset_code.dual_domain_dataset import ImageDomainDataset
from model.Image_Net import ImageRestorer

from utils.visual_utils import plot_ct_comparison, plot_metrics_curve
from utils.train_utils import (
    set_seed,
    create_exp_dir,
    calc_batch_psnr_ssim,
    tensor_to_numpy_img,
    build_warmup_cosine_scheduler,
    save_used_code_files,
    save_checkpoint,
    EarlyStopping
)

def main():

    # ============== 随机种子 =================
    set_seed(42)

    # ============== 输出目录 =================
    exp_dir = create_exp_dir(
        root_dir = "./outputs",
        exp_name = "image_domain"
    )

    print("输出目录：", exp_dir)

    # ============== 保存代码快照 =================
    save_used_code_files(
        file_paths=[
            "ImageDomain_train.py",
            "dataset_code/dual_domain_dataset.py",
            "model/Image_Net.py",
            "utils/train_utils.py",
            "utils/visual_utils.py",
        ],
        save_dir=os.path.join(exp_dir, "code")
    )


    # ============== 设备 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ============== 模型 =================
    model = ImageRestorer(
        in_c=1,
        out_c=1,
        stage1_width=32,
        stage2_width=32,
        num_cab=6
    ).to(device)

    # ============== 损失函数 =================
    criterion = nn.L1Loss()

    # ============== 优化器 =================
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # ============== 训练配置 =================
    num_epochs = 100
    batch_size = 2
    num_workers = 4
    pin_memory = True

    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []

    log_path = os.path.join(exp_dir, "logs", "train_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,valid_loss,psnr,ssim,lr\n")

    best_psnr = 0.0

    early_stopper = EarlyStopping(
        patience=20,
        mode="max",
        min_delta=0.001
    )

    #============ 数据加载 =================
    train_dataset = ImageDomainDataset(
        input_dir = "./dataset/dataset_view18/train/input",
        gt_dir = "./dataset/dataset_view18/train/gt"
    )

    valid_dataset = ImageDomainDataset(
        input_dir = "./dataset/dataset_view18/valid/input",
        gt_dir = "./dataset/dataset_view18/valid/gt"
    )

    test_dataset = ImageDomainDataset(
        input_dir = "./dataset/dataset_view18/test/input",
        gt_dir = "./dataset/dataset_view18/test/gt"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print("train 数据数量:", len(train_dataset))
    print("valid 数据数量:", len(valid_dataset))
    print("test 数据数量:", len(test_dataset))


    # ============== 学习率调度器 =================
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=5,
        total_epochs=num_epochs,
        min_lr=1e-6
    )

    # ============== 训练主循环 =================
    for epoch in range(num_epochs):
        print(f"\n========== Epoch [{epoch+1}/{num_epochs}] ==========")

        # ============== train =================
        model.train()
        total_loss = 0

        for x,y,name in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            final_pred, stage1_pred = model(x)
            loss = criterion(final_pred, y) + 0.1 * criterion(stage1_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ============== valid =================
        model.eval()
        valid_loss = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        img_count = 0

        with torch.no_grad():
            for x,y,name in tqdm(valid_loader):
                x = x.to(device)
                y = y.to(device)

                final_pred, stage1_pred = model(x)
                loss = criterion(final_pred, y) + 0.1 * criterion(stage1_pred, y)
                pred_for_metric = torch.clamp(final_pred, 0.0, 1.0)

                
                valid_loss += loss.item()
                p_sum, s_sum, bs = calc_batch_psnr_ssim(pred_for_metric, y)
                psnr_sum += p_sum
                ssim_sum += s_sum
                img_count += bs

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_psnr = psnr_sum / img_count
        avg_ssim = ssim_sum / img_count

        # ============== scheduler =================
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Valid Loss: {avg_valid_loss:.6f} | "
            f"PSNR: {avg_psnr:.4f} | "
            f"SSIM: {avg_ssim:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_valid_loss)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},"
                f"{avg_train_loss:.6f},"
                f"{avg_valid_loss:.6f},"
                f"{avg_psnr:.4f},"
                f"{avg_ssim:.4f},"
                f"{current_lr:.8e}\n"
            )

        plot_metrics_curve(
            train_losses=train_losses,
            val_losses=val_losses,
            val_psnrs=val_psnrs,
            val_ssims=val_ssims,
            save_path=os.path.join(exp_dir, "curve", "metrics_curve.png")
        )

        #对比图
        plot_ct_comparison(
            inp=x,
            gt=y,
            pred=pred_for_metric,
            save_path=os.path.join(exp_dir, "compare", f"epoch_{epoch+1:03d}.png")
        )

        # ============== 保存模型 =================
        # 保存 best
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_psnr,
                save_path=os.path.join(exp_dir, "checkpoints", "model_best.pth")
            )

            print(" ⭐ 保存最佳模型 ⭐")


        # 保存 latest 模型
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_metric=best_psnr,
            save_path=os.path.join(exp_dir, "checkpoints", "model_latest.pth")
        )

        # ============= 早停 =================
        if early_stopper(avg_psnr):
            print(f"早停触发：连续 {early_stopper.patience} 轮 PSNR 没有提升")
            break


if __name__=="__main__":
    main()




















