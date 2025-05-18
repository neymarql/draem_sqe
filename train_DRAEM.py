#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRAEM 训练脚本 – 单机多 GPU（DistributedDataParallel）版
保留原始逻辑，仅补充 DDP、分布式采样器、主进程输出/保存控制等多卡支持代码。
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data_loader import MVTecDRAEMTrainDataset, RefinedDRAEMTrainDataset
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
from test_DRAEM import test


# ------------------------- 工具函数 ------------------------- #
def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def weights_init(m):
    """与原脚本一致的权重初始化"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def is_main_process(rank: int) -> bool:
    """判断是否主进程（rank==0）"""
    return rank == 0


# ------------------------- 训练核心 ------------------------- #
def train_on_device(obj_names, args, rank, world_size):
    """在指定 GPU（由 rank 决定）上训练选定类别"""

    # 仅主进程负责创建目录，其他进程等待同步
    if is_main_process(rank):
        os.makedirs(args.checkpoint_path, exist_ok=True)
        os.makedirs(args.log_path, exist_ok=True)
    dist.barrier()  # 同步所有进程

    for obj_name in obj_names:
        run_name = (f"DRAEM_ddp_{args.lr}_{args.epochs}_bs{args.bs}_"
                    f"{obj_name}_rank{rank}")

        # -------------- 构建模型并封装为 DDP -------------- #
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.apply(weights_init).cuda(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.apply(weights_init).cuda(rank)
        model_seg = DDP(model_seg, device_ids=[rank], output_device=rank)



        optimizer = torch.optim.Adam(
            [{"params": model.parameters(), "lr": args.lr},
             {"params": model_seg.parameters(), "lr": args.lr}]
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.epochs * 0.8, args.epochs * 0.9],
            gamma=0.2, last_epoch=-1
        )

        loss_l2 = torch.nn.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        # --------- 数据集、分布式采样器与 DataLoader --------- #
        if args.use_refined:
            anomaly_dir = os.path.join(args.anomaly_source_path, obj_name)
            dataset = RefinedDRAEMTrainDataset(
                os.path.join(args.data_path, obj_name, "train/good"),
                anomaly_dir,
                resize_shape=[256, 256],
            )
        else:
            dataset = MVTecDRAEMTrainDataset(
                os.path.join(args.data_path, obj_name, "train/good"),
                args.anomaly_source_path,
                resize_shape=[256, 256],
            )

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=False)
        dataloader = DataLoader(
            dataset, batch_size=args.bs, sampler=sampler,
            num_workers=8, pin_memory=True
        )

        # --------------------- 训练循环 --------------------- #
        n_iter = 0
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)          # 确保每轮 shuffle 不同
            epoch_loss, batch_count = 0.0, 0

            if is_main_process(rank):
                print(f"[Rank {rank}] Epoch {epoch}")

            for batch in dataloader:
                gray_batch = batch["image"].cuda(rank, non_blocking=True)
                aug_gray_batch = batch["augmented_image"].cuda(rank, non_blocking=True)
                anomaly_mask = batch["anomaly_mask"].cuda(rank, non_blocking=True)

                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # 三项损失
                l2_loss = loss_l2(gray_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                n_iter += 1

            # 各进程计算平均 loss
            avg_loss_tensor = torch.tensor(epoch_loss / max(batch_count, 1),
                                           device=rank)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / world_size).item()

            if is_main_process(rank):
                print(f"Epoch {epoch} completed | Avg Loss (all GPUs): {avg_loss:.4f}")

            scheduler.step()

            # -------------- 保存权重 & 测试（仅主进程） -------------- #
            if is_main_process(rank):
                torch.save(model.module.state_dict(),
                           os.path.join(args.checkpoint_path, f"{run_name}.pckl"))
                torch.save(model_seg.module.state_dict(),
                           os.path.join(args.checkpoint_path, f"{run_name}_seg.pckl"))

                # 每轮结束后测试
                print("Running test after epoch", epoch)
                test([obj_name], args.data_path, args.checkpoint_path, run_name)

        torch.cuda.empty_cache()


# ------------------------- 主函数 ------------------------- #
def main():
    parser = argparse.ArgumentParser()
    # 与原脚本基本一致，删除 gpu_id，新增分布式相关参数
    parser.add_argument("--obj_id", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True,
                        help="每卡 batch_size；总 batch 为 bs × nproc_per_node")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--anomaly_source_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--use_refined", action="store_true",
                        help="使用 RefinedDRAEMTrainDataset 并带 anomaly mask")
    # ↓↓ 以下两项由 torchrun 自动传入，勿手动指定 ↓↓
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="（由 torchrun 注入）")
    parser.add_argument("--world_size", type=int, default=-1,
                        help="（由 torchrun 注入）")

    args = parser.parse_args()
    if args.local_rank == -1:  # 兼容某些 torchrun / launch 情况
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # ---------------- 初始化分布式 ---------------- #
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)

    # ---------------- 对象类别 ---------------- #
    obj_batch = [
        ["capsule"], ["bottle"], ["carpet"], ["leather"], ["pill"],
        ["transistor"], ["tile"], ["cable"], ["zipper"], ["toothbrush"],
        ["metal_nut"], ["hazelnut"], ["screw"], ["grid"], ["wood"],
    ]

    if args.obj_id == -1:
        picked_classes = [o[0] for o in obj_batch]  # 全部 15 类
    else:
        picked_classes = obj_batch[args.obj_id]

    # ---------------- 开始训练 ---------------- #
    train_on_device(picked_classes, args, rank, world_size)

    # ---------------- 结束清理 ---------------- #
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
