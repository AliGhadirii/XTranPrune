import argparse
import yaml
import time
import os
from tqdm import tqdm

import torch

from Utils.Misc_utils import set_seeds
from Datasets.dataloaders import get_fitz17k_dataloaders
from Models.ViT_LRP.ViT_LRP import deit_small_patch16_224
from Evaluation import eval_model


def describe_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    stats = {
        "Mean": tensor.mean().item(),
        "Std Deviation": tensor.std().item(),
        "Minimum Value": tensor.min().item(),
        "Maximum Value": tensor.max().item(),
    }

    return stats


def DisTranPrune(main_model, SA_model, dataloaders, device, config):
    train_DL_iter = iter(dataloaders["train"])
    # HARDCODED: TOBE fixed
    blk_attrs_shape = (12, 6, 197, 197)
    main_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)
    SA_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)

    for itr in tqdm(
        range(config["prune"]["num_batch_per_iter"]),
        total=config["prune"]["num_batch_per_iter"],
        desc="Generating masks",
    ):
        batch = next(train_DL_iter)
        inputs = batch["image"].to(device)
        labels = batch["high"].to(device)

        main_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)
        SA_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)

        for i in range(inputs.shape[0]):  # iterate over batch size
            cam, main_blk_attrs_input = main_model.generate_LRP(
                input=inputs[i].unsqueeze(0), index=labels[i]
            )
            cam, SA_blk_attrs_input = SA_model.generate_LRP(
                input=inputs[i].unsqueeze(0), index=labels[i]
            )

            main_blk_attrs_batch = main_blk_attrs_batch + main_blk_attrs_input.detach()
            SA_blk_attrs_batch = SA_blk_attrs_batch + SA_blk_attrs_input.detach()

            main_blk_attrs_input = None
            SA_blk_attrs_input = None

        # Averaging the block importances for the batch
        main_blk_attrs_batch = main_blk_attrs_batch / inputs.shape[0]
        SA_blk_attrs_batch = SA_blk_attrs_batch / inputs.shape[0]

        main_blk_attrs_iter = main_blk_attrs_iter + main_blk_attrs_batch
        SA_blk_attrs_iter = SA_blk_attrs_iter + SA_blk_attrs_batch

    main_blk_attrs_iter = main_blk_attrs_iter / config["prune"]["num_batch_per_iter"]
    SA_blk_attrs_iter = SA_blk_attrs_iter / config["prune"]["num_batch_per_iter"]

    prun_mask = []
    for blk_idx in range(main_blk_attrs_iter.shape[0]):
        prun_mask_blk = []
        for h in range(main_blk_attrs_iter.shape[1]):
            # Generating the main mask
            main_attrs_flt = main_blk_attrs_iter[blk_idx][h].flatten()

            threshold = torch.quantile(
                main_attrs_flt, config["prune"]["main_mask_quantile"]
            )  # (1- this param)% of the most important main params will be retained

            # print(f"threshold={threshold}")

            main_mask = (main_blk_attrs_iter[blk_idx][h] < threshold).float()

            # print(f"number of params allowed to be pruned {main_mask.sum()}")

            # Generating the pruning mask from SA branch
            can_be_pruned = SA_blk_attrs_iter[blk_idx][h] * main_mask
            can_be_pruned_flt = can_be_pruned.flatten()

            k = int(
                config["prune"]["pruning_rate"] * main_mask.sum()
            )  # Pruning Pruning_rate% of the paramters allowed by the main branch to be pruned

            top_k_values, top_k_indices = torch.topk(can_be_pruned_flt, k)
            prun_mask_blk_head = torch.ones_like(can_be_pruned_flt)
            prun_mask_blk_head[top_k_indices] = 0
            # HARDCODED: TOBE fixed
            prun_mask_blk_head = prun_mask_blk_head.reshape((197, 197))
            prun_mask_blk.append(prun_mask_blk_head)

            # print(
            #     f"number of params pruned in head {h}: {(197*197) - prun_mask_blk_head.sum()}/{(197*197)}"
            # )
        prun_mask_blk = torch.stack(prun_mask_blk, dim=0)
        prun_mask.append(prun_mask_blk)
        # print(
        #     f"+++++++++++++++++++++++++++++ Block {blk_idx} +++++++++++++++++++++++++++++"
        # )

    prun_mask = torch.stack(prun_mask, dim=0)
    main_model.set_attn_mask(prun_mask)
    return main_model


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    p_dataloaders, p_dataset_sizes, p_num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        holdout_set="random_holdout",
        batch_size=config["prune"]["batch_size"],
        num_workers=1,
    )

    dataloaders, dataset_sizes, num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        holdout_set="random_holdout",
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    # load both models
    main_model = deit_small_patch16_224(
        pretrained=False,
        num_classes=3,
        add_hook=True,
        weight_path=config["prune"]["main_br_path"],
    )
    main_model = main_model.eval().to(device)

    SA_model = deit_small_patch16_224(
        pretrained=False,
        num_classes=6,
        add_hook=True,
        weight_path=config["prune"]["SA_br_path"],
    )
    SA_model = SA_model.eval().to(device)

    prun_iter_cnt = 0
    consecutive_no_improvement = 0
    best_bias_metric = config["prune"]["bias_metric_prev"]

    while (
        consecutive_no_improvement <= config["prune"]["max_consecutive_no_improvement"]
    ):
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt} +++++++++++++++++++++++++++++"
        )

        if prun_iter_cnt == 0:
            pruned_model = DisTranPrune(
                main_model, SA_model, p_dataloaders, device, config
            )
        else:
            pruned_model = DisTranPrune(
                pruned_model, SA_model, p_dataloaders, device, config
            )

        model_name = f"DeiT-S_LRP_PIter{prun_iter_cnt}"

        val_metrics, _ = eval_model(
            pruned_model,
            dataloaders,
            dataset_sizes,
            num_classes
            device,
            config["default"]["level"],
            model_name,
            config,
            save_preds=True,
        )

        if val_metrics[config["prune"]["target_bias_metric"]] > best_bias_metric:
            best_bias_metric = val_metrics[config["prune"]["target_bias_metric"]]

            # Save the best model
            print("New leading model val metrics, saving the weights...\n")
            print(val_metrics)

            best_model_path = os.path.join(
                config["output_folder_path"],
                f"DeiT-S_LRP_checkpoint_prune_Iter={prun_iter_cnt}.pth",
            )
            checkpoint = {
                "config": config,
                "leading_val_metrics": val_metrics,
                "model_state_dict": pruned_model.state_dict(),
            }
            torch.save(checkpoint, best_model_path)
            print("Checkpoint saved:", best_model_path)

            # Reset the counter
            consecutive_no_improvement = 0
        else:
            print(f"No improvements in Iteration {prun_iter_cnt}, val metrics: \n")
            print(val_metrics)
            consecutive_no_improvement += 1

        prun_iter_cnt += 1

        time_elapsed = time.time() - since
        print(
            "This iteration took {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
