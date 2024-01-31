import argparse
import yaml
import time
import os
from tqdm import tqdm
import pandas as pd
import shutil

import torch

from Utils.Misc_utils import set_seeds
from Utils.Metrics import plot_metrics
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Evaluation import eval_model


def XTranPrune(
    main_model, SA_model, main_dataloader, SA_dataloader, device, config, verbose=2
):
    main_DL_iter = iter(main_dataloader)
    SA_DL_iter = iter(SA_dataloader)

    ###############################  Getting the attention masks for all modules ###############################

    num_tokens = main_model.patch_embed.num_patches + 1
    blk_attrs_shape = (
        main_model.depth,
        main_model.num_heads,
        num_tokens,
        num_tokens,
    )
    main_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)
    SA_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)

    for itr in tqdm(
        range(config["prune"]["num_batch_per_iter"]),
        total=config["prune"]["num_batch_per_iter"],
        desc="Generating masks",
    ):
        main_BR_batch = next(main_DL_iter)
        main_inputs = main_BR_batch["image"].to(device)
        main_labels = main_BR_batch[config["prune"]["main_level"]].to(device)

        SA_BR_batch = next(SA_DL_iter)
        SA_inputs = SA_BR_batch["image"].to(device)
        SA_labels = SA_BR_batch[config["prune"]["SA_level"]].to(device)

        main_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)
        SA_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)

        for i in range(main_inputs.shape[0]):  # iterate over batch size
            if config["prune"]["method"] == "attn":
                main_blk_attrs_input = main_model.generate_attn(
                    input=main_inputs[i].unsqueeze(0)
                )
                SA_blk_attrs_input = SA_model.generate_attn(
                    input=SA_inputs[i].unsqueeze(0)
                )
            else:
                cam, main_blk_attrs_input = main_model.generate_LRP(
                    input=main_inputs[i].unsqueeze(0),
                    index=main_labels[i],
                    method=config["prune"]["method"],
                )
                cam, SA_blk_attrs_input = SA_model.generate_LRP(
                    input=SA_inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                    method=config["prune"]["method"],
                )

            main_blk_attrs_batch = main_blk_attrs_batch + main_blk_attrs_input.detach()
            SA_blk_attrs_batch = SA_blk_attrs_batch + SA_blk_attrs_input.detach()

            main_blk_attrs_input = None
            SA_blk_attrs_input = None
            cam = None

        # Averaging the block importances for the batch
        main_blk_attrs_batch = main_blk_attrs_batch / main_inputs.shape[0]
        SA_blk_attrs_batch = SA_blk_attrs_batch / SA_inputs.shape[0]

        main_blk_attrs_iter = main_blk_attrs_iter + main_blk_attrs_batch
        SA_blk_attrs_iter = SA_blk_attrs_iter + SA_blk_attrs_batch

    main_blk_attrs_iter = main_blk_attrs_iter / config["prune"]["num_batch_per_iter"]
    SA_blk_attrs_iter = SA_blk_attrs_iter / config["prune"]["num_batch_per_iter"]

    ###############################  Generating the pruning mask ###############################

    prun_mask = []
    for blk_idx in range(main_blk_attrs_iter.shape[0]):
        prun_mask_blk = []
        for h in range(main_blk_attrs_iter.shape[1]):
            # Generating the main mask
            main_attrs_flt = main_blk_attrs_iter[blk_idx][h].flatten()

            threshold = torch.quantile(
                main_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
            )  # (this param)% of the most important main params will be retained

            main_mask = (main_blk_attrs_iter[blk_idx][h] < threshold).float()

            # Generating the pruning mask from SA branch
            can_be_pruned = SA_blk_attrs_iter[blk_idx][h] * main_mask
            can_be_pruned_flt = can_be_pruned.flatten()

            k = int(
                config["prune"]["pruning_rate"] * main_mask.sum()
            )  # Pruning Pruning_rate% of the paramters allowed by the main branch to be pruned

            top_k_values, top_k_indices = torch.topk(can_be_pruned_flt, k)
            prun_mask_blk_head = torch.ones_like(can_be_pruned_flt)
            prun_mask_blk_head[top_k_indices] = 0

            prun_mask_blk_head = prun_mask_blk_head.reshape(
                (main_blk_attrs_iter.shape[2], main_blk_attrs_iter.shape[3])
            )
            prun_mask_blk.append(prun_mask_blk_head)

            if verbose == 2:
                print(
                    f"#params pruned in head {h+1}: {(num_tokens*num_tokens) - prun_mask_blk_head.sum()}/{(num_tokens*num_tokens)} \
                    | Rate: {((num_tokens*num_tokens) - prun_mask_blk_head.sum())/(num_tokens*num_tokens)}"
                )

        prun_mask_blk = torch.stack(prun_mask_blk, dim=0)
        prun_mask.append(prun_mask_blk)
        if verbose == 2:
            print(
                f"@@@ #params pruned in block {blk_idx+1}: {(num_tokens*num_tokens*main_model.num_heads) - prun_mask_blk.sum()}/{(num_tokens*num_tokens*main_model.num_heads)} \
                | Rate: {((num_tokens*num_tokens*main_model.num_heads) - prun_mask_blk.sum())/(num_tokens*num_tokens*main_model.num_heads)}"
            )

    prun_mask = torch.stack(prun_mask, dim=0)

    prev_mask = main_model.get_attn_mask()

    if verbose > 0 and prev_mask is not None:
        new_mask = prev_mask * prun_mask
        num_pruned_prev = (
            prev_mask.shape[0]
            * prev_mask.shape[1]
            * prev_mask.shape[2]
            * prev_mask.shape[3]
        ) - prev_mask.sum()
        num_pruned_new = (
            new_mask.shape[0]
            * new_mask.shape[1]
            * new_mask.shape[2]
            * new_mask.shape[3]
        ) - new_mask.sum()
        print(
            f"New #pruned_parameters - Previous #pruned_parameters = {num_pruned_new} - {num_pruned_prev} = {num_pruned_new - num_pruned_prev} "
        )

    main_model.set_attn_mask(prun_mask)

    return main_model


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Pruning configs:")
    print(config["prune"])

    shutil.copy(
        "Configs/configs_server.yml",
        os.path.join(config["output_folder_path"], "configs.yml"),
    )

    set_seeds(config["seed"])

    main_dataloaders, main_dataset_sizes, main_num_classes = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        dataset_name=config["dataset_name"],
        level=config["prune"]["main_level"],
        batch_size=config["prune"]["batch_size"],
        num_workers=1,
    )

    SA_dataloaders, SA_dataset_sizes, SA_num_classes = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        dataset_name=config["dataset_name"],
        level=config["prune"]["SA_level"],
        batch_size=config["prune"]["batch_size"],
        num_workers=1,
    )

    dataloaders, dataset_sizes, num_classes = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        dataset_name=config["dataset_name"],
        level=config["prune"]["main_level"],
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    # load both models
    main_model = deit_small_patch16_224(
        num_classes=main_num_classes,
        add_hook=True,
        weight_path=config["prune"]["main_br_path"],
    )
    main_model = main_model.eval().to(device)

    SA_model = deit_small_patch16_224(
        num_classes=SA_num_classes,
        add_hook=True,
        weight_path=config["prune"]["SA_br_path"],
    )
    SA_model = SA_model.eval().to(device)

    prun_iter_cnt = 0
    consecutive_no_improvement = 0
    best_bias_metric = config["prune"]["bias_metric_prev"]
    val_metrics_df = None

    while (
        consecutive_no_improvement <= config["prune"]["max_consecutive_no_improvement"]
    ):
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt+1} +++++++++++++++++++++++++++++"
        )

        if prun_iter_cnt == 0:
            pruned_model = XTranPrune(
                main_model=main_model,
                SA_model=SA_model,
                main_dataloader=main_dataloaders["train"],
                SA_dataloader=SA_dataloaders["train"],
                device=device,
                verbose=config["prune"]["verbose"],
                config=config,
            )
        else:
            pruned_model = XTranPrune(
                main_model=pruned_model,
                SA_model=SA_model,
                main_dataloader=main_dataloaders["train"],
                SA_dataloader=SA_dataloaders["train"],
                device=device,
                verbose=config["prune"]["verbose"],
                config=config,
            )

        model_name = f"DeiT_S_LRP_PIter{prun_iter_cnt+1}"

        val_metrics, _ = eval_model(
            pruned_model,
            dataloaders,
            dataset_sizes,
            num_classes,
            device,
            config["prune"]["main_level"],
            model_name,
            config,
            save_preds=True,
        )

        if config["prune"]["target_bias_metric"] in [
            "EOpp0",
            "EOpp1",
            "EOdd",
            "NAR",
            "NFR_W",
            "NFR_Mac",
        ]:
            if val_metrics[config["prune"]["target_bias_metric"]] < best_bias_metric:
                best_bias_metric = val_metrics[config["prune"]["target_bias_metric"]]

                # Save the best model
                print(
                    f'Achieved new leading val metrics: {config["prune"]["target_bias_metric"]}={best_bias_metric} \n'
                )

                # Reset the counter
                consecutive_no_improvement = 0
            else:
                print(
                    f"No improvements observed in Iteration {prun_iter_cnt+1}, val metrics: \n"
                )
                consecutive_no_improvement += 1
        else:
            if val_metrics[config["prune"]["target_bias_metric"]] > best_bias_metric:
                best_bias_metric = val_metrics[config["prune"]["target_bias_metric"]]

                # Save the best model
                print(
                    f'Achieved new leading val metrics: {config["prune"]["target_bias_metric"]}={best_bias_metric} \n'
                )

                # Reset the counter
                consecutive_no_improvement = 0
            else:
                print(
                    f"No improvements observed in Iteration {prun_iter_cnt+1}, val metrics: \n"
                )
                consecutive_no_improvement += 1

        print(val_metrics)

        model_path = os.path.join(
            config["output_folder_path"],
            f"{model_name}.pth",
        )
        checkpoint = {
            "config": config,
            "leading_val_metrics": val_metrics,
            "model_state_dict": pruned_model.state_dict(),
        }
        torch.save(checkpoint, model_path)

        prun_iter_cnt += 1

        time_elapsed = time.time() - since
        print(
            "This iteration took {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        if prun_iter_cnt == 0:
            val_metrics_df = pd.DataFrame([val_metrics])
        else:
            val_metrics_df = pd.concat(
                [val_metrics_df, pd.DataFrame([val_metrics])], ignore_index=True
            )
        val_metrics_df.to_csv(
            os.path.join(config["output_folder_path"], f"Pruning_metrics.csv"),
            index=False,
        )

        plot_metrics(val_metrics_df, ["F1_Mac", "Worst_F1_Mac"], "F1", config)
        plot_metrics(val_metrics_df, ["DPM", "EOM"], "positive", config)
        plot_metrics(
            val_metrics_df,
            ["EOpp0", "EOpp1", "EOdd", "NFR_Mac"],
            "negative",
            config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
