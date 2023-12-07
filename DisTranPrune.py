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


def DisTranPrune(
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
            cam, main_blk_attrs_input = main_model.generate_LRP(
                input=main_inputs[i].unsqueeze(0), index=main_labels[i]
            )
            cam, SA_blk_attrs_input = SA_model.generate_LRP(
                input=SA_inputs[i].unsqueeze(0), index=SA_labels[i]
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
                    f"#params pruned in head {h}: {(num_tokens*num_tokens) - prun_mask_blk_head.sum()}/{(num_tokens*num_tokens)} 
                    | Rate: {((num_tokens*num_tokens) - prun_mask_blk_head.sum())/(num_tokens*num_tokens)}"
                )
            
        prun_mask_blk = torch.stack(prun_mask_blk, dim=0)
        prun_mask.append(prun_mask_blk)
        if verbose == 2:
            print(
                f"@@@ #params pruned in block {blk_idx}: {(num_tokens*num_tokens*main_model.num_heads) - prun_mask_blk.sum()}/{(num_tokens*num_tokens*main_model.num_heads)} 
                | Rate: {((num_tokens*num_tokens*main_model.num_heads) - prun_mask_blk.sum())/(num_tokens*num_tokens*main_model.num_heads)}"
            )

    prun_mask = torch.stack(prun_mask, dim=0)
    
    prev_mask = main_model.get_attn_mask()

    if verbose>0 and prev_mask is not None:
    
        num_pruned_prev = (prev_mask.shape[0]*prev_mask.shape[1]*prev_mask.shape[2]*prev_mask.shape[3]) - prev_mask.sum()
        num_pruned_new = (prun_mask.shape[0]*prun_mask.shape[1]*prun_mask.shape[2]*prun_mask.shape[3]) - prun_mask.sum()
        print(f"New #pruned_parameters - Previous #pruned_parameters = {num_pruned_new} - {num_pruned_prev} = {num_pruned_new - num_pruned_prev} ")
    
    main_model.set_attn_mask(prun_mask)

    return main_model


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    main_dataloaders, main_dataset_sizes, main_num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["prune"]["main_level"],
        holdout_set="random_holdout",
        batch_size=config["prune"]["batch_size"],
        num_workers=1,
    )

    SA_dataloaders, SA_dataset_sizes, SA_num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["prune"]["SA_level"],
        holdout_set="random_holdout",
        batch_size=config["prune"]["batch_size"],
        num_workers=1,
    )

    dataloaders, dataset_sizes, num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["prune"]["main_level"],
        holdout_set="random_holdout",
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

    while (
        consecutive_no_improvement <= config["prune"]["max_consecutive_no_improvement"]
    ):
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt} +++++++++++++++++++++++++++++"
        )

        if prun_iter_cnt == 0:
            pruned_model = DisTranPrune(
                main_model=main_model,
                SA_model=SA_model,
                main_dataloader=main_dataloaders["train"],
                SA_dataloader=SA_dataloaders["train"],
                device=device,
                config=config,
            )
        else:
            pruned_model = DisTranPrune(
                main_model=pruned_model,
                SA_model=SA_model,
                main_dataloader=main_dataloaders["train"],
                SA_dataloader=SA_dataloaders["train"],
                device=device,
                config=config,
            )

        model_name = f"DeiT_S_LRP_PIter{prun_iter_cnt}"

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

        if val_metrics[config["prune"]["target_bias_metric"]] > best_bias_metric:
            best_bias_metric = val_metrics[config["prune"]["target_bias_metric"]]

            # Save the best model
            print("New leading val metrics, saving the weights...\n")
            print(val_metrics)

            best_model_path = os.path.join(
                config["output_folder_path"],
                f"DeiT_S_LRP_checkpoint_prune_Iter={prun_iter_cnt}.pth",
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
            print(
                f"No improvements observed in Iteration {prun_iter_cnt}, val metrics: \n"
            )
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
