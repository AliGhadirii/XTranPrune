import argparse
import yaml
import time
import os
from tqdm import tqdm
import pandas as pd
import shutil
import sys

import torch

from Utils.Misc_utils import set_seeds, Logger, get_stat, get_mask_idx
from Utils.Metrics import plot_metrics
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Explainability.ViT_Explainer import Explainer
from Evaluation import eval_model


def XTranPrune(
    main_model,
    SA_model,
    main_dataloader,
    SA_dataloader,
    device,
    config,
    prun_iter_cnt,
    verbose=2,
    MA_vectors=None,
):
    main_DL_iter = iter(main_dataloader)
    SA_DL_iter = iter(SA_dataloader)

    main_explainer = Explainer(main_model)
    SA_explainer = Explainer(SA_model)

    ###############################  Getting the attribution vectors for all nodes in both branches ###############################

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
            if config["prune"]["cont_method"] == "attn":
                main_blk_attrs_input = main_explainer.generate_attn(
                    input=main_inputs[i].unsqueeze(0)
                )
                SA_blk_attrs_input = SA_explainer.generate_attn(
                    input=SA_inputs[i].unsqueeze(0)
                )
            elif config["prune"]["cont_method"] == "TranInter":
                _, main_blk_attrs_input = main_explainer.generate_TranInter(
                    input=main_inputs[i].unsqueeze(0),
                    index=main_labels[i],
                )
                _, SA_blk_attrs_input = SA_explainer.generate_TranInter(
                    input=SA_inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                )

            elif config["prune"]["cont_method"] == "TAM":
                
                _, main_blk_attrs_input = main_explainer.generate_TAM(
                    input=main_inputs[i].unsqueeze(0),
                    index=main_labels[i],
                    start_layer=0,
                    steps=10,
                )
                _, SA_blk_attrs_input = SA_explainer.generate_TAM(
                    input=SA_inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                    start_layer=0,
                    steps=10,
                )
                main_blk_attrs_input = main_blk_attrs_input.squeeze(0)
                SA_blk_attrs_input = SA_blk_attrs_input.squeeze(0)
            elif config["prune"]["cont_method"] == "AttrRoll":
                    
                _, main_blk_attrs_input = main_explainer.generate_AttrRoll(input=main_inputs[i].unsqueeze(0), index=main_labels[i])
                _, SA_blk_attrs_input = SA_explainer.generate_AttrRoll(input=SA_inputs[i].unsqueeze(0), index=SA_labels[i])
            elif config["prune"]["cont_method"] == "FTeylor":
                main_blk_attrs_input = main_explainer.generate_FTeylor(
                    input=main_inputs[i].unsqueeze(0),
                    index=main_labels[i],
                )
                SA_blk_attrs_input = SA_explainer.generate_FTeylor(
                    input=SA_inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                )
            elif config["prune"]["cont_method"] == "FTeylorpow2":
                main_blk_attrs_input = main_explainer.generate_FTeylorpow2(
                    input=main_inputs[i].unsqueeze(0),
                    index=main_labels[i],
                )
                SA_blk_attrs_input = SA_explainer.generate_FTeylorpow2(
                    input=SA_inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                )
            else:
                _, main_blk_attrs_input = main_explainer.generate_LRP(
                    input=main_inputs[i].unsqueeze(0),
                    index=main_labels[i],
                    method=config["prune"]["cont_method"],
                )
                _, SA_blk_attrs_input = SA_explainer.generate_LRP(
                    input=SA_inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                    method=config["prune"]["cont_method"],
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

    torch.save(
        main_blk_attrs_iter,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"main_attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )
    torch.save(
        SA_blk_attrs_iter,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"SA_attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )

    # getting the moving average of attribution vectors and measuing the uncertainty
    if config["prune"]["method"] == "MA":

        main_blk_attrs_MA = MA_vectors[0]
        SA_blk_attrs_MA = MA_vectors[1]

        if main_blk_attrs_MA == None:
            main_blk_attrs_MA = torch.zeros_like(main_blk_attrs_iter)
        if SA_blk_attrs_MA == None:
            SA_blk_attrs_MA = torch.zeros_like(SA_blk_attrs_iter)

        beta1 = config["prune"]["beta1"]
        main_blk_attrs_MA = (
            beta1 * main_blk_attrs_MA + (1 - beta1) * main_blk_attrs_iter
        )
        SA_blk_attrs_MA = beta1 * SA_blk_attrs_MA + (1 - beta1) * SA_blk_attrs_iter

        MA_vectors[0] = main_blk_attrs_MA
        MA_vectors[1] = SA_blk_attrs_MA

        main_blk_attrs_iter_final = main_blk_attrs_MA
        SA_blk_attrs_iter_final = SA_blk_attrs_MA

    elif config["prune"]["method"] == "MA_Uncertainty":
        main_blk_attrs_MA = MA_vectors[0]
        SA_blk_attrs_MA = MA_vectors[1]
        main_blk_Uncer_MA = MA_vectors[2]
        SA_blk_Uncer_MA = MA_vectors[3]

        if main_blk_attrs_MA == None:
            main_blk_attrs_MA = torch.zeros_like(main_blk_attrs_iter)
        if SA_blk_attrs_MA == None:
            SA_blk_attrs_MA = torch.zeros_like(SA_blk_attrs_iter)
        if main_blk_Uncer_MA == None:
            main_blk_Uncer_MA = torch.zeros_like(main_blk_attrs_iter)
        if SA_blk_Uncer_MA == None:
            SA_blk_Uncer_MA = torch.zeros_like(SA_blk_attrs_iter)

        beta1 = config["prune"]["beta1"]
        main_blk_attrs_MA = (
            beta1 * main_blk_attrs_MA + (1 - beta1) * main_blk_attrs_iter
        )
        SA_blk_attrs_MA = beta1 * SA_blk_attrs_MA + (1 - beta1) * SA_blk_attrs_iter

        beta2 = config["prune"]["beta2"]
        main_blk_Uncer_MA = (
            beta2 * main_blk_Uncer_MA
            + (1 - beta2) * (main_blk_attrs_iter - main_blk_attrs_MA).abs()
        )
        SA_blk_Uncer_MA = (
            beta2 * SA_blk_Uncer_MA
            + (1 - beta2) * (SA_blk_attrs_iter - SA_blk_attrs_MA).abs()
        )

        MA_vectors[0] = main_blk_attrs_MA
        MA_vectors[1] = SA_blk_attrs_MA
        MA_vectors[2] = main_blk_Uncer_MA
        MA_vectors[3] = SA_blk_Uncer_MA

        SA_blk_Uncer_MA_flt = SA_blk_Uncer_MA.view(
            SA_blk_Uncer_MA.size(0), SA_blk_Uncer_MA.size(1), -1
        )
        SA_blk_Uncer_MA_max_values, _ = SA_blk_Uncer_MA_flt.max(dim=2, keepdim=True)
        SA_blk_Uncer_MA_max_values = SA_blk_Uncer_MA_max_values.unsqueeze(2)

        main_blk_attrs_iter_final = main_blk_attrs_MA * main_blk_Uncer_MA
        SA_blk_attrs_iter_final = SA_blk_attrs_MA * (
            SA_blk_Uncer_MA_max_values - SA_blk_Uncer_MA
        )
    else:
        main_blk_attrs_iter_final = main_blk_attrs_iter
        SA_blk_attrs_iter_final = SA_blk_attrs_iter

    ###############################  Generating the pruning mask ###############################

    prun_mask = []
    for blk_idx in range(main_blk_attrs_iter_final.shape[0]):
        prun_mask_blk = []
        for h in range(main_blk_attrs_iter_final.shape[1]):
            # Generating the main mask
            main_attrs_flt = main_blk_attrs_iter_final[blk_idx][h].flatten()

            threshold = torch.quantile(
                main_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
            )  # (this param)% of the most important main params will be retained

            main_mask = (main_blk_attrs_iter_final[blk_idx][h] < threshold).float()
            torch.save(
                main_mask,
                os.path.join(
                    config["output_folder_path"],
                    "Log_files",
                    f"main_mask_Iter={prun_iter_cnt + 1}.pth",
                ),
            )

            # Generating the pruning mask from SA branch
            can_be_pruned = SA_blk_attrs_iter_final[blk_idx][h] * main_mask
            can_be_pruned_flt = can_be_pruned.flatten()

            k = int(
                config["prune"]["pruning_rate"] * main_mask.sum()
            )  # Pruning Pruning_rate% of the paramters allowed by the main branch to be pruned

            top_k_values, top_k_indices = torch.topk(can_be_pruned_flt, k)
            prun_mask_blk_head = torch.ones_like(can_be_pruned_flt)
            prun_mask_blk_head[top_k_indices] = 0

            prun_mask_blk_head = prun_mask_blk_head.reshape(
                (main_blk_attrs_iter_final.shape[2], main_blk_attrs_iter_final.shape[3])
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

    # NEEDS FIXING: generalize it LATER
    if prun_mask.shape[0] < main_model.depth:
        ones_tensor = torch.ones((main_model.depth - prun_mask.shape[0],) + prun_mask.shape[1:], dtype=prun_mask.dtype, device=prun_mask.device)
            
        # Concatenate the original tensor with the ones tensor along the first dimension
        prun_mask = torch.cat([prun_mask, ones_tensor], dim=0)

    prev_mask = main_model.get_attn_pruning_mask()

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
    main_model.set_attn_pruning_mask(prun_mask)

    return main_model, MA_vectors


def main(config, args):

    if not os.path.exists(os.path.join(config["output_folder_path"], "Log_files")):
        os.mkdir(os.path.join(config["output_folder_path"], "Log_files"))

    if not os.path.exists(os.path.join(config["output_folder_path"], "Weights")):
        os.mkdir(os.path.join(config["output_folder_path"], "Weights"))

    log_file_path = os.path.join(
        config["output_folder_path"], "Log_files", "output.log"
    )

    sys.stdout = Logger(log_file_path)

    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Pruning configs:")
    print(config["prune"])

    shutil.copy(
        args.config,
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
        need_ig=True if config["prune"]["cont_method"] == "AttrRoll" else False,
        weight_path=config["prune"]["main_br_path"],
    )
    main_model = main_model.eval().to(device)

    SA_model = deit_small_patch16_224(
        num_classes=SA_num_classes,
        add_hook=True,
        need_ig=True if config["prune"]["cont_method"] == "AttrRoll" else False,
        weight_path=config["prune"]["SA_br_path"],
    )
    SA_model = SA_model.eval().to(device)

    prun_iter_cnt = 0
    consecutive_no_improvement = 0
    best_bias_metric = config["prune"]["bias_metric_prev"]
    val_metrics_df = None

    main_blk_attrs_MA = None
    SA_blk_attrs_MA = None
    main_blk_Uncer_MA = None
    SA_blk_Uncer_MA = None

    MA_vectors = [
        main_blk_attrs_MA,
        SA_blk_attrs_MA,
        main_blk_Uncer_MA,
        SA_blk_Uncer_MA,
    ]

    while (
        consecutive_no_improvement <= config["prune"]["max_consecutive_no_improvement"]
    ):
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt+1} +++++++++++++++++++++++++++++"
        )

        model_name = f"DeiT_S_LRP_PIter{prun_iter_cnt+1}"

        if prun_iter_cnt == 0:
            pruned_model, MA_vectors = XTranPrune(
                main_model=main_model,
                SA_model=SA_model,
                main_dataloader=main_dataloaders["train"],
                SA_dataloader=SA_dataloaders["train"],
                device=device,
                verbose=config["prune"]["verbose"],
                config=config,
                MA_vectors=MA_vectors,
                prun_iter_cnt=prun_iter_cnt,
            )
        else:
            pruned_model, MA_vectors = XTranPrune(
                main_model=pruned_model,
                SA_model=SA_model,
                main_dataloader=main_dataloaders["train"],
                SA_dataloader=SA_dataloaders["train"],
                device=device,
                verbose=config["prune"]["verbose"],
                config=config,
                MA_vectors=MA_vectors,
                prun_iter_cnt=prun_iter_cnt,
            )

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
            "Weights",
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
    main(config, args)
