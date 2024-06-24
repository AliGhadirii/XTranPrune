import argparse
import yaml
import time
import os
from tqdm import tqdm
import pandas as pd
import shutil
import sys
from pprint import pprint

import torch

from Utils.Misc_utils import set_seeds, Logger, get_stat, get_mask_idx, js_divergence
from Utils.Metrics import plot_metrics
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Explainability.ViT_Explainer import Explainer
from Evaluation import eval_model


def Contrastive(
    model,
    dataloaders,
    S0_dataloaders,
    S1_dataloaders,
    device,
    config,
    prun_iter_cnt,
    verbose=2,
    MA_vectors=None,
):
    DL_iter = iter(dataloaders)
    S0_iter = iter(S0_dataloaders)
    S1_iter = iter(S1_dataloaders)

    explainer = Explainer(model)

    ###############################  Getting the attribution vectors for all nodes in both branches ###############################

    num_tokens = model.patch_embed.num_patches + 1
    blk_attrs_shape = (
        model.depth,
        model.num_heads,
        num_tokens,
        num_tokens,
    )
    blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)
    S0_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)
    S1_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)

    for itr in tqdm(
        range(config["prune"]["num_batch_per_iter"]),
        total=config["prune"]["num_batch_per_iter"],
        desc="Generating masks",
    ):
        batch = next(DL_iter)
        inputs = batch["image"].to(device)
        labels = batch[config["prune"]["main_level"]].to(device)

        S0_batch = next(S0_iter)
        S0_inputs = S0_batch["image"].to(device)
        S0_labels = S0_batch[config["prune"]["main_level"]].to(device)

        S1_batch = next(S1_iter)
        S1_inputs = S1_batch["image"].to(device)
        S1_labels = S1_batch[config["prune"]["SA_level"]].to(device)

        blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)
        S0_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)
        S1_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)

        for i in range(config["prune"]["batch_size"]):  # iterate over batch size
            if config["prune"]["cont_method"] == "attn":
                blk_attrs_input = explainer.generate_attn(input=inputs[i].unsqueeze(0))
                S0_blk_attrs_input = explainer.generate_attn(
                    input=S0_inputs[i].unsqueeze(0)
                )
                S1_blk_attrs_input = explainer.generate_attn(
                    input=S1_inputs[i].unsqueeze(0)
                )
            elif config["prune"]["cont_method"] == "TranInter":
                _, blk_attrs_input = explainer.generate_TranInter(
                    input=inputs[i].unsqueeze(0),
                    index=labels[i],
                )
                _, S0_blk_attrs_input = explainer.generate_TranInter(
                    input=S0_inputs[i].unsqueeze(0),
                    index=S0_labels[i],
                )
                _, S1_blk_attrs_input = explainer.generate_TranInter(
                    input=S1_inputs[i].unsqueeze(0),
                    index=S1_labels[i],
                )

            elif config["prune"]["cont_method"] == "TAM":
                _, blk_attrs_input = explainer.generate_TAM(
                    input=inputs[i].unsqueeze(0),
                    index=labels[i],
                    start_layer=0,
                    steps=10,
                )
                _, S0_blk_attrs_input = explainer.generate_TAM(
                    input=S0_inputs[i].unsqueeze(0),
                    index=S0_labels[i],
                    start_layer=0,
                    steps=10,
                )
                _, S1_blk_attrs_input = explainer.generate_TAM(
                    input=S1_inputs[i].unsqueeze(0),
                    index=S1_labels[i],
                    start_layer=0,
                    steps=10,
                )
                blk_attrs_input = blk_attrs_input.squeeze(0)
                S0_blk_attrs_input = S0_blk_attrs_input.squeeze(0)
                S1_blk_attrs_input = S1_blk_attrs_input.squeeze(0)
            elif config["prune"]["cont_method"] == "AttrRoll":
                _, blk_attrs_input = explainer.generate_AttrRoll(
                    input=inputs[i].unsqueeze(0), index=labels[i]
                )
                _, S0_blk_attrs_input = explainer.generate_AttrRoll(
                    input=S0_inputs[i].unsqueeze(0), index=S0_labels[i]
                )
                _, S1_blk_attrs_input = explainer.generate_AttrRoll(
                    input=S1_inputs[i].unsqueeze(0), index=S1_labels[i]
                )
            elif config["prune"]["cont_method"] == "FTaylor":
                blk_attrs_input = explainer.generate_FTaylor(
                    input=inputs[i].unsqueeze(0),
                    index=labels[i],
                )
                S0_blk_attrs_input = explainer.generate_FTaylor(
                    input=S0_inputs[i].unsqueeze(0),
                    index=S0_labels[i],
                )
                S1_blk_attrs_input = explainer.generate_FTaylor(
                    input=S1_inputs[i].unsqueeze(0),
                    index=S1_labels[i],
                )
            elif config["prune"]["cont_method"] == "FTaylorpow2":
                blk_attrs_input = explainer.generate_FTaylorpow2(
                    input=inputs[i].unsqueeze(0),
                    index=labels[i],
                )
                S0_blk_attrs_input = explainer.generate_FTaylorpow2(
                    input=S0_inputs[i].unsqueeze(0),
                    index=S0_labels[i],
                )
                S1_blk_attrs_input = explainer.generate_FTaylorpow2(
                    input=S1_inputs[i].unsqueeze(0),
                    index=S1_labels[i],
                )
            else:
                _, blk_attrs_input = explainer.generate_LRP(
                    input=inputs[i].unsqueeze(0),
                    index=labels[i],
                    method=config["prune"]["cont_method"],
                )
                _, S0_blk_attrs_input = explainer.generate_LRP(
                    input=S0_inputs[i].unsqueeze(0),
                    index=S0_labels[i],
                    method=config["prune"]["cont_method"],
                )
                _, S1_blk_attrs_input = explainer.generate_LRP(
                    input=S1_inputs[i].unsqueeze(0),
                    index=S1_labels[i],
                    method=config["prune"]["cont_method"],
                )

            blk_attrs_batch = blk_attrs_batch + blk_attrs_input.detach()
            S0_blk_attrs_batch = S0_blk_attrs_batch + S0_blk_attrs_input.detach()
            S1_blk_attrs_batch = S1_blk_attrs_batch + S1_blk_attrs_input.detach()

            blk_attrs_input = None
            S0_blk_attrs_input = None
            S1_blk_attrs_input = None

        # Averaging the block importances for the batch
        blk_attrs_batch = blk_attrs_batch / inputs.shape[0]
        S0_blk_attrs_batch = S0_blk_attrs_batch / S0_inputs.shape[0]
        S1_blk_attrs_batch = S1_blk_attrs_batch / S1_inputs.shape[0]

        blk_attrs_iter = blk_attrs_iter + blk_attrs_batch
        S0_blk_attrs_iter = S0_blk_attrs_iter + S0_blk_attrs_batch
        S1_blk_attrs_iter = S1_blk_attrs_iter + S1_blk_attrs_batch

    blk_attrs_iter = blk_attrs_iter / config["prune"]["num_batch_per_iter"]
    S0_blk_attrs_iter = S0_blk_attrs_iter / config["prune"]["num_batch_per_iter"]
    S1_blk_attrs_iter = S1_blk_attrs_iter / config["prune"]["num_batch_per_iter"]

    torch.save(
        blk_attrs_iter,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )
    torch.save(
        S0_blk_attrs_iter,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"S0_attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )
    torch.save(
        S1_blk_attrs_iter,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"S1_attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )

    if config["prune"]["FObjective"] == "MinJSD_Diff":
        jsd = torch.ones(S0_blk_attrs_iter.shape[0], S0_blk_attrs_iter.shape[1]).to(
            device
        )
        for encoder_idx in range(jsd.shape[0]):
            for head_idx in range(jsd.shape[1]):

                # Get the attention masks for the current head and encoder
                P = S0_blk_attrs_iter[encoder_idx, head_idx, :, :]
                Q = S1_blk_attrs_iter[encoder_idx, head_idx, :, :]

                # Calculate JSD and store in the result tensor
                jsd_value = js_divergence(P, Q)
                jsd[encoder_idx, head_idx] = jsd_value

        disc_blk_attrs_iter = jsd.unsqueeze(-1).unsqueeze(-1) * (
            S0_blk_attrs_iter - S1_blk_attrs_iter
        )

    elif config["prune"]["FObjective"] == "MinDiff":
        disc_blk_attrs_iter = S0_blk_attrs_iter - S1_blk_attrs_iter
    else:
        raise ValueError("Invalid FObjective")

    # getting the moving average of attribution vectors and measuing the uncertainty
    if config["prune"]["method"] == "MA":

        blk_attrs_MA = MA_vectors[0]
        disc_blk_attrs_MA = MA_vectors[1]

        if blk_attrs_MA == None:
            blk_attrs_MA = torch.zeros_like(blk_attrs_iter)
        if disc_blk_attrs_MA == None:
            disc_blk_attrs_MA = torch.zeros_like(disc_blk_attrs_iter)

        beta1 = config["prune"]["beta1"]
        blk_attrs_MA = beta1 * blk_attrs_MA + (1 - beta1) * blk_attrs_iter
        disc_blk_attrs_MA = (
            beta1 * disc_blk_attrs_MA + (1 - beta1) * disc_blk_attrs_iter
        )

        MA_vectors[0] = blk_attrs_MA
        MA_vectors[1] = disc_blk_attrs_MA

        blk_attrs_iter_final = blk_attrs_MA
        disc_blk_attrs_iter_final = disc_blk_attrs_MA

    elif config["prune"]["method"] == "MA_Uncertainty":

        blk_attrs_MA = MA_vectors[0]
        disc_blk_attrs_MA = MA_vectors[1]
        blk_Uncer_MA = MA_vectors[2]
        disc_blk_Uncer_MA = MA_vectors[3]

        if blk_attrs_MA == None:
            blk_attrs_MA = torch.zeros_like(blk_attrs_iter)
        if disc_blk_attrs_MA == None:
            disc_blk_attrs_MA = torch.zeros_like(disc_blk_attrs_iter)
        if blk_Uncer_MA == None:
            blk_Uncer_MA = torch.zeros_like(blk_attrs_iter)
        if disc_blk_Uncer_MA == None:
            disc_blk_Uncer_MA = torch.zeros_like(disc_blk_attrs_iter)

        beta1 = config["prune"]["beta1"]
        blk_attrs_MA = beta1 * blk_attrs_MA + (1 - beta1) * blk_attrs_iter
        disc_blk_attrs_MA = (
            beta1 * disc_blk_attrs_MA + (1 - beta1) * disc_blk_attrs_iter
        )

        beta2 = config["prune"]["beta2"]
        blk_Uncer_MA = (
            beta2 * blk_Uncer_MA + (1 - beta2) * (blk_attrs_iter - blk_attrs_MA).abs()
        )
        disc_blk_Uncer_MA = (
            beta2 * disc_blk_Uncer_MA
            + (1 - beta2) * (disc_blk_attrs_iter - disc_blk_Uncer_MA).abs()
        )

        MA_vectors[0] = blk_attrs_MA
        MA_vectors[1] = disc_blk_attrs_MA
        MA_vectors[2] = blk_Uncer_MA
        MA_vectors[3] = disc_blk_Uncer_MA

        disc_blk_Uncer_MA_flt = disc_blk_Uncer_MA.view(
            disc_blk_Uncer_MA.size(0), disc_blk_Uncer_MA.size(1), -1
        )
        disc_blk_Uncer_MA_max_values, _ = disc_blk_Uncer_MA_flt.max(dim=2, keepdim=True)
        disc_blk_Uncer_MA_max_values = disc_blk_Uncer_MA_max_values.unsqueeze(2)

        # more uncertainty makes the importance score higher, thus the node is less probabable to be pruned
        blk_attrs_iter_final = blk_attrs_MA * blk_Uncer_MA
        disc_blk_attrs_iter_final = disc_blk_attrs_MA * (
            disc_blk_Uncer_MA_max_values - disc_blk_Uncer_MA
        )
    else:
        blk_attrs_iter_final = blk_attrs_iter
        disc_blk_attrs_iter_final = disc_blk_attrs_iter

    ###############################  Generating the pruning mask ###############################

    prun_mask = []
    for blk_idx in range(blk_attrs_iter_final.shape[0]):
        prun_mask_blk = []
        for h in range(blk_attrs_iter_final.shape[1]):

            blk_attrs_flt = blk_attrs_iter_final[blk_idx][h].flatten()

            # (this param)% of the most important main params will be retained
            threshold = torch.quantile(
                blk_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
            )

            performance_mask = (blk_attrs_iter_final[blk_idx][h] < threshold).float()
            torch.save(
                performance_mask,
                os.path.join(
                    config["output_folder_path"],
                    "Log_files",
                    f"performance_mask_Iter={prun_iter_cnt + 1}.pth",
                ),
            )

            # Generating the pruning mask from SA branch
            score = disc_blk_attrs_iter_final[blk_idx][h].abs() * performance_mask
            score_flt = score.flatten()

            # Pruning Pruning_rate% of the paramters
            k = int(config["prune"]["pruning_rate"] * performance_mask.sum())

            top_k_values, top_k_indices = torch.topk(score_flt, k)
            prun_mask_blk_head = torch.ones_like(score_flt)
            prun_mask_blk_head[top_k_indices] = 0

            prun_mask_blk_head = prun_mask_blk_head.reshape(
                (blk_attrs_iter_final.shape[2], blk_attrs_iter_final.shape[3])
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
                f"@@@ #params pruned in block {blk_idx+1}: {(num_tokens*num_tokens*model.num_heads) - prun_mask_blk.sum()}/{(num_tokens*num_tokens*model.num_heads)} \
                | Rate: {((num_tokens*num_tokens*model.num_heads) - prun_mask_blk.sum())/(num_tokens*num_tokens*model.num_heads)}"
            )

    prun_mask = torch.stack(prun_mask, dim=0)
    torch.save(
        prun_mask,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"pruning_mask_Iter={prun_iter_cnt + 1}.pth",
        ),
    )

    # NEEDS FIXING: generalize it LATER
    if prun_mask.shape[0] < model.depth:
        ones_tensor = torch.ones(
            (model.depth - prun_mask.shape[0],) + prun_mask.shape[1:],
            dtype=prun_mask.dtype,
            device=prun_mask.device,
        )

        # Concatenate the original tensor with the ones tensor along the first dimension
        prun_mask = torch.cat([prun_mask, ones_tensor], dim=0)

    prev_mask = model.get_attn_pruning_mask()

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
    model.set_attn_pruning_mask(prun_mask)

    return model, MA_vectors


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

    dataloaders, dataset_sizes, main_num_classes, SA_num_classes = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        sampler_type=config["default"]["sampler_type"],
        dataset_name=config["dataset_name"],
        stratify_cols=config["default"]["stratify_cols"],
        main_level=config["prune"]["main_level"],
        SA_level=config["prune"]["SA_level"],
        batch_size=config["prune"]["batch_size"],
        num_workers=1,
    )

    val_dataloaders, val_dataset_sizes, val_main_num_classes, val_SA_num_classes = (
        get_dataloaders(
            root_image_dir=config["root_image_dir"],
            Generated_csv_path=config["Generated_csv_path"],
            sampler_type="WeightedRandom",
            dataset_name=config["dataset_name"],
            stratify_cols=["low"],
            main_level=config["prune"]["main_level"],
            SA_level=config["prune"]["SA_level"],
            batch_size=config["default"]["batch_size"],
            num_workers=1,
        )
    )

    # load both models
    model = deit_small_patch16_224(
        num_classes=main_num_classes,
        add_hook=True,
        need_ig=True if config["prune"]["cont_method"] == "AttrRoll" else False,
        weight_path=config["prune"]["main_br_path"],
    )
    model = model.eval().to(device)

    val_metrics, _ = eval_model(
        model,
        val_dataloaders,
        val_dataset_sizes,
        val_main_num_classes,
        device,
        config["prune"]["main_level"],
        "DeiT_S_LRP_PIter0",
        config,
        save_preds=True,
    )

    val_metrics_df = pd.DataFrame([val_metrics])

    print("Validation metrics using the original model:")
    pprint(val_metrics)
    print()

    prun_iter_cnt = 0
    consecutive_no_improvement = 0
    best_bias_metric = config["prune"]["bias_metric_prev"]

    blk_attrs_MA = None
    disc_blk_attrs_MA = None
    blk_Uncer_MA = None
    disc_blk_Uncer_MA = None

    MA_vectors = [blk_attrs_MA, disc_blk_attrs_MA, blk_Uncer_MA, disc_blk_Uncer_MA]

    while (
        consecutive_no_improvement <= config["prune"]["max_consecutive_no_improvement"]
    ):
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt+1} +++++++++++++++++++++++++++++"
        )

        model_name = f"DeiT_S_LRP_PIter{prun_iter_cnt+1}"

        if prun_iter_cnt == 0:
            pruned_model, MA_vectors = Contrastive(
                model=model,
                dataloaders=dataloaders["train"],
                S0_dataloaders=S0_dataloaders["train"],
                S1_dataloaders=S1_dataloaders["train"],
                device=device,
                config=config,
                prun_iter_cnt=prun_iter_cnt,
                verbose=config["prune"]["verbose"],
                MA_vectors=MA_vectors,
            )
        else:
            pruned_model, MA_vectors = Contrastive(
                model=pruned_model,
                dataloaders=dataloaders["train"],
                S0_dataloaders=S0_dataloaders["train"],
                S1_dataloaders=S1_dataloaders["train"],
                device=device,
                config=config,
                prun_iter_cnt=prun_iter_cnt,
                verbose=config["prune"]["verbose"],
                MA_vectors=MA_vectors,
            )

        val_metrics, _ = eval_model(
            pruned_model,
            val_dataloaders,
            val_dataset_sizes,
            val_main_num_classes,
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

        pprint(val_metrics)
        print()

        model_path = os.path.join(
            config["output_folder_path"],
            "Weights",
            f"{model_name}.pth",
        )
        checkpoint = {
            "config": config,
            "leading_val_metrics": val_metrics,
            # "model_state_dict": pruned_model.state_dict(),
            "model": pruned_model,
        }
        torch.save(checkpoint, model_path)

        prun_iter_cnt += 1

        time_elapsed = time.time() - since
        print(
            "This iteration took {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

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
        plot_metrics(
            val_metrics_df,
            ["EOpp0_new", "EOpp1_new", "EOdd_new"],
            "negative_new",
            config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config, args)
