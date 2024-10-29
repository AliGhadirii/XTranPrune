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
import networkx as nx

from Utils.Misc_utils import (
    set_seeds,
    Logger,
    preprocess_matrix,
    get_stat,
    get_mask_idx,
)
from Utils.Metrics import plot_metrics
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Explainability.ViT_Explainer import Explainer
from Evaluation import eval_model


def get_required_matrices(
    main_model,
    SA_model,
    dataloader,
    device,
    config,
    required_matrices=["attr", "attn"],
):

    assert all(
        param in ["attr", "attn", "grad"] for param in required_matrices
    ), "Not all provided elements are in the required_matrices list."

    DL_iter = iter(dataloader)

    num_tokens = main_model.patch_embed.num_patches + 1
    matrix_shape = (
        main_model.depth,
        main_model.num_heads,
        num_tokens,
        num_tokens,
    )

    if "attr" in required_matrices:
        main_explainer = Explainer(main_model)
        SA_explainer = Explainer(SA_model)

        main_attr_iter = torch.zeros(matrix_shape).to(device)
        SA_attr_iter = torch.zeros(matrix_shape).to(device)
    if "attn" in required_matrices:
        main_attn_iter = torch.zeros(matrix_shape).to(device)
        SA_attn_iter = torch.zeros(matrix_shape).to(device)
    if "grad" in required_matrices:
        main_grad_iter = torch.zeros(matrix_shape).to(device)
        SA_grad_iter = torch.zeros(matrix_shape).to(device)

    for itr in tqdm(
        range(config["prune"]["num_batch_per_iter"]),
        total=config["prune"]["num_batch_per_iter"],
        desc="Generating node-level required matrices",
    ):

        try:
            batch = next(DL_iter)
        except StopIteration:
            DL_iter = iter(dataloader)
            batch = next(DL_iter)

        inputs = batch["image"].to(device)
        main_labels = batch[config["train"]["main_level"]].to(device)
        SA_labels = batch[config["train"]["SA_level"]].to(device)

        if "attr" in required_matrices:

            main_attr_batch = torch.zeros(matrix_shape).to(device)
            SA_attr_batch = torch.zeros(matrix_shape).to(device)

            for i in range(inputs.shape[0]):
                if config["prune"]["cont_method"] == "attn":
                    main_attr_input = main_explainer.generate_attn(
                        input=inputs[i].unsqueeze(0)
                    )
                    SA_attr_input = SA_explainer.generate_attn(
                        input=inputs[i].unsqueeze(0)
                    )
                elif config["prune"]["cont_method"] == "TranInter":
                    _, main_attr_input = main_explainer.generate_TranInter(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                    )
                    _, SA_attr_input = SA_explainer.generate_TranInter(
                        input=inputs[i].unsqueeze(0),
                        index=SA_labels[i],
                    )
                elif config["prune"]["cont_method"] == "TAM":

                    _, main_attr_input = main_explainer.generate_TAM(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                        start_layer=0,
                        steps=10,
                    )
                    _, SA_attr_input = SA_explainer.generate_TAM(
                        input=inputs[i].unsqueeze(0),
                        index=SA_labels[i],
                        start_layer=0,
                        steps=10,
                    )
                    main_attr_input = main_attr_input.squeeze(0)
                    SA_attr_input = SA_attr_input.squeeze(0)
                elif config["prune"]["cont_method"] == "AttrRoll":

                    _, main_attr_input = main_explainer.generate_AttrRoll(
                        input=inputs[i].unsqueeze(0), index=main_labels[i]
                    )
                    _, SA_attr_input = SA_explainer.generate_AttrRoll(
                        input=inputs[i].unsqueeze(0), index=SA_labels[i]
                    )
                elif config["prune"]["cont_method"] == "FTaylor":
                    main_attr_input = main_explainer.generate_FTaylor(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                    )
                    SA_attr_input = SA_explainer.generate_FTaylor(
                        input=inputs[i].unsqueeze(0),
                        index=SA_labels[i],
                    )
                elif config["prune"]["cont_method"] == "FTaylorpow2":
                    main_attr_input = main_explainer.generate_FTaylorpow2(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                    )
                    SA_attr_input = SA_explainer.generate_FTaylorpow2(
                        input=inputs[i].unsqueeze(0),
                        index=SA_labels[i],
                    )
                else:
                    _, main_attr_input = main_explainer.generate_LRP(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                        method=config["prune"]["cont_method"],
                    )
                    _, SA_attr_input = SA_explainer.generate_LRP(
                        input=inputs[i].unsqueeze(0),
                        index=SA_labels[i],
                        method=config["prune"]["cont_method"],
                    )

                main_attr_batch = main_attr_batch + main_attr_input.detach()
                SA_attr_batch = SA_attr_batch + SA_attr_input.detach()

                main_attr_input = None
                SA_attr_input = None

            # Averaging the block importances for the batch
            main_attr_batch = main_attr_batch / inputs.shape[0]
            SA_attr_batch = SA_attr_batch / inputs.shape[0]

            main_attr_iter = main_attr_iter + main_attr_batch
            SA_attr_iter = SA_attr_iter + SA_attr_batch

        if "attn" in required_matrices or "grad" in required_matrices:
            main_output = main_model(inputs)
            SA_output = SA_model(inputs)
            if "attn" in required_matrices:
                main_attn_batch = torch.stack(
                    [
                        main_model.blocks[block_idx]
                        .attn.get_attn_map()
                        .mean(dim=0)
                        .detach()
                        for block_idx in range(main_model.depth)
                    ],
                    dim=0,
                )
                SA_attn_batch = torch.stack(
                    [
                        SA_model.blocks[block_idx]
                        .attn.get_attn_map()
                        .mean(dim=0)
                        .detach()
                        for block_idx in range(SA_model.depth)
                    ],
                    dim=0,
                )

                main_attn_iter = main_attn_iter + main_attn_batch
                SA_attn_iter = SA_attn_iter + SA_attn_batch
            if "grad" in required_matrices:
                main_grad_batch = torch.stack(
                    [
                        main_model.blocks[block_idx]
                        .attn.get_attn_gradients()
                        .mean(dim=0)
                        .detach()
                        for block_idx in range(main_model.depth)
                    ],
                    dim=0,
                )
                SA_grad_batch = torch.stack(
                    [
                        SA_model.blocks[block_idx]
                        .attn.get_attn_gradients()
                        .mean(dim=0)
                        .detach()
                        for block_idx in range(SA_model.depth)
                    ],
                    dim=0,
                )
                main_grad_iter = main_grad_iter + main_grad_batch
                SA_grad_iter = SA_grad_iter + SA_grad_batch

    response = {}

    if "attr" in required_matrices:
        main_attr_iter = main_attr_iter / config["prune"]["num_batch_per_iter"]
        SA_attr_iter = SA_attr_iter / config["prune"]["num_batch_per_iter"]
        response["attr"] = (main_attr_iter, SA_attr_iter)
    if "attn" in required_matrices:
        main_attn_iter = main_attn_iter / config["prune"]["num_batch_per_iter"]
        SA_attn_iter = SA_attn_iter / config["prune"]["num_batch_per_iter"]
        response["attn"] = (main_attn_iter, SA_attn_iter)
    if "grad" in required_matrices:
        main_grad_iter = main_grad_iter / config["prune"]["num_batch_per_iter"]
        SA_grad_iter = SA_grad_iter / config["prune"]["num_batch_per_iter"]
        response["grad"] = (main_grad_iter, SA_grad_iter)

    return response


def generate_pruning_mask(main_attrs_final, SA_attrs_final, prun_iter_cnt, verbose=2):
    main_mask = []
    prun_mask = []

    num_blocks, num_heads, num_tokens = (
        main_attrs_final.shape[0],
        main_attrs_final.shape[1],
        main_attrs_final.shape[2],
    )

    for blk_idx in range(main_attrs_final.shape[0]):
        main_mask_blk = []
        prun_mask_blk = []
        for h in range(main_attrs_final.shape[1]):
            # Generating the main mask
            main_attrs_flt = main_attrs_final[blk_idx][h].flatten()

            threshold = torch.quantile(
                main_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
            )  # (this param)% of the most important main params will be retained

            main_mask_blk_head = (main_attrs_final[blk_idx][h] < threshold).float()

            # Generating the pruning mask from SA branch
            can_be_pruned = SA_attrs_final[blk_idx][h] * main_mask_blk_head
            can_be_pruned_flt = can_be_pruned.flatten()

            k = int(
                config["prune"]["pruning_rate"] * main_mask_blk_head.sum()
            )  # Pruning Pruning_rate% of the paramters allowed by the main branch to be pruned

            top_k_values, top_k_indices = torch.topk(can_be_pruned_flt, k)
            prun_mask_blk_head = torch.ones_like(can_be_pruned_flt)
            prun_mask_blk_head[top_k_indices] = 0

            prun_mask_blk_head = prun_mask_blk_head.reshape(
                (main_attrs_final.shape[2], main_attrs_final.shape[3])
            )
            main_mask_blk.append(main_mask_blk_head)
            prun_mask_blk.append(prun_mask_blk_head)

            if verbose == 2 and prun_iter_cnt == 0:
                print(
                    f"#params pruned in head {h+1}: {(num_tokens*num_tokens) - prun_mask_blk_head.sum()}/{(num_tokens*num_tokens)} \
                    | Rate: {((num_tokens*num_tokens) - prun_mask_blk_head.sum())/(num_tokens*num_tokens)}"
                )

        main_mask_blk = torch.stack(main_mask_blk, dim=0)
        prun_mask_blk = torch.stack(prun_mask_blk, dim=0)
        main_mask.append(main_mask_blk)
        prun_mask.append(prun_mask_blk)
        if verbose == 2 and prun_iter_cnt == 0:
            print(
                f"@@@ #params pruned in block {blk_idx+1}: {(num_tokens*num_tokens*num_heads) - prun_mask_blk.sum()}/{(num_tokens*num_tokens*num_heads)} \
                | Rate: {((num_tokens*num_tokens*num_heads) - prun_mask_blk.sum())/(num_tokens*num_tokens*num_heads)}"
            )

    main_mask = torch.stack(main_mask, dim=0)
    prun_mask = torch.stack(prun_mask, dim=0)

    # NEEDS FIXING: generalize it LATER
    if prun_mask.shape[0] < num_blocks:
        ones_tensor = torch.ones(
            (num_blocks - prun_mask.shape[0],) + prun_mask.shape[1:],
            dtype=prun_mask.dtype,
            device=prun_mask.device,
        )

        # Concatenate the original tensor with the ones tensor along the first dimension
        prun_mask = torch.cat([prun_mask, ones_tensor], dim=0)

    return main_mask, prun_mask


def generate_pruning_mask_block_agnostic(
    main_attrs_final, SA_attrs_final, prun_iter_cnt, verbose=2
):
    num_blocks, num_heads, num_tokens = (
        main_attrs_final.shape[0],
        main_attrs_final.shape[1],
        main_attrs_final.shape[2],
    )

    print("QQQ1")
    for b in range(num_blocks):
        for h in range(num_heads):
            print(f"Block {b+1}, Head {h+1}")
            print(get_stat(main_attrs_final[b][h]))
            print()
    print()

    print("QQQ2")
    for b in range(num_blocks):
        for h in range(num_heads):
            print(f"Block {b+1}, Head {h+1}")
            print(get_stat(SA_attrs_final[b][h]))
            print()
    print()

    # Generating the main mask
    main_attrs_flt = main_attrs_final.flatten()

    # (this param)% of the most important main params will be retained
    threshold = torch.quantile(
        main_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
    )

    main_mask = (main_attrs_flt < threshold).float()

    # Generating the pruning mask from SA branch
    SA_attrs_flt = SA_attrs_final.flatten()
    can_be_pruned = SA_attrs_flt * main_mask

    print("QQQ3")
    temp = can_be_pruned.reshape((num_blocks, num_heads, num_tokens, num_tokens))
    for b in range(num_blocks):
        for h in range(num_heads):
            print(f"Block {b+1}, Head {h+1}")
            print(get_stat(temp[b][h]))
            print()
    print()

    # Pruning Pruning_rate% of the paramters allowed by the main branch to be pruned
    k = int(config["prune"]["pruning_rate"] * main_mask.sum())

    top_k_values, top_k_indices = torch.topk(can_be_pruned, k)
    print("QQQ4")
    print(top_k_indices)
    print()
    prun_mask = torch.ones_like(can_be_pruned)
    prun_mask[top_k_indices] = 0

    main_mask = main_mask.reshape((num_blocks, num_heads, num_tokens, num_tokens))
    prun_mask = prun_mask.reshape((num_blocks, num_heads, num_tokens, num_tokens))

    # printing stats
    for b in range(num_blocks):
        for h in range(num_heads):
            prun_mask_blk_head = prun_mask[b][h]

            if verbose == 2:
                print(
                    f"#params pruned in head {h+1}: {(num_tokens*num_tokens) - prun_mask_blk_head.sum()}/{(num_tokens*num_tokens)} \
                    | Rate: {((num_tokens*num_tokens) - prun_mask_blk_head.sum())/(num_tokens*num_tokens)}"
                )

        prun_mask_blk = prun_mask[b]
        if verbose == 2:
            print(
                f"@@@ #params pruned in block {b+1}: {(num_tokens*num_tokens*num_heads) - prun_mask_blk.sum()}/{(num_tokens*num_tokens*num_heads)} \
                | Rate: {((num_tokens*num_tokens*num_heads) - prun_mask_blk.sum())/(num_tokens*num_tokens*num_heads)}"
            )

    return main_mask, prun_mask


def XTranPrune(
    main_model,
    SA_model,
    dataloader,
    device,
    config,
    prun_iter_cnt,
    verbose=2,
):
    required_matrices = ["attr"]

    matrices = get_required_matrices(
        main_model=main_model,
        SA_model=SA_model,
        dataloader=dataloader,
        device=device,
        config=config,
        required_matrices=required_matrices,
    )

    main_attrs, SA_attrs = matrices["attr"]

    torch.save(
        main_attrs,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"main_attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )
    torch.save(
        SA_attrs,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"SA_attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )

    main_attrs_final = main_attrs
    SA_attrs_final = SA_attrs

    ###############################  Generating the pruning mask ###############################

    if config["prune"]["BlockAgnosticMask"]:
        print("Generating the pruning mask (block-agnostic) ...")
        main_mask, prun_mask = generate_pruning_mask_block_agnostic(
            main_attrs_final, SA_attrs_final, prun_iter_cnt
        )
    else:
        print("Generating the pruning mask...")
        main_mask, prun_mask = generate_pruning_mask(
            main_attrs_final, SA_attrs_final, prun_iter_cnt
        )

    torch.save(
        main_mask,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"main_mask_Iter={prun_iter_cnt + 1}.pth",
        ),
    )
    torch.save(
        prun_mask,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"pruning_mask_Iter={prun_iter_cnt + 1}.pth",
        ),
    )

    prev_mask = main_model.get_attn_pruning_mask()

    main_model.set_attn_pruning_mask(prun_mask, config["prune"]["MaskUpdate_Type"])

    if prev_mask is not None:
        new_mask = main_model.get_attn_pruning_mask()
        num_total_nodes_pruned = (new_mask == 0).sum()
        num_new_nodes_pruned = ((new_mask == 0) & (prev_mask == 1)).sum()
        num_old_nodes_pruned = ((new_mask == 0) & (prev_mask == 0)).sum()

        num_total_nodes_unpruned = (new_mask == 1).sum()
        num_new_nodes_unpruned = ((new_mask == 1) & (prev_mask == 0)).sum()
        num_old_nodes_unpruned = ((new_mask == 1) & (prev_mask == 1)).sum()

        print()
        print("Pruning Statistics:")
        print(f"Total nodes pruned: {num_total_nodes_pruned.item()}")
        print(f"New nodes pruned: {num_new_nodes_pruned.item()}")
        print(f"Old nodes pruned: {num_old_nodes_pruned.item()}")

        print(f"Total nodes unpruned: {num_total_nodes_unpruned.item()}")
        print(f"New nodes unpruned: {num_new_nodes_unpruned.item()}")
        print(f"Old nodes unpruned: {num_old_nodes_unpruned.item()}")

        prune_stat = {
            "Total_pruned": int(num_total_nodes_pruned.item()),
            "New_pruned": int(num_new_nodes_pruned.item()),
            "Old_pruned": int(num_old_nodes_pruned.item()),
            "Total_unpruned": int(num_total_nodes_unpruned.item()),
            "New_unpruned": int(num_new_nodes_unpruned.item()),
            "Old_unpruned": int(num_old_nodes_unpruned.item()),
        }
    else:
        new_mask = main_model.get_attn_pruning_mask()
        num_total_nodes_pruned = (new_mask == 0).sum()
        num_total_nodes_unpruned = (new_mask == 1).sum()

        print()
        print("Pruning Statistics:")
        print(f"Total nodes pruned: {num_total_nodes_pruned.item()}")
        print(f"Total nodes unpruned: {num_total_nodes_unpruned.item()}")

        prune_stat = {
            "Total_pruned": int(num_total_nodes_pruned.item()),
            "New_pruned": None,
            "Old_pruned": None,
            "Total_unpruned": int(num_total_nodes_unpruned.item()),
            "New_unpruned": None,
            "Old_unpruned": None,
        }

    return main_model, prune_stat


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
    pprint(config)
    print()

    shutil.copy(
        args.config,
        os.path.join(config["output_folder_path"], "configs.yml"),
    )

    set_seeds(config["seed"])

    if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
        dataloaders, dataset_sizes, main_num_classes, SA_num_classes = get_dataloaders(
            root_image_dir=config["root_image_dir"],
            Generated_csv_path=config["Generated_csv_path"],
            sampler_type=config["prune"]["sampler_type"],
            dataset_name=config["dataset_name"],
            stratify_cols=config["prune"]["stratify_cols"],
            main_level=config["train"]["main_level"],
            SA_level=config["train"]["SA_level"],
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
                main_level=config["train"]["main_level"],
                SA_level=config["train"]["SA_level"],
                batch_size=config["train"]["batch_size"],
                num_workers=1,
            )
        )
    elif config["dataset_name"] in ["GF3300"]:
        dataloaders, dataset_sizes, main_num_classes, SA_num_classes = get_dataloaders(
            root_image_dir=config["root_image_dir"],
            train_csv_path=config["train_csv_path"],
            val_csv_path=config["val_csv_path"],
            sampler_type=config["prune"]["sampler_type"],
            dataset_name=config["dataset_name"],
            main_level=config["train"]["main_level"],
            SA_level=config["train"]["SA_level"],
            batch_size=config["prune"]["batch_size"],
            num_workers=1,
        )
        val_dataloaders, val_dataset_sizes, val_main_num_classes, val_SA_num_classes = (
            get_dataloaders(
                root_image_dir=config["root_image_dir"],
                train_csv_path=config["train_csv_path"],
                val_csv_path=config["val_csv_path"],
                sampler_type="WeightedRandom",
                dataset_name=config["dataset_name"],
                main_level=config["train"]["main_level"],
                SA_level=config["train"]["SA_level"],
                batch_size=config["train"]["batch_size"],
                num_workers=1,
            )
        )
    else:
        raise ValueError("Invalid dataset name.")

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

    val_metrics, _ = eval_model(
        main_model,
        val_dataloaders,
        val_dataset_sizes,
        val_main_num_classes,
        device,
        config["train"]["main_level"],
        "DeiT_S_LRP_PIter0",
        config,
        save_preds=True,
    )

    val_metrics_df = pd.DataFrame([val_metrics])
    prune_stat_df = pd.DataFrame(
        columns=[
            "Total_pruned",
            "New_pruned",
            "Old_pruned",
            "Total_unpruned",
            "New_unpruned",
            "Old_unpruned",
        ]
    )
    print("Validation metrics using the original model:")
    pprint(val_metrics)
    print()

    prun_iter_cnt = 0
    consecutive_no_improvement = 0
    best_bias_metric = val_metrics[config["prune"]["target_bias_metric"]]
    print(
        f"Baseline selected bias metric: {config['prune']['target_bias_metric']} = {best_bias_metric}"
    )

    while (
        consecutive_no_improvement <= config["prune"]["max_consecutive_no_improvement"]
    ):
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt+1} +++++++++++++++++++++++++++++"
        )

        model_name = f"DeiT_S_LRP_PIter{prun_iter_cnt+1}"

        if prun_iter_cnt == 0:
            pruned_model, prune_stat = XTranPrune(
                main_model=main_model,
                SA_model=SA_model,
                dataloader=dataloaders["train"],
                device=device,
                verbose=config["prune"]["verbose"],
                config=config,
                prun_iter_cnt=prun_iter_cnt,
            )
        else:
            pruned_model, prune_stat = XTranPrune(
                main_model=pruned_model,
                SA_model=SA_model,
                dataloader=dataloaders["train"],
                device=device,
                verbose=config["prune"]["verbose"],
                config=config,
                prun_iter_cnt=prun_iter_cnt,
            )

        val_metrics, _ = eval_model(
            pruned_model,
            val_dataloaders,
            val_dataset_sizes,
            val_main_num_classes,
            device,
            config["train"]["main_level"],
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

        prune_stat_df = pd.concat(
            [prune_stat_df, pd.DataFrame([prune_stat])], ignore_index=True
        )
        prune_stat_df.to_csv(
            os.path.join(config["output_folder_path"], f"Pruning_Statistics.csv"),
            index=False,
        )

        plot_metrics(
            prune_stat_df, ["Total_pruned", "New_pruned", "Old_pruned"], "PS", config
        )

        if (val_metrics_df.iloc[0]["F1_Mac"] - val_metrics_df.iloc[-1]["F1_Mac"]) > 4:
            print(
                "F1_Mac decreased by more than 6% from the initial value, Stopping..."
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)
    main(config, args)
