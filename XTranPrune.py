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

from Utils.Misc_utils import set_seeds, Logger, get_stat, get_mask_idx
from Utils.Metrics import plot_metrics
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Explainability.ViT_Explainer import Explainer
from Evaluation import eval_model


def get_hooked_matrices(main_model, SA_model, dataloader, blk_mat_type, device, config):

    DL_iter = iter(dataloader)

    num_tokens = main_model.patch_embed.num_patches + 1
    blk_mat_shape = (
        main_model.depth,
        main_model.num_heads,
        num_tokens,
        num_tokens,
    )

    main_blk_mat_iter = torch.zeros(blk_mat_shape).to(device)
    SA_blk_mat_iter = torch.zeros(blk_mat_shape).to(device)

    for itr in tqdm(
        range(config["prune"]["num_batch_per_iter"]),
        total=config["prune"]["num_batch_per_iter"],
        desc="Getting hooked matrices",
    ):

        try:
            batch = next(DL_iter)
        except StopIteration:
            DL_iter = iter(dataloader)
            batch = next(DL_iter)

        inputs = batch["image"].to(device)
        main_output = main_model(inputs)
        SA_output = SA_model(inputs)

        if blk_mat_type == "attention_weights":
            main_blk_mat_batch = torch.stack(
                [
                    main_model.blocks[block_idx]
                    .attn.get_attn_map()
                    .mean(dim=0)
                    .detach()
                    for block_idx in range(main_model.depth)
                ],
                dim=0,
            )
            SA_blk_mat_batch = torch.stack(
                [
                    SA_model.blocks[block_idx].attn.get_attn_map().mean(dim=0).detach()
                    for block_idx in range(SA_model.depth)
                ],
                dim=0,
            )

        elif blk_mat_type == "gradients":
            main_blk_mat_batch = torch.stack(
                [
                    main_model.blocks[block_idx]
                    .attn.get_attn_gradients()
                    .mean(dim=0)
                    .detach()
                    for block_idx in range(main_model.depth)
                ],
                dim=0,
            )
            SA_blk_mat_batch = torch.stack(
                [
                    SA_model.blocks[block_idx]
                    .attn.get_attn_gradients()
                    .mean(dim=0)
                    .detach()
                    for block_idx in range(SA_model.depth)
                ],
                dim=0,
            )
        else:
            raise ValueError("Invalid block matrix type.")

        main_blk_mat_iter = main_blk_mat_iter + main_blk_mat_batch
        SA_blk_mat_iter = SA_blk_mat_iter + SA_blk_mat_batch
        main_blk_mat_batch = None
        SA_blk_mat_batch = None

    main_blk_mat_iter = main_blk_mat_iter / config["prune"]["num_batch_per_iter"]
    SA_blk_mat_iter = SA_blk_mat_iter / config["prune"]["num_batch_per_iter"]

    return main_blk_mat_iter.cpu(), SA_blk_mat_iter.cpu()


def get_attr_score(main_explainer, SA_explainer, dataloader, device, config):
    """
    Calculate the attribute scores for the main and SA levels.

    Args:
        main_explainer (Explainer): The explainer for the main level.
        SA_explainer (Explainer): The explainer for the SA level.
        blk_attrs_shape (tuple): The shape of the block attributes.
        dataloader (DataLoader): The data loader for the dataset.
        device (torch.device): The device to perform computations on.
        config (dict): The configuration parameters.

    Returns:
        tuple: A tuple containing the attribute scores for the main and SA levels.
    """
    DL_iter = iter(dataloader)

    num_tokens = main_explainer.model.patch_embed.num_patches + 1
    blk_attrs_shape = (
        main_explainer.model.depth,
        main_explainer.model.num_heads,
        num_tokens,
        num_tokens,
    )
    main_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)
    SA_blk_attrs_iter = torch.zeros(blk_attrs_shape).to(device)

    for itr in tqdm(
        range(config["prune"]["num_batch_per_iter"]),
        total=config["prune"]["num_batch_per_iter"],
        desc="Generating node attributions",
    ):

        try:
            batch = next(DL_iter)
        except StopIteration:
            DL_iter = iter(dataloader)
            batch = next(DL_iter)

        inputs = batch["image"].to(device)
        main_labels = batch[config["train"]["main_level"]].to(device)
        SA_labels = batch[config["train"]["SA_level"]].to(device)

        main_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)
        SA_blk_attrs_batch = torch.zeros(blk_attrs_shape).to(device)

        for i in range(inputs.shape[0]):  # iterate over batch size
            if config["prune"]["cont_method"] == "attn":
                main_blk_attrs_input = main_explainer.generate_attn(
                    input=inputs[i].unsqueeze(0)
                )
                SA_blk_attrs_input = SA_explainer.generate_attn(
                    input=inputs[i].unsqueeze(0)
                )
            elif config["prune"]["cont_method"] == "TranInter":
                _, main_blk_attrs_input = main_explainer.generate_TranInter(
                    input=inputs[i].unsqueeze(0),
                    index=main_labels[i],
                )
                _, SA_blk_attrs_input = SA_explainer.generate_TranInter(
                    input=inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                )

            elif config["prune"]["cont_method"] == "TAM":

                _, main_blk_attrs_input = main_explainer.generate_TAM(
                    input=inputs[i].unsqueeze(0),
                    index=main_labels[i],
                    start_layer=0,
                    steps=10,
                )
                _, SA_blk_attrs_input = SA_explainer.generate_TAM(
                    input=inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                    start_layer=0,
                    steps=10,
                )
                main_blk_attrs_input = main_blk_attrs_input.squeeze(0)
                SA_blk_attrs_input = SA_blk_attrs_input.squeeze(0)
            elif config["prune"]["cont_method"] == "AttrRoll":

                _, main_blk_attrs_input = main_explainer.generate_AttrRoll(
                    input=inputs[i].unsqueeze(0), index=main_labels[i]
                )
                _, SA_blk_attrs_input = SA_explainer.generate_AttrRoll(
                    input=inputs[i].unsqueeze(0), index=SA_labels[i]
                )
            elif config["prune"]["cont_method"] == "FTaylor":
                main_blk_attrs_input = main_explainer.generate_FTaylor(
                    input=inputs[i].unsqueeze(0),
                    index=main_labels[i],
                )
                SA_blk_attrs_input = SA_explainer.generate_FTaylor(
                    input=inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                )
            elif config["prune"]["cont_method"] == "FTaylorpow2":
                main_blk_attrs_input = main_explainer.generate_FTaylorpow2(
                    input=inputs[i].unsqueeze(0),
                    index=main_labels[i],
                )
                SA_blk_attrs_input = SA_explainer.generate_FTaylorpow2(
                    input=inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                )
            else:
                _, main_blk_attrs_input = main_explainer.generate_LRP(
                    input=inputs[i].unsqueeze(0),
                    index=main_labels[i],
                    method=config["prune"]["cont_method"],
                )
                _, SA_blk_attrs_input = SA_explainer.generate_LRP(
                    input=inputs[i].unsqueeze(0),
                    index=SA_labels[i],
                    method=config["prune"]["cont_method"],
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

    return main_blk_attrs_iter, SA_blk_attrs_iter


def generate_pruning_mask(
    main_blk_attrs_iter_final, SA_blk_attrs_iter_final, prun_iter_cnt, verbose=2
):
    main_mask = []
    prun_mask = []

    num_blocks, num_heads, num_tokens = (
        main_blk_attrs_iter_final.shape[0],
        main_blk_attrs_iter_final.shape[1],
        main_blk_attrs_iter_final.shape[2],
    )

    for blk_idx in range(main_blk_attrs_iter_final.shape[0]):
        main_mask_blk = []
        prun_mask_blk = []
        for h in range(main_blk_attrs_iter_final.shape[1]):
            # Generating the main mask
            main_attrs_flt = main_blk_attrs_iter_final[blk_idx][h].flatten()

            threshold = torch.quantile(
                main_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
            )  # (this param)% of the most important main params will be retained

            main_mask_blk_head = (
                main_blk_attrs_iter_final[blk_idx][h] < threshold
            ).float()

            # Generating the pruning mask from SA branch
            can_be_pruned = SA_blk_attrs_iter_final[blk_idx][h] * main_mask_blk_head
            can_be_pruned_flt = can_be_pruned.flatten()

            k = int(
                config["prune"]["pruning_rate"] * main_mask_blk_head.sum()
            )  # Pruning Pruning_rate% of the paramters allowed by the main branch to be pruned

            top_k_values, top_k_indices = torch.topk(can_be_pruned_flt, k)
            prun_mask_blk_head = torch.ones_like(can_be_pruned_flt)
            prun_mask_blk_head[top_k_indices] = 0

            prun_mask_blk_head = prun_mask_blk_head.reshape(
                (main_blk_attrs_iter_final.shape[2], main_blk_attrs_iter_final.shape[3])
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


def run_pagerank(node_attr, attention_weights, model, config):

    def aggregate_attr_scores(attr, method="combined"):

        if attr.ndimension() != 4 or attr.size(2) != attr.size(3):
            raise ValueError(
                "attr must have shape (num_blocks, num_heads, num_tokens, num_tokens)"
            )

        if method == "outflow":
            # Sum across rows to get outflow scores
            token_attr_scores = torch.sum(attr, dim=-1)
        elif method == "inflow":
            # Sum across columns to get inflow scores
            token_attr_scores = torch.sum(attr, dim=-2)
        elif method == "combined":
            # Sum across both rows and columns for combined scores
            outflow_token_attr_scores = torch.sum(attr, dim=-1)
            inflow_token_attr_scores = torch.sum(attr, dim=-2)
            token_attr_scores = outflow_token_attr_scores + inflow_token_attr_scores
        else:
            raise ValueError(
                "Invalid method. Choose 'outflow', 'inflow', or 'combined'."
            )

        return token_attr_scores

    token_attr = aggregate_attr_scores(node_attr, method=config["prune"]["aggr_type"])

    if token_attr.ndimension() != 3 or token_attr.size(2) != 197:
        raise ValueError(
            "token_attr must have shape (num_blocks, num_heads, num_tokens) and num_tokens must be 197"
        )

    num_blocks, num_heads, num_tokens = token_attr.size()

    # Initialize the personalization vector for each node in the graph
    initial_values = {}

    for block_idx in range(num_blocks):
        for head_idx in range(num_heads):
            attribution_vector = token_attr[block_idx, head_idx, :]

            # Normalize the subgraph attribution scores to sum to 1
            total_sum = torch.sum(attribution_vector).item()
            if total_sum == 0:
                raise ValueError(
                    "Sum of attribution scores in a subgraph cannot be zero."
                )

            normalized_vector = attribution_vector / total_sum

            # Assign normalized values to the corresponding nodes
            subgraph_nodes = [
                tuple([block_idx, head_idx, token_idx])
                for token_idx in range(num_tokens)
            ]
            for node, attr_value in zip(subgraph_nodes, normalized_vector):
                initial_values[node] = attr_value.item()

    assert list(model.graph.nodes) == list(initial_values.keys()), "Nodes mismatch"

    # Run PageRank with the initial vector
    pagerank_scores = nx.pagerank(
        model.get_graph(),
        alpha=config["prune"]["PR_alpha"],
        nstart=initial_values,
        max_iter=5000,
        weight="weight",
    )

    pagerank_tensor = torch.zeros((num_blocks, num_heads, num_tokens))

    # Populate the tensor with the PageRank scores
    for node, score in pagerank_scores.items():

        block_idx, head_idx, token_idx = node
        pagerank_tensor[block_idx, head_idx, token_idx] = score

    def reverse_aggregate_attr_scores(token_attr, attention_weights, method="combined"):
        num_blocks, num_heads, num_tokens = token_attr.shape

        if attention_weights.ndimension() != 4 or attention_weights.size(
            2
        ) != attention_weights.size(3):
            raise ValueError(
                "attention_weights must have shape (num_blocks, num_heads, num_tokens, num_tokens)"
            )

        # Initialize node attributions tensor
        node_attr_scores = torch.zeros(
            num_blocks,
            num_heads,
            num_tokens,
            num_tokens,
            dtype=token_attr.dtype,
            device=token_attr.device,
        )

        if method == "outflow":
            # Use attention weights to distribute token attribution scores across rows
            for i in range(num_tokens):
                node_attr_scores[:, :, i, :] = (
                    token_attr[:, :, i].unsqueeze(-1) * attention_weights[:, :, i, :]
                )
        elif method == "inflow":
            # Use attention weights to distribute token attribution scores across columns
            for j in range(num_tokens):
                node_attr_scores[:, :, :, j] = (
                    token_attr[:, :, j].unsqueeze(-1) * attention_weights[:, :, :, j]
                )
        elif method == "combined":
            # Distribute based on a combination of attention weights for inflow and outflow
            for i in range(num_tokens):
                for j in range(num_tokens):
                    outflow_contrib = (
                        token_attr[:, :, i] * attention_weights[:, :, i, j]
                    )
                    inflow_contrib = token_attr[:, :, j] * attention_weights[:, :, i, j]
                    node_attr_scores[:, :, i, j] = (
                        outflow_contrib + inflow_contrib
                    ) / attention_weights[:, :, i, j].sum(-1, keepdim=True).clamp(
                        min=1e-10
                    )
        else:
            raise ValueError(
                "Invalid method. Choose 'outflow', 'inflow', or 'combined'."
            )

        return node_attr

    node_attr = reverse_aggregate_attr_scores(
        token_attr=pagerank_tensor,
        attention_weights=attention_weights,
        method=config["prune"]["aggr_type"],
    )

    return node_attr


def XTranPrune(
    main_model,
    SA_model,
    dataloader,
    device,
    config,
    prun_iter_cnt,
    verbose=2,
    MA_vectors=None,
):

    main_explainer = Explainer(main_model)
    SA_explainer = Explainer(SA_model)

    main_blk_attrs_iter, SA_blk_attrs_iter = get_attr_score(
        main_explainer=main_explainer,
        SA_explainer=SA_explainer,
        dataloader=dataloader,
        device=device,
        config=config,
    )

    if config["prune"]["apply_pagerank"]:

        main_edge_weights, SA_edge_weights = get_hooked_matrices(
            main_model=main_model,
            SA_model=SA_model,
            dataloader=dataloader,
            blk_mat_type=config["prune"]["edge_type"],
            device=device,
            config=config,
        )

        main_model.build_graph(
            edge_weights=main_edge_weights.numpy(),
            branch="main",
            edge_type=config["prune"]["edge_type"],
            prun_iter_cnt=prun_iter_cnt + 1,
            # output_folder_path=config["output_folder_path"],
        )
        SA_model.build_graph(
            edge_weights=SA_edge_weights.numpy(),
            branch="SA",
            edge_type=config["prune"]["edge_type"],
            prun_iter_cnt=prun_iter_cnt + 1,
            # output_folder_path=config["output_folder_path"],
        )

        print("Graphs built successfully. Running PageRank...")

        main_blk_attrs_iter = run_pagerank(
            node_attr=main_blk_attrs_iter,
            attention_weights=main_edge_weights,
            model=main_model,
            config=config,
        )
        SA_blk_attrs_iter = run_pagerank(
            node_attr=SA_blk_attrs_iter,
            attention_weights=SA_edge_weights,
            model=SA_model,
            config=config,
        )

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

    print("Generating the pruning mask...")
    main_mask, prun_mask = generate_pruning_mask(
        main_blk_attrs_iter_final, SA_blk_attrs_iter_final, prun_iter_cnt
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

    if verbose > 0 and prev_mask is not None:
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

    print("Validation metrics using the original model:")
    pprint(val_metrics)
    print()

    prun_iter_cnt = 0
    consecutive_no_improvement = 0
    best_bias_metric = config["prune"]["bias_metric_prev"]

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
                dataloader=dataloaders["train"],
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
                dataloader=dataloaders["train"],
                device=device,
                verbose=config["prune"]["verbose"],
                config=config,
                MA_vectors=MA_vectors,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)
    main(config, args)
