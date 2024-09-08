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
    get_stat,
    get_mask_idx,
    js_divergence,
    preprocess_matrix,
)
from Utils.Metrics import plot_metrics
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Explainability.ViT_Explainer import Explainer
from Evaluation import eval_model


def get_required_matrices(
    model, dataloader, device, config, required_matrices=["attr", "attn"]
):

    assert all(
        param in ["attr", "attn", "grad"] for param in required_matrices
    ), "Not all provided elements are in the required_matrices list."

    DL_iter = iter(dataloader)
    num_tokens = model.patch_embed.num_patches + 1
    matrix_shape = (
        model.depth,
        model.num_heads,
        num_tokens,
        num_tokens,
    )

    if "attr" in required_matrices:
        explainer = Explainer(model)

        main_attr_iter = torch.zeros(matrix_shape).to(device)
        S0_attr_iter = torch.zeros(matrix_shape).to(device)
        S1_attr_iter = torch.zeros(matrix_shape).to(device)
    if "attn" in required_matrices:
        main_attn_iter = torch.zeros(matrix_shape).to(device)
        S0_attn_iter = torch.zeros(matrix_shape).to(device)
        S1_attn_iter = torch.zeros(matrix_shape).to(device)
    if "grad" in required_matrices:
        main_grad_iter = torch.zeros(matrix_shape).to(device)
        S0_grad_iter = torch.zeros(matrix_shape).to(device)
        S1_grad_iter = torch.zeros(matrix_shape).to(device)

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

        S0_inputs = inputs[SA_labels == 0]
        S0_labels = main_labels[SA_labels == 0]

        S1_inputs = inputs[SA_labels == 1]
        S1_labels = main_labels[SA_labels == 1]

        if "attr" in required_matrices:
            main_attr_batch = torch.zeros(matrix_shape).to(device)
            for i in range(config["prune"]["batch_size"]):
                if config["prune"]["cont_method"] == "attn":
                    main_attr_input = explainer.generate_attn(
                        input=inputs[i].unsqueeze(0)
                    )
                elif config["prune"]["cont_method"] == "TranInter":
                    _, main_attr_input = explainer.generate_TranInter(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                    )
                elif config["prune"]["cont_method"] == "TAM":
                    _, main_attr_input = explainer.generate_TAM(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                        start_layer=0,
                        steps=10,
                    )
                    main_attr_input = main_attr_input.squeeze(0)
                elif config["prune"]["cont_method"] == "AttrRoll":
                    _, main_attr_input = explainer.generate_AttrRoll(
                        input=inputs[i].unsqueeze(0), index=main_labels[i]
                    )
                elif config["prune"]["cont_method"] == "FTaylor":
                    main_attr_input = explainer.generate_FTaylor(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                    )
                elif config["prune"]["cont_method"] == "FTaylorpow2":
                    main_attr_input = explainer.generate_FTaylorpow2(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                    )
                else:
                    _, main_attr_input = explainer.generate_LRP(
                        input=inputs[i].unsqueeze(0),
                        index=main_labels[i],
                        method=config["prune"]["cont_method"],
                    )

                main_attr_batch = main_attr_batch + main_attr_input.detach()
                main_attr_input = None

            S0_attr_batch = torch.zeros(matrix_shape).to(device)
            S1_attr_batch = torch.zeros(matrix_shape).to(device)
            SA_batch_size = config["prune"]["batch_size"] // 2

            for i in range(SA_batch_size):  # iterate over SA batch
                if config["prune"]["cont_method"] == "attn":
                    S0_attr_input = explainer.generate_attn(
                        input=S0_inputs[i].unsqueeze(0)
                    )
                    S1_attr_input = explainer.generate_attn(
                        input=S1_inputs[i].unsqueeze(0)
                    )
                elif config["prune"]["cont_method"] == "TranInter":
                    _, S0_attr_input = explainer.generate_TranInter(
                        input=S0_inputs[i].unsqueeze(0),
                        index=S0_labels[i],
                    )
                    _, S1_attr_input = explainer.generate_TranInter(
                        input=S1_inputs[i].unsqueeze(0),
                        index=S1_labels[i],
                    )
                elif config["prune"]["cont_method"] == "TAM":
                    _, S0_attr_input = explainer.generate_TAM(
                        input=S0_inputs[i].unsqueeze(0),
                        index=S0_labels[i],
                        start_layer=0,
                        steps=10,
                    )
                    _, S1_attr_input = explainer.generate_TAM(
                        input=S1_inputs[i].unsqueeze(0),
                        index=S1_labels[i],
                        start_layer=0,
                        steps=10,
                    )
                    S0_attr_input = S0_attr_input.squeeze(0)
                    S1_attr_input = S1_attr_input.squeeze(0)
                elif config["prune"]["cont_method"] == "AttrRoll":
                    _, S0_attr_input = explainer.generate_AttrRoll(
                        input=S0_inputs[i].unsqueeze(0), index=S0_labels[i]
                    )
                    _, S1_attr_input = explainer.generate_AttrRoll(
                        input=S1_inputs[i].unsqueeze(0), index=S1_labels[i]
                    )
                elif config["prune"]["cont_method"] == "FTaylor":
                    S0_attr_input = explainer.generate_FTaylor(
                        input=S0_inputs[i].unsqueeze(0),
                        index=S0_labels[i],
                    )
                    S1_attr_input = explainer.generate_FTaylor(
                        input=S1_inputs[i].unsqueeze(0),
                        index=S1_labels[i],
                    )
                elif config["prune"]["cont_method"] == "FTaylorpow2":
                    S0_attr_input = explainer.generate_FTaylorpow2(
                        input=S0_inputs[i].unsqueeze(0),
                        index=S0_labels[i],
                    )
                    S1_attr_input = explainer.generate_FTaylorpow2(
                        input=S1_inputs[i].unsqueeze(0),
                        index=S1_labels[i],
                    )
                else:
                    _, S0_attr_input = explainer.generate_LRP(
                        input=S0_inputs[i].unsqueeze(0),
                        index=S0_labels[i],
                        method=config["prune"]["cont_method"],
                    )
                    _, S1_attr_input = explainer.generate_LRP(
                        input=S1_inputs[i].unsqueeze(0),
                        index=S1_labels[i],
                        method=config["prune"]["cont_method"],
                    )

                S0_attr_batch = S0_attr_batch + S0_attr_input.detach()
                S1_attr_batch = S1_attr_batch + S1_attr_input.detach()

                S0_attr_input = None
                S1_attr_input = None

            # Averaging the block importances for the batch
            main_attr_batch = main_attr_batch / inputs.shape[0]
            S0_attr_batch = S0_attr_batch / S0_inputs.shape[0]
            S1_attr_batch = S1_attr_batch / S1_inputs.shape[0]

            main_attr_iter = main_attr_iter + main_attr_batch
            S0_attr_iter = S0_attr_iter + S0_attr_batch
            S1_attr_iter = S1_attr_iter + S1_attr_batch

        if "attn" in required_matrices or "grad" in required_matrices:
            main_output = model(inputs)
            if "attn" in required_matrices:
                main_attn_batch = torch.stack(
                    [
                        model.blocks[block_idx].attn.get_attn_map().mean(dim=0).detach()
                        for block_idx in range(model.depth)
                    ],
                    dim=0,
                )
                main_attn_iter = main_attn_iter + main_attn_batch
            if "grad" in required_matrices:
                main_grad_batch = torch.stack(
                    [
                        model.blocks[block_idx]
                        .attn.get_attn_gradients()
                        .mean(dim=0)
                        .detach()
                        for block_idx in range(model.depth)
                    ],
                    dim=0,
                )
                main_grad_iter = main_grad_iter + main_grad_batch

            S0_output = model(S0_inputs)
            if "attn" in required_matrices:
                S0_attn_batch = torch.stack(
                    [
                        model.blocks[block_idx].attn.get_attn_map().mean(dim=0).detach()
                        for block_idx in range(model.depth)
                    ],
                    dim=0,
                )
                S0_attn_iter = S0_attn_iter + S0_attn_batch

            if "grad" in required_matrices:
                S0_grad_batch = torch.stack(
                    [
                        model.blocks[block_idx]
                        .attn.get_attn_gradients()
                        .mean(dim=0)
                        .detach()
                        for block_idx in range(model.depth)
                    ],
                    dim=0,
                )
                S0_grad_iter = S0_grad_iter + S0_grad_batch

            S1_output = model(S1_inputs)
            if "attn" in required_matrices:
                S1_attn_batch = torch.stack(
                    [
                        model.blocks[block_idx].attn.get_attn_map().mean(dim=0).detach()
                        for block_idx in range(model.depth)
                    ],
                    dim=0,
                )
                S1_attn_iter = S1_attn_iter + S1_attn_batch
            if "grad" in required_matrices:
                S1_grad_batch = torch.stack(
                    [
                        model.blocks[block_idx]
                        .attn.get_attn_gradients()
                        .mean(dim=0)
                        .detach()
                        for block_idx in range(model.depth)
                    ],
                    dim=0,
                )
                S1_grad_iter = S1_grad_iter + S1_grad_batch

    response = {}

    if "attr" in required_matrices:
        main_attr_iter = main_attr_iter / config["prune"]["num_batch_per_iter"]
        S0_attr_iter = S0_attr_iter / config["prune"]["num_batch_per_iter"]
        S1_attr_iter = S1_attr_iter / config["prune"]["num_batch_per_iter"]
        response["attr"] = (main_attr_iter, S0_attr_iter, S1_attr_iter)
    if "attn" in required_matrices:
        main_attn_iter = main_attn_iter / config["prune"]["num_batch_per_iter"]
        S0_attn_iter = S0_attn_iter / config["prune"]["num_batch_per_iter"]
        S1_attn_iter = S1_attn_iter / config["prune"]["num_batch_per_iter"]
        response["attn"] = (main_attn_iter, S0_attn_iter, S1_attn_iter)
    if "grad" in required_matrices:
        main_grad_iter = main_grad_iter / config["prune"]["num_batch_per_iter"]
        S0_grad_iter = S0_grad_iter / config["prune"]["num_batch_per_iter"]
        S1_grad_iter = S1_grad_iter / config["prune"]["num_batch_per_iter"]
        response["grad"] = (main_grad_iter, S0_grad_iter, S1_grad_iter)

    return response


def run_pagerank(node_attr, attention_weights, model, config):

    node_attr_copy = node_attr

    def aggregate_attr_scores(attr, method="combined"):
        if attr.ndimension() != 4 or attr.size(2) != attr.size(3):
            raise ValueError(
                "attr must have shape (num_blocks, num_heads, num_tokens, num_tokens)"
            )
        assert method in [
            "outflow",
            "inflow",
            "combined",
        ], "Invalid method. Choose 'outflow', 'inflow', or 'combined'."

        if method == "outflow":
            # Sum across rows to get outflow scores
            row_sums = torch.sum(attr, dim=-1, keepdim=True)
            ratios = attr / row_sums.clamp(min=1e-10)  # Avoid division by zero
            token_attr_scores = row_sums.squeeze(-1)
            return token_attr_scores, ratios

        elif method == "inflow":
            # Sum across columns to get inflow scores
            column_sums = torch.sum(attr, dim=-2, keepdim=True)
            ratios = attr / column_sums.clamp(min=1e-10)  # Avoid division by zero
            token_attr_scores = column_sums.squeeze(-2)
            return token_attr_scores, ratios

        elif method == "combined":
            # Sum across both rows and columns for combined scores
            outflow_token_attr_scores = torch.sum(attr, dim=-1)
            inflow_token_attr_scores = torch.sum(attr, dim=-2)
            combined_ratios_outflow = attr / torch.sum(
                attr, dim=-1, keepdim=True
            ).clamp(min=1e-10)
            combined_ratios_inflow = attr / torch.sum(attr, dim=-2, keepdim=True).clamp(
                min=1e-10
            )
            token_attr_scores = outflow_token_attr_scores + inflow_token_attr_scores
            return token_attr_scores, combined_ratios_outflow, combined_ratios_inflow

    token_attr, ratios = aggregate_attr_scores(
        node_attr, method=config["prune"]["aggr_type"]
    )

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
        max_iter=int(1e07),
        tol=1e-06,
        weight="weight",
    )

    pagerank_tensor = torch.zeros((num_blocks, num_heads, num_tokens)).to(
        node_attr.device
    )

    # Populate the tensor with the PageRank scores
    for node, score in pagerank_scores.items():
        block_idx, head_idx, token_idx = node
        pagerank_tensor[block_idx, head_idx, token_idx] = score

    def reverse_aggregate_attr_scores(
        token_attr, attention_weights=None, ratios=None, method="combined"
    ):
        num_blocks, num_heads, num_tokens = token_attr.shape

        if attention_weights is None and ratios is None:
            raise ValueError("Either attention_weights or ratios must be provided.")

        assert method in [
            "outflow",
            "inflow",
            "combined",
        ], "Invalid method. Choose 'outflow', 'inflow', or 'combined'."

        node_attr_scores = torch.zeros(
            num_blocks,
            num_heads,
            num_tokens,
            num_tokens,
            dtype=token_attr.dtype,
            device=token_attr.device,
        )

        if ratios is not None:
            if method == "outflow":
                # Use the stored ratios to distribute token attribution scores across rows
                for i in range(num_tokens):
                    node_attr_scores[:, :, i, :] = (
                        token_attr[:, :, i].unsqueeze(-1) * ratios[:, :, i, :]
                    )

            elif method == "inflow":
                # Use the stored ratios to distribute token attribution scores across columns
                for j in range(num_tokens):
                    node_attr_scores[:, :, :, j] = (
                        token_attr[:, :, j].unsqueeze(-1) * ratios[:, :, :, j]
                    )

            elif method == "combined":
                combined_ratios_outflow, combined_ratios_inflow = ratios
                for i in range(num_tokens):
                    for j in range(num_tokens):
                        outflow_contrib = (
                            token_attr[:, :, i] * combined_ratios_outflow[:, :, i, j]
                        )
                        inflow_contrib = (
                            token_attr[:, :, j] * combined_ratios_inflow[:, :, i, j]
                        )
                        node_attr_scores[:, :, i, j] = outflow_contrib + inflow_contrib

        else:  # Use attention_weights
            if method == "outflow":
                # Use attention weights to distribute token attribution scores across rows
                for i in range(num_tokens):
                    node_attr_scores[:, :, i, :] = (
                        token_attr[:, :, i].unsqueeze(-1)
                        * attention_weights[:, :, i, :]
                    )
            elif method == "inflow":
                # Use attention weights to distribute token attribution scores across columns
                for j in range(num_tokens):
                    node_attr_scores[:, :, :, j] = (
                        token_attr[:, :, j].unsqueeze(-1)
                        * attention_weights[:, :, :, j]
                    )
            elif method == "combined":
                # Distribute based on a combination of attention weights for inflow and outflow
                for i in range(num_tokens):
                    for j in range(num_tokens):
                        # Normalize outflow and inflow contributions before combining
                        outflow_contrib = outflow_contrib / attention_weights[
                            :, :, i, :
                        ].sum(-1, keepdim=True).clamp(min=1e-10)
                        inflow_contrib = inflow_contrib / attention_weights[
                            :, :, :, j
                        ].sum(-2, keepdim=True).clamp(min=1e-10)

                        node_attr_scores[:, :, i, j] = outflow_contrib + inflow_contrib

        return node_attr_scores

    if config["prune"]["redistribution_type"] == "attention_weights":
        node_attr = reverse_aggregate_attr_scores(
            token_attr=pagerank_tensor,
            attention_weights=attention_weights,
            method=config["prune"]["aggr_type"],
        )
    elif config["prune"]["redistribution_type"] == "original_ratio":
        node_attr = reverse_aggregate_attr_scores(
            token_attr=pagerank_tensor,
            ratios=ratios,
            method=config["prune"]["aggr_type"],
        )
    else:
        raise ValueError("Invalid redistribution type.")

    print(
        f"pagerank input == pagerank output: {torch.equal(node_attr, node_attr_copy)}"
    )

    return node_attr


def run_pagerank_BlockWise(node_attr, attention_weights, model, config):

    node_attr_copy = node_attr

    def aggregate_attr_scores(attr, method="combined"):
        if attr.ndimension() != 4 or attr.size(2) != attr.size(3):
            raise ValueError(
                "attr must have shape (num_blocks, num_heads, num_tokens, num_tokens)"
            )
        assert method in [
            "outflow",
            "inflow",
            "combined",
        ], "Invalid method. Choose 'outflow', 'inflow', or 'combined'."

        if method == "outflow":
            # Sum across rows to get outflow scores
            row_sums = torch.sum(attr, dim=-1, keepdim=True)
            ratios = attr / row_sums.clamp(min=1e-10)  # Avoid division by zero
            token_attr_scores = row_sums.squeeze(-1)
            return token_attr_scores, ratios

        elif method == "inflow":
            # Sum across columns to get inflow scores
            column_sums = torch.sum(attr, dim=-2, keepdim=True)
            ratios = attr / column_sums.clamp(min=1e-10)  # Avoid division by zero
            token_attr_scores = column_sums.squeeze(-2)
            return token_attr_scores, ratios

        elif method == "combined":
            # Sum across both rows and columns for combined scores
            outflow_token_attr_scores = torch.sum(attr, dim=-1)
            inflow_token_attr_scores = torch.sum(attr, dim=-2)
            combined_ratios_outflow = attr / torch.sum(
                attr, dim=-1, keepdim=True
            ).clamp(min=1e-10)
            combined_ratios_inflow = attr / torch.sum(attr, dim=-2, keepdim=True).clamp(
                min=1e-10
            )
            token_attr_scores = outflow_token_attr_scores + inflow_token_attr_scores
            return token_attr_scores, combined_ratios_outflow, combined_ratios_inflow

    token_attr, ratios = aggregate_attr_scores(
        node_attr, method=config["prune"]["aggr_type"]
    )

    if token_attr.ndimension() != 3 or token_attr.size(2) != 197:
        raise ValueError(
            "token_attr must have shape (num_blocks, num_heads, num_tokens) and num_tokens must be 197"
        )

    num_blocks, num_heads, num_tokens = token_attr.size()

    pagerank_tensor = torch.zeros((num_blocks, num_heads, num_tokens)).to(
        node_attr.device
    )

    for block_idx in range(num_blocks):
        for head_idx in range(num_heads):
            attribution_vector = token_attr[block_idx, head_idx, :]

            total_sum = torch.sum(attribution_vector).item()
            if total_sum == 0:
                raise ValueError(
                    "Sum of attribution scores in a subgraph cannot be zero."
                )
            normalized_vector = attribution_vector / total_sum

            subgraph_nodes = [
                tuple([block_idx, head_idx, token_idx])
                for token_idx in range(num_tokens)
            ]
            initial_values = {
                node: normalized_vector[i].item()
                for i, node in enumerate(subgraph_nodes)
            }

            subgraph = model.graph.subgraph(subgraph_nodes)

            assert sorted(list(subgraph.nodes)) == sorted(
                list(initial_values.keys())
            ), "Nodes mismatch"

            pagerank_scores = nx.pagerank(
                subgraph,
                alpha=config["prune"]["PR_alpha"],
                nstart=initial_values,
                max_iter=int(1e07),
                tol=1e-06,
                weight="weight",
            )

            for node, score in pagerank_scores.items():
                block_idx, head_idx, token_idx = node
                pagerank_tensor[block_idx, head_idx, token_idx] = score

    def reverse_aggregate_attr_scores(
        token_attr, attention_weights=None, ratios=None, method="combined"
    ):
        num_blocks, num_heads, num_tokens = token_attr.shape

        if attention_weights is None and ratios is None:
            raise ValueError("Either attention_weights or ratios must be provided.")

        assert method in [
            "outflow",
            "inflow",
            "combined",
        ], "Invalid method. Choose 'outflow', 'inflow', or 'combined'."

        node_attr_scores = torch.zeros(
            num_blocks,
            num_heads,
            num_tokens,
            num_tokens,
            dtype=token_attr.dtype,
            device=token_attr.device,
        )

        if ratios is not None:
            if method == "outflow":
                # Use the stored ratios to distribute token attribution scores across rows
                for i in range(num_tokens):
                    node_attr_scores[:, :, i, :] = (
                        token_attr[:, :, i].unsqueeze(-1) * ratios[:, :, i, :]
                    )

            elif method == "inflow":
                # Use the stored ratios to distribute token attribution scores across columns
                for j in range(num_tokens):
                    node_attr_scores[:, :, :, j] = (
                        token_attr[:, :, j].unsqueeze(-1) * ratios[:, :, :, j]
                    )

            elif method == "combined":
                combined_ratios_outflow, combined_ratios_inflow = ratios
                for i in range(num_tokens):
                    for j in range(num_tokens):
                        outflow_contrib = (
                            token_attr[:, :, i] * combined_ratios_outflow[:, :, i, j]
                        )
                        inflow_contrib = (
                            token_attr[:, :, j] * combined_ratios_inflow[:, :, i, j]
                        )
                        node_attr_scores[:, :, i, j] = outflow_contrib + inflow_contrib

        else:  # Use attention_weights
            if method == "outflow":
                # Use attention weights to distribute token attribution scores across rows
                for i in range(num_tokens):
                    node_attr_scores[:, :, i, :] = (
                        token_attr[:, :, i].unsqueeze(-1)
                        * attention_weights[:, :, i, :]
                    )
            elif method == "inflow":
                # Use attention weights to distribute token attribution scores across columns
                for j in range(num_tokens):
                    node_attr_scores[:, :, :, j] = (
                        token_attr[:, :, j].unsqueeze(-1)
                        * attention_weights[:, :, :, j]
                    )
            elif method == "combined":
                # Distribute based on a combination of attention weights for inflow and outflow
                for i in range(num_tokens):
                    for j in range(num_tokens):
                        # Normalize outflow and inflow contributions before combining
                        outflow_contrib = outflow_contrib / attention_weights[
                            :, :, i, :
                        ].sum(-1, keepdim=True).clamp(min=1e-10)
                        inflow_contrib = inflow_contrib / attention_weights[
                            :, :, :, j
                        ].sum(-2, keepdim=True).clamp(min=1e-10)

                        node_attr_scores[:, :, i, j] = outflow_contrib + inflow_contrib

        return node_attr_scores

    if config["prune"]["redistribution_type"] == "attention_weights":
        node_attr = reverse_aggregate_attr_scores(
            token_attr=pagerank_tensor,
            attention_weights=attention_weights,
            method=config["prune"]["aggr_type"],
        )
    elif config["prune"]["redistribution_type"] == "original_ratio":
        node_attr = reverse_aggregate_attr_scores(
            token_attr=pagerank_tensor,
            ratios=ratios,
            method=config["prune"]["aggr_type"],
        )
    else:
        raise ValueError("Invalid redistribution type.")

    print(
        f"pagerank input == pagerank output: {torch.equal(node_attr, node_attr_copy)}"
    )

    return node_attr


def generate_pruning_mask(main_attrs_final, disc_main_attrs_final, verbose=2):
    performance_mask = []
    prun_mask = []

    num_blocks, num_heads, num_tokens = (
        main_attrs_final.shape[0],
        main_attrs_final.shape[1],
        main_attrs_final.shape[2],
    )

    for blk_idx in range(main_attrs_final.shape[0]):
        performance_mask_blk = []
        prun_mask_blk = []
        for h in range(main_attrs_final.shape[1]):

            blk_attrs_flt = main_attrs_final[blk_idx][h].flatten()

            # (this param)% of the most important main params will be retained
            threshold = torch.quantile(
                blk_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
            )

            performance_mask_blk_head = (
                main_attrs_final[blk_idx][h] < threshold
            ).float()

            # Generating the pruning mask from SA branch
            score = disc_main_attrs_final[blk_idx][h] * performance_mask_blk_head
            score_flt = score.flatten()

            # Pruning Pruning_rate% of the paramters
            k = int(config["prune"]["pruning_rate"] * performance_mask_blk_head.sum())

            top_k_values, top_k_indices = torch.topk(score_flt, k)
            prun_mask_blk_head = torch.ones_like(score_flt)
            prun_mask_blk_head[top_k_indices] = 0

            prun_mask_blk_head = prun_mask_blk_head.reshape(
                (main_attrs_final.shape[2], main_attrs_final.shape[3])
            )

            performance_mask_blk.append(performance_mask_blk_head)
            prun_mask_blk.append(prun_mask_blk_head)

            if verbose == 2:
                print(
                    f"#params pruned in head {h+1}: {(num_tokens*num_tokens) - prun_mask_blk_head.sum()}/{(num_tokens*num_tokens)} \
                    | Rate: {((num_tokens*num_tokens) - prun_mask_blk_head.sum())/(num_tokens*num_tokens)}"
                )

        performance_mask_blk = torch.stack(performance_mask_blk, dim=0)
        prun_mask_blk = torch.stack(prun_mask_blk, dim=0)

        performance_mask.append(performance_mask_blk)
        prun_mask.append(prun_mask_blk)
        if verbose == 2:
            print(
                f"@@@ #params pruned in block {blk_idx+1}: {(num_tokens*num_tokens*num_heads) - prun_mask_blk.sum()}/{(num_tokens*num_tokens*num_heads)} \
                | Rate: {((num_tokens*num_tokens*num_heads) - prun_mask_blk.sum())/(num_tokens*num_tokens*num_heads)}"
            )

    performance_mask = torch.stack(performance_mask, dim=0)
    prun_mask = torch.stack(prun_mask, dim=0)

    return performance_mask, prun_mask


def generate_pruning_mask_block_agnostic(main_attrs_final, SA_attrs_final, verbose=2):
    num_blocks, num_heads, num_tokens = (
        main_attrs_final.shape[0],
        main_attrs_final.shape[1],
        main_attrs_final.shape[2],
    )

    # Generating the main mask
    main_attrs_flt = main_attrs_final.flatten()

    # (this param)% of the most important main params will be retained
    threshold = torch.quantile(
        main_attrs_flt, 1 - config["prune"]["main_mask_retain_rate"]
    )

    performance_mask = (main_attrs_flt < threshold).float()

    # Generating the pruning mask from SA branch
    SA_attrs_flt = SA_attrs_final.flatten()
    can_be_pruned = SA_attrs_flt * performance_mask

    # Pruning Pruning_rate% of the paramters allowed by the main branch to be pruned
    k = int(config["prune"]["pruning_rate"] * performance_mask.sum())

    top_k_values, top_k_indices = torch.topk(can_be_pruned, k)
    prun_mask = torch.ones_like(can_be_pruned)
    prun_mask[top_k_indices] = 0

    performance_mask = performance_mask.reshape(
        (num_blocks, num_heads, num_tokens, num_tokens)
    )
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

    return performance_mask, prun_mask


def Contrastive(
    model,
    dataloader,
    device,
    config,
    prun_iter_cnt,
    verbose=2,
    MA_vectors=None,
):
    required_matrices = ["attr"]

    if config["prune"]["apply_pagerank"]:
        if config["prune"]["edge_type"] == "attention_weights":
            required_matrices.append("attn")
        elif config["prune"]["edge_type"] == "gradients":
            required_matrices.append("grad")
        else:
            raise ValueError("Invalid edge type.")

    matrices = get_required_matrices(
        model=model,
        dataloader=dataloader,
        device=device,
        config=config,
        required_matrices=required_matrices,
    )

    main_attrs, S0_attrs, S1_attrs = matrices["attr"]

    if config["prune"]["apply_pagerank"]:

        if config["prune"]["edge_type"] == "attention_weights":
            edge_weights, _, _ = matrices["attn"]
        elif config["prune"]["edge_type"] == "gradients":
            edge_weights, _, _ = matrices["grad"]

        print("EDGE WEIGHTS: Before preprocessing ...")
        for i in range(edge_weights.shape[0]):
            for j in range(edge_weights.shape[1]):
                print(get_stat(edge_weights[i][j]))

        print("Preprocessing the edge weights...")
        edge_weights = preprocess_matrix(
            edge_weights,
            clip_threshold=1e-6,
            log_transform=True,
            normalize=True,
            min_max_scale=True,
        )

        print("EDGE WEIGHTS: After preprocessing ...")
        for i in range(edge_weights.shape[0]):
            for j in range(edge_weights.shape[1]):
                print(get_stat(edge_weights[i][j]))
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )

        print("Building the graphs...")
        model.build_graph(
            edge_weights=edge_weights.cpu().numpy(),
            branch="main",
            edge_type=config["prune"]["edge_type"],
            prun_iter_cnt=prun_iter_cnt + 1,
            # output_folder_path=config["output_folder_path"],
        )

        print("Graphs built successfully. Running PageRank...")

        # print("NODE PAGERANK SCORES (ATTR): Before preprocessing ...")

        # for i in range(main_attrs.shape[0]):
        #     for j in range(main_attrs.shape[1]):
        #         print(get_stat(main_attrs[i][j]))

        # main_attrs = preprocess_matrix(
        #     main_attrs,
        #     clip_threshold=1e-6,
        #     log_transform=True,
        #     normalize=True,
        #     min_max_scale=True,
        # )
        # S0_attrs = preprocess_matrix(
        #     S0_attrs,
        #     clip_threshold=1e-6,
        #     log_transform=True,
        #     normalize=True,
        #     min_max_scale=True,
        # )
        # S1_attrs = preprocess_matrix(
        #     S1_attrs,
        #     clip_threshold=1e-6,
        #     log_transform=True,
        #     normalize=True,
        #     min_max_scale=True,
        # )

        print("NODE PAGERANK SCORES (ATTR): Before Pagerank ...")
        for i in range(main_attrs.shape[0]):
            for j in range(main_attrs.shape[1]):
                print(get_stat(main_attrs[i][j]))

        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )

        main_attrs = run_pagerank(
            node_attr=main_attrs,
            attention_weights=edge_weights,
            model=model,
            config=config,
        )
        S0_attrs = run_pagerank(
            node_attr=S0_attrs,
            attention_weights=edge_weights,
            model=model,
            config=config,
        )
        S1_attrs = run_pagerank(
            node_attr=S1_attrs,
            attention_weights=edge_weights,
            model=model,
            config=config,
        )

        print("NODE PAGERANK SCORES (ATTR): After running Pagerank ...")
        for i in range(main_attrs.shape[0]):
            for j in range(main_attrs.shape[1]):
                print(get_stat(main_attrs[i][j]))

    torch.save(
        main_attrs,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"attr_Iter={prun_iter_cnt + 1}.pth",
        ),
    )
    torch.save(
        S0_attrs,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"S0_attrs={prun_iter_cnt + 1}.pth",
        ),
    )
    torch.save(
        S1_attrs,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"S1_attrs={prun_iter_cnt + 1}.pth",
        ),
    )

    if config["prune"]["FObjective"] == "MinJSD_Diff":
        S0_attrs_normalized = S0_attrs / S0_attrs.sum(dim=(-2, -1), keepdim=True)
        S1_attrs_normalized = S1_attrs / S1_attrs.sum(dim=(-2, -1), keepdim=True)
        jsd = torch.ones(S0_attrs_normalized.shape[0], S0_attrs_normalized.shape[1]).to(
            device
        )
        for encoder_idx in range(jsd.shape[0]):
            for head_idx in range(jsd.shape[1]):

                # Get the attention masks for the current head and encoder
                P = S0_attrs_normalized[encoder_idx, head_idx, :, :]
                Q = S1_attrs_normalized[encoder_idx, head_idx, :, :]

                # Calculate JSD and store in the result tensor
                jsd_value = js_divergence(P, Q)
                jsd[encoder_idx, head_idx] = jsd_value

        disc_main_attrs = jsd.unsqueeze(-1).unsqueeze(-1) * (S0_attrs - S1_attrs).abs()

    elif config["prune"]["FObjective"] == "MinDiff":
        disc_main_attrs = (S0_attrs - S1_attrs).abs()
    else:
        raise ValueError("Invalid FObjective")

    # getting the moving average of attribution vectors and measuing the uncertainty
    if config["prune"]["method"] == "MA":

        blk_attrs_MA = MA_vectors[0]
        disc_blk_attrs_MA = MA_vectors[1]

        if blk_attrs_MA == None:
            blk_attrs_MA = torch.zeros_like(main_attrs)
        if disc_blk_attrs_MA == None:
            disc_blk_attrs_MA = torch.zeros_like(disc_main_attrs)

        beta1 = config["prune"]["beta1"]
        blk_attrs_MA = beta1 * blk_attrs_MA + (1 - beta1) * main_attrs
        disc_blk_attrs_MA = beta1 * disc_blk_attrs_MA + (1 - beta1) * disc_main_attrs

        MA_vectors[0] = blk_attrs_MA
        MA_vectors[1] = disc_blk_attrs_MA

        main_attrs_final = blk_attrs_MA
        disc_main_attrs_final = disc_blk_attrs_MA

    elif config["prune"]["method"] == "MA_Uncertainty":

        blk_attrs_MA = MA_vectors[0]
        disc_blk_attrs_MA = MA_vectors[1]
        blk_Uncer_MA = MA_vectors[2]
        disc_blk_Uncer_MA = MA_vectors[3]

        if blk_attrs_MA == None:
            blk_attrs_MA = torch.zeros_like(main_attrs)
        if disc_blk_attrs_MA == None:
            disc_blk_attrs_MA = torch.zeros_like(disc_main_attrs)
        if blk_Uncer_MA == None:
            blk_Uncer_MA = torch.zeros_like(main_attrs)
        if disc_blk_Uncer_MA == None:
            disc_blk_Uncer_MA = torch.zeros_like(disc_main_attrs)

        beta1 = config["prune"]["beta1"]
        blk_attrs_MA = beta1 * blk_attrs_MA + (1 - beta1) * main_attrs
        disc_blk_attrs_MA = beta1 * disc_blk_attrs_MA + (1 - beta1) * disc_main_attrs

        beta2 = config["prune"]["beta2"]
        blk_Uncer_MA = (
            beta2 * blk_Uncer_MA + (1 - beta2) * (main_attrs - blk_attrs_MA).abs()
        )
        disc_blk_Uncer_MA = (
            beta2 * disc_blk_Uncer_MA
            + (1 - beta2) * (disc_main_attrs - disc_blk_Uncer_MA).abs()
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
        main_attrs_final = blk_attrs_MA * blk_Uncer_MA
        disc_main_attrs_final = disc_blk_attrs_MA * (
            disc_blk_Uncer_MA_max_values - disc_blk_Uncer_MA
        )
    else:
        main_attrs_final = main_attrs
        disc_main_attrs_final = disc_main_attrs

    ###############################  Generating the pruning mask ###############################

    if config["prune"]["BlockAgnosticMask"]:
        print("Generating the pruning mask (block-agnostic) ...")
        performance_mask, prun_mask = generate_pruning_mask_block_agnostic(
            main_attrs_final, disc_main_attrs_final
        )
    else:
        print("Generating the pruning mask...")
        performance_mask, prun_mask = generate_pruning_mask(
            main_attrs_final, disc_main_attrs_final
        )

    torch.save(
        performance_mask,
        os.path.join(
            config["output_folder_path"],
            "Log_files",
            f"performance_mask_Iter={prun_iter_cnt + 1}.pth",
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

    prev_mask = model.get_attn_pruning_mask()

    model.set_attn_pruning_mask(prun_mask, config["prune"]["MaskUpdate_Type"])

    if prev_mask is not None:
        new_mask = model.get_attn_pruning_mask()
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
        new_mask = model.get_attn_pruning_mask()
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

    return model, MA_vectors, prune_stat


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
        raise ValueError("Invalid dataset name")

    # load the model
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
            pruned_model, MA_vectors, prune_stat = Contrastive(
                model=model,
                dataloader=dataloaders["train"],
                device=device,
                config=config,
                prun_iter_cnt=prun_iter_cnt,
                verbose=config["prune"]["verbose"],
                MA_vectors=MA_vectors,
            )
        else:
            pruned_model, MA_vectors, prune_stat = Contrastive(
                model=pruned_model,
                dataloader=dataloaders["train"],
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

        if (val_metrics_df.iloc[0]["F1_Mac"] - val_metrics_df.iloc[-1]["F1_Mac"]) > 6:
            print(
                "F1_Mac decreased by more than 6% from the initial value, Stopping..."
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config, args)
