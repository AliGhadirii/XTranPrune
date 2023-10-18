import argparse
import yaml
import time
import os
from tqdm import tqdm

import torch

from Utils.Misc_utils import set_seeds
from Datasets.Fitz17k_dataset import get_fitz17k_dataloaders
from Models.ViT_LRP.ViT_LRP import deit_small_patch16_224
from Utils.XAI_utils import show_explanation_sample


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    dataloaders, dataset_sizes, num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        binary_subgroup=config["prune"]["binary_subgroup"],
        holdout_set="random_holdout",
        batch_size=config["prune"]["batch_size"],
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

    show_explanation_sample(main_model, dataloaders["train"])

    # metric = nn.CrossEntropyLoss()
    # # best_bias_metric = val_metrics[config['FairPrune']["target_bias_metric"]]
    # best_bias_metric = 0.6394507842223532
    # prun_iter_cnt = 0
    # no_improvement_cnt = 0
    # consecutive_no_improvement = 0

    # while no_improvement_cnt < config["FairPrune"]["max_consecutive_no_improvement"]:
    #     since = time.time()

    #     print(
    #         f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt} +++++++++++++++++++++++++++++"
    #     )
    #     model_pruned = fairprune(
    #         model=model,
    #         metric=metric,
    #         device=device,
    #         config=config,
    #         verbose=1,
    #     )

    #     dataloaders, dataset_sizes, num_classes = get_fitz17k_dataloaders(
    #         root_image_dir=config["root_image_dir"],
    #         Generated_csv_path=config["Generated_csv_path"],
    #         level=config["default"]["level"],
    #         binary_subgroup=config["default"]["binary_subgroup"],
    #         holdout_set="random_holdout",
    #         batch_size=config["default"]["batch_size"],
    #         num_workers=1,
    #     )

    #     val_metrics, df_preds = eval_model(
    #         model_pruned,
    #         dataloaders,
    #         dataset_sizes,
    #         device,
    #         config["default"]["level"],
    #         "FairPrune",
    #         config,
    #         save_preds=False,
    #     )

    #     if val_metrics[config["FairPrune"]["target_bias_metric"]] > best_bias_metric:
    #         best_bias_metric = val_metrics[config["FairPrune"]["target_bias_metric"]]

    #         # Save the df_preds

    #         df_preds.to_csv(
    #             os.path.join(
    #                 config["output_folder_path"],
    #                 f"validation_results_Resnet18_FairPrune_Iter={prun_iter_cnt}.csv",
    #             ),
    #             index=False,
    #         )

    #         # Save the best model
    #         print("New leading model val metrics \n")
    #         print(val_metrics)

    #         best_model_path = os.path.join(
    #             config["output_folder_path"],
    #             f"Resnet18_checkpoint_FairPrune_Iter={prun_iter_cnt}.pth",
    #         )
    #         checkpoint = {
    #             "leading_val_metrics": val_metrics,
    #             "model_state_dict": model.state_dict(),
    #         }
    #         torch.save(checkpoint, best_model_path)
    #         print("Checkpoint saved:", best_model_path)

    #         # Reset the counter
    #         consecutive_no_improvement = 0
    #     else:
    #         print(
    #             "Bias Metric is: {}, No improvement.\n".format(
    #                 val_metrics[config["FairPrune"]["target_bias_metric"]]
    #             )
    #         )
    #         df_preds.to_csv(
    #             os.path.join(
    #                 config["output_folder_path"],
    #                 f"validation_results_Resnet18_FairPrune_Iter={prun_iter_cnt}_temp.csv",
    #             ),
    #             index=False,
    #         )
    #         consecutive_no_improvement += 1

    #     prun_iter_cnt += 1

    #     time_elapsed = time.time() - since
    #     print(
    #         "This iteration took {:.0f}m {:.0f}s".format(
    #             time_elapsed // 60, time_elapsed % 60
    #         )
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
