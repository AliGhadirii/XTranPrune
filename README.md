# XTranPrune: eXplainability-Aware Transformer Pruning for Bias Mitigation in Dermatological Disease Classification

This repository contains code for training and evaluating models on the Fitzpatrick17k skin lesion dataset. The project includes data loading, model training, evaluation, and explainability modules.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pruning](#pruning)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AliGhadirii/XTranPrune.git
    cd XTranPrune
    ```

2. Create a conda environment:
    ```sh
    conda create --name XTranPrune_env python=3.12
    conda activate XTranPrune_env
    ```

3. Install PyTorch:
    ```sh
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    For alternative installation commands that suit your device, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

4. Install the required packages:
    ```sh
    pip install -r Requirements.txt
    ```

### Project Structure
Fitzpatrick17k/
├── Configs/
│   ├── [configs_Fitz.yml](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FConfigs%2Fconfigs_Fitz.yml%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
│   ├── configs_PAD.yml
│   └── Dataset-Prepration.ipynb
├── Datasets/
│   ├── dataloaders.py
│   └── [datasets.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FDatasets%2Fdatasets.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
├── Explainability/
│   └── ViT_Explainer.py
├── Models/
│   ├── [helpers.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FModels%2Fhelpers.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
│   ├── layer_helpers.py
│   ├── layers_ours.py
│   ├── ViT_LRP.py
│   └── weight_init.py
├── Utils/
│   ├── [Metrics.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FUtils%2FMetrics.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
│   ├── Misc_utils.py
│   └── transformers_utils.py
├── [XTranPrune.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FXTranPrune.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
├── [Evaluation.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FEvaluation.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
├── [Train_DeiT-S.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FTrain_DeiT-S.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
├── [README.md](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FREADME.md%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
└── [Requirements.txt](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FRequirements.txt%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)

## Usage

### Configuration

Configuration files are located in the `Configs/` directory. You can modify the parameters in these YAML files to suit your needs. Here is an example from `configs_Fitz.yml`:

Before running any scripts, you need to generate the CSV files for the datasets. Run the `Dataset-Prepration.ipynb` notebook to preprocess the metadata and generate the required CSV files.

Parameter Descriptions
- `seed`: Random seed for reproducibility.
- `root_image_dir`: Directory containing the images.
- `Generated_csv_path`: Path to the preprocessed metadata CSV file.
- `dataset_name`: Name of the dataset.
- `output_folder_path`: Directory to save outputs.
- `num_workers`: Number of workers for data loading.
- `train`: Training parameters.
  - `branch`: Branch of the model to train.
  - `batch_size`: Batch size for training.
  - `main_level`: Main level of classification.
  - `SA_level`: Sub-level of classification.
  - `n_epochs`: Number of training epochs.
  - `pretrained`: Whether to use a pretrained model.
- `eval`: Evaluation parameters.
  - `weight_path`: Path to the model weights for evaluation.
- `prune`: Pruning parameters.
  - `main_br_path`: Path to the main branch model weights.
  - `SA_br_path`: Path to the sub-branch model weights.
  - `stratify_cols`: Columns to stratify by.
  - `sampler_type`: Type of sampler to use.
  - `target_bias_metric`: Bias metric to target.
  - `max_consecutive_no_improvement`: Maximum consecutive iterations without improvement.
  - `batch_size`: Batch size for pruning.
  - `num_batch_per_iter`: Number of batches per iteration.
  - `main_mask_retain_rate`: Retain rate for the main mask.
  - `pruning_rate`: Pruning rate.
  - `beta1`: Beta1 parameter for pruning.
  - `beta2`: Beta2 parameter for pruning.
  - `cont_method`: Contrastive method.
  - `method`: Pruning method.
  - `FObjective`: Objective function for pruning.

### Training

To train a model, run the `Train_DeiT-S.py` script with the desired configuration file:

```sh
python [Train_DeiT-S.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FTrain_DeiT-S.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D) --config [configs_Fitz.yml](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FConfigs%2Fconfigs_Fitz.yml%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
```

### Pruning
To prune a model, use the XTranPrune.py script:
```sh
python [XTranPrune.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FXTranPrune.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D) --config [configs_Fitz.yml](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FConfigs%2Fconfigs_Fitz.yml%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
```


### Evaluation
To evaluate a model, use the Evaluation.py script:
```sh
python [Evaluation.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FEvaluation.py%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D) --config [configs_Fitz.yml](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fali%2FRepos%2FXTranPrune%2FConfigs%2Fconfigs_Fitz.yml%22%2C%22scheme%22%3A%22vscode-remote%22%2C%22authority%22%3A%22ssh-remote%2B7b22686f73744e616d65223a22475055536572766572227d%22%7D%7D)
```


### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

### License
This project is licensed under the MIT License.