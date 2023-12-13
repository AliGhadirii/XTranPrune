import os
import yaml
import subprocess

# Define your list of hyperparameters
hyperparameters_list = [
    {"learning_rate": 0.001, "batch_size": 32},
    {"learning_rate": 0.01, "batch_size": 64},
    # Add more hyperparameter combinations as needed
]

# Path to your YAML file
yaml_file_path = "Configs/configs_server.yml"

# Define the command you want to run
command_to_run = "python DisTranPrune.py --config Configs/configs_server.yml"

folder_index = 13

# Loop over hyperparameters
for main_thresh in [0.9, 0.7]:
    for prun_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # Create the output folder if it doesn't exist
        os.makedirs(
            f"/home/ali/Outputs/Pruning/PruningEXP{folder_index}/", exist_ok=True
        )

        # Load the YAML file
        with open(yaml_file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            yaml_data[
                "output_folder_path"
            ] = f"/home/ali/Outputs/Pruning/PruningEXP{folder_index}/"
            yaml_data["prune"]["main_mask_retain_rate"] = main_thresh
            yaml_data["prune"]["pruning_rate"] = prun_thresh

        # Save the modified YAML data back to the file
        with open(yaml_file_path, "w") as file:
            yaml.safe_dump(yaml_data, file)

        # Execute the command
        print(
            f"Running experiment {folder_index} with hyperparameters: {main_thresh}, {prun_thresh}"
        )
        subprocess.run(
            command_to_run.split(), check=True, cwd="/home/ali/Repos/SkinFormer/"
        )  # Run command in the iteration folder
        folder_index += 1

print("All experiments completed.")
