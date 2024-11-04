"""
### nnUNet Inference Runner

Provides functionalities related to running the nnUNet inference on the input data.
"""


import subprocess
from tqdm import tqdm
import os
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional


def maybe_make_dir(path: str) -> str:
    """
    Creates a directory at the specified path if it does not exist.

    Args:
        path (str): Path to the directory to be created.

    Returns:
        str: The path to the created or existing directory.
    """
    os.makedirs(path, exist_ok=True)
    return path


def set_env_paths(input_path: str, path: str) -> str:
    """
    Sets environment paths required for nnUNet predictions.

    Args:
        input_path (str): Path to the input data.
        path (str): Path to the output directory.

    Returns:
        str: A string containing environment variables for nnUNet.
    """
    # Define the path for preprocessed data
    preprocessed = maybe_make_dir(str(Path('/tmp/') / 'nnUNet_preprocessed'))
    
    # Return environment variable settings as a single string
    return f"nnUNet_raw='{input_path}' nnUNet_preprocessed='{preprocessed}' nnUNet_results='/tmp/'"


def get_dataset_info(challenge_name: str) -> Tuple[str, str, str, str]:
    """
    Returns the corresponding dataset name, trainer, plans, and configuration based on the provided challenge name.

    Args:
        challenge_name (str): Name of the challenge (e.g., BraTS-PED, BraTS-MET).

    Returns:
        Tuple[str, str, str, str]: 
            - dataset_name: Name of the dataset.
            - trainer: Trainer configuration.
            - plans: Plans configuration.
            - configuration: nnUNet configuration.

    Raises:
        Exception: If the challenge name is not compatible.
    """
    if challenge_name == "BraTS-PED":
        dataset_name = "Dataset021_BraTS2024-PED"
        trainer = "nnUNetTrainer_200epochs"
        plans = "nnUNetPlans"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-SSA":
        dataset_name = "Dataset022_BraTS2024-SSA"
        trainer = "nnUNetTrainer_150epochs_CustomSplit_stratified_pretrained_GLI"
        plans = "nnUNetPlans_pretrained_GLI"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-GLI":
        dataset_name = "Dataset023_BraTS2024-GLI"
        trainer = "nnUNetTrainer_200epochs"
        plans = "nnUNetPlans"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-MEN-RT":
        dataset_name = "Dataset024_BraTS2024-MEN-RT"
        trainer = "nnUNetTrainer_200epochs"
        plans = "nnUNetPlans"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-MET":
        dataset_name = "Dataset026_BraTS2024-MET"
        trainer = "nnUNetTrainer_200epochs"
        plans = "nnUNetPlans"
        configuration = "3d_fullres"
    else:
        raise Exception("Challenge name not compatible.")
    
    return dataset_name, trainer, plans, configuration


def run_infer_nnunet(
    input_folder: str,
    output_folder: str,
    challenge_name: str,
    name: str,
    folds: List[int] = [0, 1, 2, 3, 4],
    save_npz: bool = True,
    ensemble: bool = True
) -> List[Path]:
    """
    Runs nnUNet inference on the provided input data.

    This function executes the nnUNet inference command either as an ensemble
    across all folds or individually per specified fold. It extracts the
    predictions and saves them as .npz files.

    Args:
        input_folder (str): Path to the input folder containing prediction files.
        output_folder (str): Path to the folder where output predictions will be stored.
        challenge_name (str): Name of the challenge (e.g., BraTS-PED, BraTS-MET).
        name (str): Base name for the output prediction files.
        folds (List[int], optional): List of fold indices to process. Defaults to [0, 1, 2, 3, 4].
        save_npz (bool, optional): Whether to save the prediction probabilities as .npz files. Defaults to True.
        ensemble (bool, optional): Whether to perform ensemble prediction across all folds. Defaults to True.

    Returns:
        List[Path]: A list of paths to the saved .npz files from each fold.
    """
    # Retrieve dataset information based on the challenge name
    dataset_name, trainer, plans, configuration = get_dataset_info(challenge_name)
    
    # Set environment paths required for nnUNet
    env_set = set_env_paths(input_folder, output_folder)

    if ensemble:
        # Define the output directory for ensemble predictions
        output_folder_fold = os.path.join(output_folder, "ens")
        print(f"Running nnUNet inference with all folds (ensemble)..")
        
        # Construct the nnUNet inference command for ensemble
        cmd = (
            f"{env_set} nnUNetv2_predict "
            f"-i '{input_folder}' "
            f"-o '{output_folder_fold}' "
            f"-d '{dataset_name}' "
            f"-c '{configuration}' "
            f"-tr '{trainer}' "
            f"-p '{plans}'"
        )
        
        if save_npz:
            cmd += " --save_probabilities"
        
        # Execute the inference command
        subprocess.run(cmd, shell=True)
        
        # Return the path to the ensemble .npz file
        return [Path(os.path.join(output_folder_fold, f"{name}.npz"))]
    else:
        # Initialize a list to store paths to .npz files for each fold
        npz_path_list: List[Path] = [] 
        
        # Iterate over each specified fold and perform inference
        for fold in tqdm(folds, desc="Processing Folds"):
            output_folder_fold = os.path.join(output_folder, f"fold_{fold}")
            print(f"Running nnUNet inference for fold {fold}")
            
            # Construct the nnUNet inference command for the current fold
            cmd = (
                f"{env_set} nnUNetv2_predict "
                f"-i '{input_folder}' "
                f"-o '{output_folder_fold}' "
                f"-d '{dataset_name}' "
                f"-c '{configuration}' "
                f"-tr '{trainer}' "
                f"-p '{plans}' "
                f"-f '{fold}'"
            )
            
            if save_npz:
                cmd += " --save_probabilities"
            
            # Execute the inference command for the current fold
            subprocess.run(cmd, shell=True)  # Executes the command in the shell
            
            # Append the path to the saved .npz file
            npz_path_list.append(Path(os.path.join(output_folder_fold, f"{name}.npz")))
    
        return npz_path_list
