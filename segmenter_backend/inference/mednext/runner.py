"""
### MedNeXt Inference Runner

Provides functionalities related to running the MedNeXt inference on the input data.
"""


import subprocess
from tqdm import tqdm
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def maybe_make_dir(path: str) -> Path:
    """
    Creates a directory at the specified path if it does not exist.

    Args:
        path (str): Path to the directory to be created.

    Returns:
        Path: The path to the created or existing directory.
    """
    os.makedirs(path, exist_ok=True)
    return Path(path)


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
        dataset_name = "Task021_BraTS2024-PEDs"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans = "nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-SSA":
        dataset_name = "Task022_BraTS2024-SSA"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_150epochs_StratifiedSplit"
        plans = "nnUNetPlans_pretrained_SSA"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-GLI":
        dataset_name = "Task023_BraTS2024-GLI"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans = "nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-MEN-RT":
        dataset_name = "Task024_BraTS2024-MEN-RT"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans = "nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    elif challenge_name == "BraTS-MET":
        dataset_name = "Task026_BraTS2024-MET"
        trainer = "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit"
        plans = "nnUNetPlansv2.1_trgSp_1x1x1"
        configuration = "3d_fullres"
    else:
        raise Exception("Challenge name not compatible.")

    return dataset_name, trainer, plans, configuration


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
    return f"nnUNet_raw_data_base='{input_path}' nnUNet_preprocessed='{preprocessed}' RESULTS_FOLDER='/tmp/'"


def run_infer_mednext(
    input_folder: str,
    output_folder: str,
    challenge_name: str,
    name: str,
    folds: List[int] = [0, 1, 2, 3, 4],
    save_npz: bool = True,
    ensemble: bool = True
) -> Tuple[List[Path], List[Path]]:
    """
    Runs nnUNet inference using MedNext configuration on the provided input data.

    This function executes the MedNext inference command either as an ensemble
    across all folds or individually per specified fold. It extracts the
    predictions and saves them as .npz and .pkl files.

    Args:
        input_folder (str): Path to the input folder containing prediction files.
        output_folder (str): Path to the folder where output predictions will be stored.
        challenge_name (str): Name of the challenge (e.g., BraTS-PED, BraTS-MET).
        name (str): Base name for the output prediction files.
        folds (List[int], optional): List of fold indices to process. Defaults to [0, 1, 2, 3, 4].
        save_npz (bool, optional): Whether to save the prediction probabilities as .npz files. Defaults to True.
        ensemble (bool, optional): Whether to perform ensemble prediction across all folds. Defaults to True.

    Returns:
        Tuple[List[Path], List[Path]]: 
            - List of paths to the saved .npz files.
            - List of paths to the saved .pkl files.

    Raises:
        Exception: If the challenge name is not compatible.
    """
    # Retrieve dataset information based on the challenge name
    dataset_name, trainer, plans, configuration = get_dataset_info(challenge_name)
    
    # Set environment paths required for nnUNet
    env_set = set_env_paths(input_folder, output_folder)

    if ensemble:
        # Define the output directory for ensemble predictions
        output_folder_fold = os.path.join(output_folder, "ens")
        print(f"Running nnUNet inference with all folds (ensemble)..")
        
        # Construct the MedNext inference command for ensemble
        cmd = (
            f"{env_set} mednextv1_predict "
            f"-i '{input_folder}' "
            f"-o '{output_folder_fold}' "
            f"-t '{dataset_name}' "
            f"-m '{configuration}' "
            f"-tr '{trainer}' "
            f"-p '{plans}'"
        )
        
        if save_npz:
            cmd += " --save_npz"
        
        # Execute the inference command
        subprocess.run(cmd, shell=True)
        
        # Return the paths to the saved .npz and .pkl files
        return (
            [Path(os.path.join(output_folder_fold, f"{name}.npz"))],
            [Path(os.path.join(output_folder_fold, f"{name}.pkl"))]
        )
    else:
        # Initialize lists to store paths to .npz and .pkl files for each fold
        npz_path_list: List[Path] = []
        pkl_path_list: List[Path] = []
        
        # Iterate over each specified fold and perform inference
        for fold in tqdm(folds, desc="Processing Folds"):
            output_folder_fold = os.path.join(output_folder, f"fold_{fold}")
            print(f"Running nnUNet inference for fold {fold}")
            
            # Construct the MedNext inference command for the current fold
            cmd = (
                f"{env_set} mednextv1_predict "
                f"-i '{input_folder}' "
                f"-o '{output_folder_fold}' "
                f"-t '{dataset_name}' "
                f"-m '{configuration}' "
                f"-tr '{trainer}' "
                f"-p '{plans}' "
                f"-f '{fold}'"
            )
            
            if save_npz:
                cmd += " --save_npz"
            
            # Execute the inference command for the current fold
            subprocess.run(cmd, shell=True)  # Executes the command in the shell
            
            # Append the paths to the saved .npz and .pkl files
            npz_path_list.append(Path(os.path.join(output_folder_fold, f"{name}.npz")))
            pkl_path_list.append(Path(os.path.join(output_folder_fold, f"{name}.pkl")))

        return npz_path_list, pkl_path_list
