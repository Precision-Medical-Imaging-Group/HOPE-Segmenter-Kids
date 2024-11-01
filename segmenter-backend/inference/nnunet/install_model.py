import zipfile
import os
import argparse
from typing import Optional


def check_path(path: str) -> None:
    """
    Checks if the provided path points to a ZIP file.

    Args:
        path (str): Path to be checked.

    Raises:
        AssertionError: If the path does not have a ".zip" extension.
    """
    assert path.endswith(".zip"), "Path must point to a ZIP file"


def install_model_from_zip(path: str, mednext: bool = False) -> None:
    """
    Installs a pretrained nnU-Net (v2) model from a zipped file.

    Args:
        path (str): Path to the zipped model file.
        mednext (bool, optional): Flag to specify if the MedNext configuration is used. Defaults to False.

    Raises:
        AssertionError: If the provided path is not a ZIP file.
    """
    # Verify that the provided path is a ZIP file
    check_path(path)
    
    # Default extraction path
    zip_path = '/tmp/'

    if mednext:
        # If MedNext is specified, change the extraction path and create the directory if it doesn't exist
        zip_path = '/tmp/nnUNet'
        os.makedirs(zip_path, exist_ok=True)
    
    # Extract the ZIP file to the specified directory
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(zip_path)
    
    # Uncomment the following lines if you need to run an external command after extraction
    # cmd = f"nnUNetv2_install_pretrained_model_from_zip {path}"
    # subprocess.run(cmd, shell=True)  # Executes the command in the shell
