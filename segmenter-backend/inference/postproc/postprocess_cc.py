import glob
import os
import nibabel as nib
import numpy as np
import cc3d
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Any, List, Tuple, Dict


LABEL_MAPPING_FACTORY: Dict[str, Dict[int, str]] = {
    "BraTS-PED": {
        1: "ET",
        2: "NET",
        3: "CC",
        4: "ED"
    },
    "BraTS-SSA": {
        1: "NCR",
        2: "ED",
        3: "ET"
    },
    "BraTS-MEN-RT": {
        1: "GTV"
    },
    "BraTS-MET": {
        1: "NETC",
        2: "SNFH",
        3: "ET"
    }
}


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the BraTS2024 CC Postprocessing script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='BraTS2024 CC Postprocessing.')
    parser.add_argument('--challenge_name', type=str, 
                        help='The name of the challenge (e.g., BraTS-PED, BraTS-MET)')
    parser.add_argument('--input_folder_pred', type=str,
                        help='The input folder containing the predictions')
    parser.add_argument('--output_folder_pp_cc', type=str,
                        help='The output folder to save the postprocessed predictions')
    parser.add_argument('--thresholds_file', type=str,
                        help='The JSON file containing the thresholds')
    parser.add_argument('--clusters_file', type=str,
                        help='The JSON file containing the clusters')

    return parser.parse_args()


def get_connected_components(img: np.ndarray, value: int) -> Tuple[np.ndarray, int]:
    """
    Identifies connected components for a specific label in the segmentation.

    Args:
        img (np.ndarray): The segmentation array.
        value (int): The label value to find connected components for.

    Returns:
        Tuple[np.ndarray, int]: 
            - Array with labeled connected components.
            - Number of connected components found.
    """
    labels, n = cc3d.connected_components((img == value).astype(np.int8), connectivity=26, return_N=True)
    return labels, n


def postprocess_cc(
    img: np.ndarray,
    thresholds_dict: Dict[str, Dict[str, Any]],
    label_mapping: Dict[int, str]
) -> np.ndarray:
    """
    Postprocesses the segmentation image by removing small connected components based on thresholds.

    Args:
        img (np.ndarray): The original segmentation array.
        thresholds_dict (Dict[str, Dict[str, Any]]): 
            Dictionary containing threshold values for each label.
        label_mapping (Dict[int, str]): Mapping from label numbers to label names.

    Returns:
        np.ndarray: The postprocessed segmentation array.
    """
    # Make a copy of pred_orig
    pred = img.copy()
    # Iterate over each of the labels
    for label_name, th_dict in thresholds_dict.items():
        label_number = int(label_name.split("_")[-1])
        label_name = label_mapping[label_number]
        th = th_dict["th"]
        print(f"Label {label_name} - value {label_number} - Applying threshold {th}")
        # Get connected components
        components, n = get_connected_components(pred, label_number)
        # Apply threshold to each of the components
        for el in range(1, n+1):
            # get the connected component (cc)
            cc = np.where(components == el, 1, 0)
            # calculate the volume
            vol = np.count_nonzero(cc)
            # if the volume is less than the threshold, dismiss the cc
            if vol < th:
                pred = np.where(components == el, 0, pred)
    # Return the postprocessed image
    return pred


def postprocess_batch(
    input_files: List[str],
    output_folder: str,
    thresholds_dict: Dict[str, Dict[str, Any]],
    label_mapping: Dict[int, str]
) -> None:
    """
    Applies postprocessing to a batch of segmentation files.

    Args:
        input_files (List[str]): List of input file paths.
        output_folder (str): Path to the output directory to save postprocessed files.
        thresholds_dict (Dict[str, Dict[str, Any]]): 
            Dictionary containing threshold values for each label.
        label_mapping (Dict[int, str]): Mapping from label numbers to label names.
    """
    for f in input_files:
        print(f"Processing file {f}")
        save_path = os.path.join(output_folder, os.path.basename(f))
        
        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping.")
            continue
        
        # Read the segmentation file
        pred_orig = nib.load(f).get_fdata()
        
        # Apply postprocessing
        pred = postprocess_cc(pred_orig, thresholds_dict, label_mapping)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the postprocessed segmentation
        nib.save(nib.Nifti1Image(pred, nib.load(f).affine), save_path)


def get_thresholds_task(
    challenge_name: str,
    input_file: str
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves threshold settings for a specific challenge from a JSON file.

    Args:
        challenge_name (str): The name of the challenge (e.g., BraTS-PED, BraTS-MET).
        input_file (str): Path to the JSON file containing thresholds.

    Returns:
        Dict[str, Dict[str, Any]]: Thresholds for the specified challenge.

    Raises:
        ValueError: If the challenge name is not found in the JSON file.
    """
    with open(input_file, 'r') as f:
        thresholds = json.load(f)
    
    if challenge_name not in thresholds:
        raise ValueError(f"Challenge {challenge_name} not found in the thresholds JSON file.")
    
    return thresholds[challenge_name]


def get_thresholds_cluster(
    thresholds: Dict[str, Dict[str, Any]],
    cluster_name: str
) -> Dict[str, Any]:
    """
    Retrieves threshold settings for a specific cluster within a challenge.

    Args:
        thresholds (Dict[str, Dict[str, Any]]): Thresholds for the challenge.
        cluster_name (str): The name of the cluster (e.g., "cluster_1").

    Returns:
        Dict[str, Any]: Threshold settings for the specified cluster.

    Raises:
        ValueError: If the cluster name is not found in the thresholds.
    """
    if cluster_name not in thresholds:
        raise ValueError(f"Cluster {cluster_name} not found in the thresholds JSON file.")
    return thresholds[cluster_name]


def get_files(input_folder_pred: str) -> List[str]:
    """
    Retrieves all prediction file paths from the input directory.

    Args:
        input_folder_pred (str): Path to the input directory containing prediction files.

    Returns:
        List[str]: Sorted list of prediction file paths.
    """
    files_pred = sorted(glob.glob(os.path.join(input_folder_pred, "*.nii.gz")))
    print(f"Found {len(files_pred)} files to be processed.")
    return files_pred


def get_cluster_files(
    cluster_assignment: List[Dict[str, Any]],
    cluster_id: int,
    files_pred: List[str]
) -> List[str]:
    """
    Retrieves prediction files that belong to a specific cluster.

    Args:
        cluster_assignment (List[Dict[str, Any]]): 
            List of cluster assignments with StudyID and cluster.
        cluster_id (int): The cluster identifier.
        files_pred (List[str]): List of all prediction file paths.

    Returns:
        List[str]: List of prediction files belonging to the specified cluster.
    """
    # Extract StudyIDs for the given cluster
    cluster_ids = [e["StudyID"] for e in cluster_assignment if e["cluster"] == cluster_id]
    
    # Filter prediction files that match the StudyIDs
    cluster_files_pred = [
        f for f in files_pred 
        if os.path.basename(f).replace(".nii.gz", "") in cluster_ids
    ]
    print(f"Cluster {cluster_id} contains {len(cluster_files_pred)} files.")
    return cluster_files_pred


def read_cluster_assignment(clusters_json: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Reads cluster assignments from a JSON file.

    Args:
        clusters_json (str): Path to the JSON file containing cluster assignments.

    Returns:
        Tuple[List[Dict[str, Any]], List[int]]: 
            - List of cluster assignments with StudyID and cluster.
            - Sorted list of unique cluster identifiers.
    """
    with open(clusters_json, "r") as f:
        cluster_assignment = json.load(f)
    
    # Filter relevant keys
    cluster_assignment = [
        {key: value for key, value in e.items() if key in ["StudyID", "cluster"]}
        for e in cluster_assignment
    ]
    
    cluster_array = np.unique([e["cluster"] for e in cluster_assignment])
    print(f"Found {len(cluster_array)} clusters: {sorted(cluster_array)}.")
    return cluster_assignment.tolist(), sorted(cluster_array)


def read_cluster_assignment_df(clusterdf: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Reads cluster assignments from a pandas DataFrame.

    Args:
        clusterdf (pd.DataFrame): DataFrame containing cluster assignments.

    Returns:
        Tuple[List[Dict[str, Any]], List[int]]: 
            - List of cluster assignments with StudyID and cluster.
            - Sorted list of unique cluster identifiers.
    """
    # Convert DataFrame to list of dictionaries
    cluster_assignment = clusterdf.to_dict(orient="records")
    
    # Filter relevant keys
    cluster_assignment = [
        {key: value for key, value in e.items() if key in ["StudyID", "cluster"]}
        for e in cluster_assignment
    ]
    
    cluster_array = np.unique([e["cluster"] for e in cluster_assignment])
    print(f"Found {len(cluster_array)} clusters: {sorted(cluster_array)}.")
    return cluster_assignment.tolist(), sorted(cluster_array)


def remove_small_component(
    challenge_name: str,
    thresholds_file: str,
    input_folder_pred: str,
    clustersdf: pd.DataFrame,
    output_folder_pp_cc: str
) -> str:
    """
    Removes small connected components from segmentation predictions based on cluster-specific thresholds.

    Args:
        challenge_name (str): The name of the challenge (e.g., BraTS-PED, BraTS-MET).
        thresholds_file (str): Path to the JSON file containing threshold settings.
        input_folder_pred (str): Path to the input directory containing prediction files.
        clustersdf (pd.DataFrame): DataFrame containing cluster assignments.
        output_folder_pp_cc (str): Path to the output directory to save postprocessed predictions.

    Returns:
        str: Path to the output directory containing postprocessed predictions.
    """
    # Retrieve label mapping for the challenge
    label_mapping = LABEL_MAPPING_FACTORY.get(challenge_name)
    if label_mapping is None:
        raise ValueError(f"Unsupported challenge name: {challenge_name}")
    
    # Load threshold settings for the challenge
    thresholds = get_thresholds_task(challenge_name, thresholds_file)
    
    # Retrieve all prediction files
    files_pred = get_files(input_folder_pred)
    
    # Read cluster assignments from DataFrame
    cluster_assignment, cluster_array = read_cluster_assignment_df(clustersdf)
    
    # Iterate over each cluster and apply postprocessing
    for cluster in cluster_array:
        cluster_files_pred = get_cluster_files(cluster_assignment, cluster, files_pred)
        cluster_key = f"cluster_{cluster}"
        thresholds_cluster = get_thresholds_cluster(thresholds, cluster_key)
        
        # Apply postprocessing to the cluster-specific prediction files
        postprocess_batch(
            input_files=cluster_files_pred,
            output_folder=output_folder_pp_cc,
            thresholds_dict=thresholds_cluster,
            label_mapping=label_mapping
        )
    
    return output_folder_pp_cc


def main() -> None:
    """
    Main function to execute label redefinition and connected component postprocessing.

    Parses command-line arguments, loads necessary data, and performs postprocessing
    on prediction files based on cluster assignments and threshold settings.
    """
    args = parse_args()
    
    # Assign label mapping based on the challenge name
    args.label_mapping = LABEL_MAPPING_FACTORY.get(args.challenge_name)
    if args.label_mapping is None:
        raise ValueError(f"Unsupported challenge name: {args.challenge_name}")
    
    # Load threshold settings for the specified challenge
    thresholds = get_thresholds_task(args.challenge_name, args.thresholds_file)
    
    # Retrieve all prediction files from the input directory
    files_pred = get_files(args.input_folder_pred)
    
    # Read cluster assignments from the clusters JSON file
    cluster_assignment, cluster_array = read_cluster_assignment(args.clusters_file)
    
    # Iterate over each cluster and apply label redefinition postprocessing
    for cluster in cluster_array:
        cluster_files_pred = get_cluster_files(cluster_assignment, cluster, files_pred)
        cluster_key = f"cluster_{cluster}"
        thresholds_cluster = get_thresholds_cluster(thresholds, cluster_key)
        
        # Apply postprocessing to the cluster-specific prediction files
        postprocess_batch(
            input_files=cluster_files_pred,
            output_folder=args.output_folder_pp_cc,
            thresholds_dict=thresholds_cluster,
            label_mapping=args.label_mapping
        )


if __name__ == "__main__":
    main()

    # Example Command to Run the Script:
    # python remove_small_component.py --challenge_name BraTS-PED \
    # --input_folder_pred /path/to/predictions \
    # --output_folder_pp_cc /path/to/output \
    # --thresholds_file /path/to/thresholds.json \
    # --clusters_file /path/to/clusters.json
