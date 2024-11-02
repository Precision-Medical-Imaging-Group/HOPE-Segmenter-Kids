from pathlib import Path
import pandas as pd
import pickle
import argparse
import json
from typing import Any, List, Dict


def load_json(path: Path) -> Any:
    """
    Loads a JSON file from the specified path.

    Args:
        path (Path): A Path representing the file path.

    Returns:
        Any: The data loaded from the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """
    Saves data to a JSON file at the specified path.

    Args:
        path (Path): A Path representing the file path.
        data (Any): The data to be serialized and saved.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def get_cluster(radiomics_df: pd.DataFrame, cluster_artifacts: Dict[str, Any]) -> List[int]:
    """
    Assigns clusters to each case based on radiomics features.

    Args:
        radiomics_df (pd.DataFrame): DataFrame containing radiomics features.
        cluster_artifacts (Dict[str, Any]): Dictionary containing cluster artifacts including normalizer, PCA, and KMeans models.

    Returns:
        List[int]: List of cluster assignments for each case.
    """
    # Normalize the radiomics features
    normalizer = cluster_artifacts['normalizer']
    normal_values = normalizer.transform(radiomics_df.iloc[:, 1:].values)
    
    # Apply PCA transformation
    pca = cluster_artifacts['pca']
    scores_pca = pca.transform(normal_values)
    
    # Predict cluster assignments using KMeans
    kmeans = cluster_artifacts['kmeans']
    cluster_assignment = kmeans.predict(scores_pca)
    
    return cluster_assignment.tolist()


def get_cluster_artifacts(task: str) -> Dict[str, Any]:
    """
    Retrieves cluster artifacts based on the specified task.

    Args:
        task (str): The task identifier (e.g., 'BraTS-SSA', 'BraTS-PED').

    Returns:
        Dict[str, Any]: Dictionary containing cluster artifacts.
    """
    script_dir = Path(__file__).resolve().parent.parent
    if task == 'BraTS-SSA':
        pkl_path = script_dir / "kmeans-cluster-artifacts" / "SSA_cluster.pkl"
    elif task == 'BraTS-PED':
        pkl_path = script_dir / "kmeans-cluster-artifacts" / "PEDS_cluster.pkl"
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def main() -> None:
    """
    Main function to assign clusters to cases based on radiomics features.

    Parses command-line arguments, loads radiomics data and cluster artifacts,
    assigns clusters, and saves the updated data to a JSON file.
    """
    parser = argparse.ArgumentParser(description='Cluster Assignment for BraTS Data')
    parser.add_argument("-i", "--input_json", type=str, required=True, help="Path to input radiomics JSON file")
    parser.add_argument("-o", "--output_json", type=str, required=True, help="Path to output JSON file with cluster assignments")
    parser.add_argument("-c", "--cluster_pickle", type=str, required=True, help="Path to cluster artifacts pickle file")
    args = parser.parse_args()

    # Load radiomics data from JSON
    data = load_json(Path(args.input_json))
    df_radiomics = pd.DataFrame(data)

    # Load cluster artifacts from pickle file
    with open(Path(args.cluster_pickle), 'rb') as f:
        cluster_artifacts = pickle.load(f)

    # Assign clusters based on radiomics features
    cluster_assignment = get_cluster(df_radiomics, cluster_artifacts)

    # Update each case with its cluster assignment
    for case, cluster in zip(data, cluster_assignment):
        case['cluster'] = int(cluster)

    # Save the updated data to the output JSON file
    save_json(Path(args.output_json), data)


if __name__ == '__main__':
    main()

# Example Command to Run the Script:
# python pp_cluster/infer.py -i /home/abhijeet/Code/BraTS2024/datalist/radiomics/BraTS2024-GLI-training_data1_v2-radiomics.json -o datalist/pp_cluster_assignment/GLI.json -c kmeans-cluster-artifacts/GLI_cluster.pkl
