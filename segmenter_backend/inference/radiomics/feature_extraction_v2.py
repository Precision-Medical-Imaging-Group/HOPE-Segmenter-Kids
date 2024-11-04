"""
### Radimoics Feature Extraction Module

Provides functionalities related to radioimcs feature extraction from MRI sequences.
"""


from pathlib import Path
from typing import Any, List, Dict, Tuple, Union, Optional
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import cc3d
from scipy.ndimage import binary_dilation, generate_binary_structure
import shutil
import warnings


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


def load_jsonl(path: Path) -> List[Any]:
    """
    Loads a JSONL file (JSON lines) from the specified path.

    Args:
        path (Path): A Path representing the file path.

    Returns:
        List[Any]: A list of data loaded from the JSONL file.
    """
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, data: List[Any]) -> None:
    """
    Saves data to a JSONL file at the specified path.

    Args:
        path (Path): A Path representing the file path.
        data (List[Any]): A list of data to be serialized and saved.
    """
    with open(path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')


def maybe_make_dir(path: str) -> Path:
    """
    Creates a directory at the specified path if it does not exist.

    Args:
        path (str): A string representing the directory path.

    Returns:
        Path: A Path object for the created or existing directory.
    """
    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    return dir_path


def maybe_remove_dir(path: str) -> Path:
    """
    Removes a directory at the specified path if it exists.

    Args:
        path (str): A string representing the directory path.

    Returns:
        Path: A Path object for the removed or existing directory.
    """
    path_obj = Path(path)
    if path_obj.exists() and path_obj.is_dir():
        try:
            shutil.rmtree(path_obj)
            print(f"Directory {path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {path}: {e}")
    return path_obj


def extract_feature(
    paramfile: Path,
    data_path: Path,
    seg_path: Path,
    pid: str,
    seq: str,
    region: str = 'wt',
    tmpp: str = '.'
) -> Dict[str, Any]:
    """
    Extract features from a single MRI sequence.

    Args:
        paramfile (Path): Path to the parameter file for feature extraction.
        data_path (Path): Path to the data directory.
        seg_path (Path): Path to the segmentation file.
        pid (str): Patient ID.
        seq (str): Sequence identifier.
        region (str, optional): BraTS Region of interest. Defaults to 'wt'.
        tmpp (str, optional): Temporary directory path. Defaults to '.'.

    Returns:
        Dict[str, Any]: Dictionary of extracted features.
    """
    img_path = data_path / f"{pid}{seq}.nii.gz"
    tmp_path = maybe_make_dir(tmpp) / f"{pid}{seq}.json"
    
    # Construct and execute the feature extraction command
    cmd = f"pyradiomics {str(img_path)} {str(seg_path)} -o {str(tmp_path)} -f json --param {str(paramfile)}"
    os.system(cmd)
    
    # Load and process the extracted features
    new_dict = {'StudyID': pid}
    dt = load_json(tmp_path)
    tmp_path.unlink()  # Remove temporary file
    
    old_dict = dt[0]
    for k in old_dict.keys():
        if k.startswith('original_shape_'):
            new_k = k.replace('original_shape_', f'{region}_shape_', 1)
            new_dict[new_k] = old_dict[k]
        elif k.startswith('original_'):
            new_k = k.replace('original_', f'{region}{seq.replace("-", "_")}_', 1)
            new_dict[new_k] = old_dict[k]

    return new_dict


def extract_case(
    paramfile: Path,
    data_path: Path,
    seg_path: Path,
    pid: str,
    region: str = 'wt',
    tmpp: str = '.',
    sequences: List[str] = ['-t1n', '-t1c', '-t2w', '-t2f']
) -> Dict[str, Any]:
    """
    Extract features for a single case across multiple sequences.

    Args:
        paramfile (Path): Path to the parameter file for feature extraction.
        data_path (Path): Path to the data directory.
        seg_path (Path): Path to the segmentation file.
        pid (str): Patient ID.
        region (str, optional): BraTS Region of interest. Defaults to 'wt'.
        tmpp (str, optional): Temporary directory path. Defaults to '.'.
        sequences (List[str], optional): List of MRI sequences to process. Defaults to ['-t1n', '-t1c', '-t2w', '-t2f'].

    Returns:
        Dict[str, Any]: Dictionary of aggregated features for the case.
    """
    new_dict: Dict[str, Any] = {}
    for i, seq in enumerate(sequences):
        feature = extract_feature(paramfile, data_path, seg_path, pid, seq, region, tmpp)
        
        # Aggregate features, avoiding duplication of certain keys
        if i == 0:
            for k in feature.keys():
                new_dict[k] = feature[k]
        else:
            for k in feature.keys():
                if not (k.startswith('StudyID') or k.startswith(f'{region}_shape')):
                    new_dict[k] = feature[k]
    return new_dict


def create_dilation(
    seg_path: Path,
    out_path: Path,
    dilation_factor: int = 3,
    region: str = 'wt'
) -> None:
    """
    Create a dilated segmentation mask.

    Args:
        seg_path (Path): Path to the original segmentation file.
        out_path (Path): Path to save the dilated segmentation.
        dilation_factor (int, optional): Number of dilation iterations. Defaults to 3.
        region (str, optional): BraTS Region of interest. Defaults to 'wt'.
    """
    img_obj = nib.load(seg_path)
    img_data = img_obj.get_fdata()
    
    # Create a binary segmentation mask
    if region == 'wt':
        binary_seg = np.where(img_data > 0, 1, 0)
    else:
        warnings.warn(f"Invalid region: {region}. Computing whole tumor instead.")
        binary_seg = np.where(img_data > 0, 1, 0)
    
    # dilation_struct = generate_binary_structure(3, 1)
    # dilated_seg = binary_dilation(binary_seg, structure=dilation_struct, iterations=dilation_factor)
    # Identify connected components and retain the largest one
    labels_out, n = cc3d.connected_components(binary_seg, connectivity=26, return_N=True)
    vol_max = 0
    label_max = 0
    for i in range(n):
        tmp = np.where(labels_out == i + 1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol > vol_max:
            vol_max = vol
            label_max = i + 1

    dilated_seg = np.where(labels_out == label_max, 1, 0)
    seg_obj = nib.Nifti1Image(dilated_seg.astype(np.int8), img_obj.affine)
    nib.save(seg_obj, out_path)


def extract_all(
    paramfile: Path,
    data_path: Path,
    case_: str,
    seg_path: Path,
    dilation_factor: int = 3,
    region: str = 'wt',
    tmpp: str = '.',
    seg_suffix: str = '-seg',
    sequences: List[str] = ['-t1n', '-t1c', '-t2w', '-t2f']
) -> pd.DataFrame:
    """
    Extract all features for a given case.

    Args:
        paramfile (Path): Path to the parameter file for feature extraction.
        data_path (Path): Path to the data directory.
        case_ (str): Case identifier.
        seg_path (Path): Path to the segmentation file.
        dilation_factor (int, optional): Number of dilation iterations. Defaults to 3.
        region (str, optional): BraTS Region of interest. Defaults to 'wt'.
        tmpp (str, optional): Temporary directory path. Defaults to '.'.
        seg_suffix (str, optional): Suffix for segmentation files. Defaults to '-seg'.
        sequences (List[str], optional): List of sequences to process. Defaults to ['-t1n', '-t1c', '-t2w', '-t2f'].

    Returns:
        pd.DataFrame: DataFrame containing all extracted features.
    """
    cases = [data_path / case_]
    features: List[Dict[str, Any]] = []
    t0 = time.time()
    
    for i, case in enumerate(cases):
        dilated_seg_path = maybe_make_dir(tmpp) / f"{case.name}_{region}_dilated.nii.gz"
        
        # Create a dilated segmentation mask
        create_dilation(seg_path, dilated_seg_path, dilation_factor, region)
        
        # Extract features from the dilated segmentation
        features.append(
            extract_case(paramfile, data_path / case, dilated_seg_path, case.name, region, tmpp, sequences)
        )
        
        dilated_seg_path.unlink()  # Remove temporary dilated segmentation
        
        t1 = (time.time() - t0) / 60.0
        print(f"{i + 1:04d} {case} extraction time: {t1:.1f} min")
    
    return pd.DataFrame(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, help="Input directory containing cases")
    parser.add_argument("-o", "--output", type=str, help="Output directory for extracted features")
    parser.add_argument("-d", "--dilation", type=int, help="Dilation factor")
    parser.add_argument("-r", "--region", type=str, help="Region of interest")
    parser.add_argument("-p", "--param", type=str, help="Parameter file for feature extraction")
    parser.add_argument("-t", "--tmpp", default='./tmp', type=str, help="Temporary directory path")
    parser.add_argument("-s", "--seg-suffix", default='-seg', type=str, help="Segmentation file suffix")
    parser.add_argument("-seq", "--sequences", nargs='+', default=['-t1n', '-t1c', '-t2w', '-t2f'], help="MRI sequences to process")
    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    
    # Ensure temporary directory exists
    maybe_make_dir(args.tmpp)
    
    # Extract all features for the specified case
    extract_all(
        paramfile=Path(args.param),
        data_path=Path(args.input_dir),
        case_='Case1',
        seg_path=Path(args.output),
        dilation_factor=args.dilation,
        region=args.region,
        tmpp=args.tmpp,
        seg_suffix=args.seg_suffix,
        sequences=args.sequences
    )
    
    # Remove temporary directory after processing
    maybe_remove_dir(args.tmpp)
