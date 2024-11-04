"""
### Post-processing 

Provides functionalities related to computing connected components and postprocessing segmentation predictions by removing small connected components based on thresholds.
"""


from pathlib import Path
import pandas as pd
import pickle
import argparse
import json
import numpy as np
import nibabel as nib
import cc3d
import SimpleITK as sitk
import glob
import os
from tqdm import tqdm
import shutil
import warnings
from typing import Any, List, Tuple, Dict


def maybe_make_dir(path: str) -> None:
    """
    Creates a directory at the specified path if it does not exist.

    Args:
        path (str): A string representing the directory path.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def read_image(path: Path) -> Tuple[sitk.Image, np.ndarray]:
    """
    Reads an image file and returns the SimpleITK image object and its NumPy array.

    Args:
        path (Path): Path to the image file.

    Returns:
        Tuple[sitk.Image, np.ndarray]: The SimpleITK image object and its corresponding NumPy array.
    """
    img_sitk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_sitk)
    return img_sitk, img


def convert_labels_back_to_BraTS(seg: np.ndarray) -> np.ndarray:
    """
    Converts segmentation labels back to BraTS format.

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        np.ndarray: The converted segmentation array.
    """
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 3
    new_seg[seg == 2] = 1
    return new_seg


def get_ratio_ncr_wt(seg: np.ndarray) -> float:
    """
    Calculates the ratio of necrotic and non-enhancing tumor core (NCR) voxels to whole tumor (WT) voxels.

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        float: The NCR to WT voxel ratio.
    """
    ncr_voxels = np.sum(seg == 1)
    wt_voxels = np.sum(seg != 0)
    if wt_voxels == 0:
        return 1.0
    return ncr_voxels / wt_voxels


def get_ratio_ed_wt(seg: np.ndarray) -> float:
    """
    Calculates the ratio of peritumoral edema (ED) voxels to whole tumor (WT) voxels.

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        float: The ED to WT voxel ratio.
    """
    ed_voxels = np.sum(seg == 2)
    wt_voxels = np.sum(seg != 0)
    if wt_voxels == 0:
        return 1.0
    return ed_voxels / wt_voxels


def get_ratio_et_wt(seg: np.ndarray) -> float:
    """
    Calculates the ratio of enhancing tumor (ET) voxels to whole tumor (WT) voxels.

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        float: The ET to WT voxel ratio.
    """
    et_voxels = np.sum(seg == 3)
    wt_voxels = np.sum(seg != 0)
    if wt_voxels == 0:
        return 1.0
    return et_voxels / wt_voxels


def get_ratio_tc_wt(seg: np.ndarray) -> float:
    """
    Calculates the ratio of tumor core (TC) voxels to whole tumor (WT) voxels.

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        float: The TC to WT voxel ratio.
    """
    tc_voxels = np.sum((seg == 1) & (seg == 3))
    wt_voxels = np.sum(seg != 0)
    if wt_voxels == 0:
        return 1.0
    return tc_voxels / wt_voxels


def convert_et_to_ncr(seg: np.ndarray) -> np.ndarray:
    """
    Converts enhancing tumor (ET) labels to necrotic and non-enhancing tumor core (NCR).

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        np.ndarray: The modified segmentation array.
    """
    seg[seg == 3] = 1
    return seg


def convert_ed_to_ncr(seg: np.ndarray) -> np.ndarray:
    """
    Converts peritumoral edema (ED) labels to necrotic and non-enhancing tumor core (NCR).

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        np.ndarray: The modified segmentation array.
    """
    seg[seg == 2] = 1
    return seg


def get_greatest_label(seg: np.ndarray) -> Tuple[str, float]:
    """
    Determines the label with the highest ratio to whole tumor (WT) voxels.

    Args:
        seg (np.ndarray): The segmentation array.

    Returns:
        Tuple[str, float]: The label with the highest ratio and its corresponding ratio value.
    """
    ratios = {
        "ncr": get_ratio_ncr_wt(seg),
        "ed": get_ratio_ed_wt(seg),
        "et": get_ratio_et_wt(seg),
        # "tc": get_ratio_tc_wt(seg),
    }
    greatest_label = max(ratios, key=ratios.get)
    return greatest_label, ratios[greatest_label]


def redefine_et_ed_labels(
    seg_file: Path,
    out_file: Path,
    label: str = "et",
    ratio: float = 0.0
) -> np.ndarray:
    """
    Redefines ET or ED labels to NCR in the segmentation based on a specified ratio.

    Args:
        seg_file (Path): Path to the input segmentation file.
        out_file (Path): Path to save the postprocessed segmentation file.
        label (str, optional): Label to optimize ("et" or "ed"). Defaults to "et".
        ratio (float, optional): Threshold ratio for label optimization. Defaults to 0.0.

    Returns:
        np.ndarray: The modified segmentation array.
    """
    seg_obj = nib.load(seg_file)
    seg = seg_obj.get_fdata()
    if label == "et":
        ratio_et_wt = get_ratio_et_wt(seg)
        if ratio_et_wt < ratio:
            seg = convert_et_to_ncr(seg)
    elif label == "ed":
        ratio_ed_wt = get_ratio_ed_wt(seg)
        if ratio_ed_wt < ratio:
            seg = convert_ed_to_ncr(seg)
    new_obj = nib.Nifti1Image(seg.astype(np.int8), seg_obj.affine)
    nib.save(new_obj, out_file)
    return seg


def postprocess_image(
    seg: np.ndarray,
    label: str,
    ratio: float = 0.04
) -> np.ndarray:
    """
    Postprocesses the segmentation image by redefining ET or ED labels based on a ratio.

    Args:
        seg (np.ndarray): The segmentation array.
        label (str): Label to optimize ("et" or "ed").
        ratio (float, optional): Threshold ratio for label optimization. Defaults to 0.04.

    Returns:
        np.ndarray: The postprocessed segmentation array.
    """
    if label == "et":
        ratio_et_wt = get_ratio_et_wt(seg)
        if ratio_et_wt < ratio:
            seg = convert_et_to_ncr(seg)
    elif label == "ed":
        ratio_ed_wt = get_ratio_ed_wt(seg)
        if ratio_ed_wt < ratio:
            seg = convert_ed_to_ncr(seg)
    return seg


def save_image(
    img: np.ndarray,
    img_sitk: sitk.Image,
    out_path: Path
) -> None:
    """
    Saves the NumPy array as a NIfTI image with the original image's metadata.

    Args:
        img (np.ndarray): The image array to save.
        img_sitk (sitk.Image): The original SimpleITK image object.
        out_path (Path): Path to save the new image.
    """
    new_img_sitk = sitk.GetImageFromArray(img)
    new_img_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(new_img_sitk, out_path)


def postprocess_batch(
    input_folder: Path,
    output_folder: Path,
    label_to_optimize: str,
    ratio: float = 0.04,
    convert_to_brats_labels: bool = False
) -> None:
    """
    Postprocesses a batch of segmentation files by optimizing specified labels.

    Args:
        input_folder (Path): Path to the input directory containing segmentation files.
        output_folder (Path): Path to the output directory to save postprocessed files.
        label_to_optimize (str): Label to optimize ("et" or "ed").
        ratio (float, optional): Threshold ratio for label optimization. Defaults to 0.04.
        convert_to_brats_labels (bool, optional): Whether to convert labels back to BraTS format. Defaults to False.
    """
    seg_list = sorted(glob.glob(os.path.join(input_folder, "*.nii.gz")))
    for seg_path in tqdm(seg_list):
        seg_sitk, seg = read_image(Path(seg_path))
        if convert_to_brats_labels:
            seg = convert_labels_back_to_BraTS(seg)
        seg_pp = postprocess_image(seg, label_to_optimize, ratio)
        out_path = output_folder / Path(seg_path).name
        save_image(seg_pp, seg_sitk, out_path)


def get_connected_labels(seg_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Identifies connected components for NCR, ED, and ET labels in the segmentation.

    Args:
        seg_file (Path): Path to the segmentation file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]: 
            - Connected labels for NCR, ED, ET
            - Number of connected components for NCR, ED, ET
    """
    seg_obj = nib.load(seg_file)
    seg = seg_obj.get_fdata()
    seg_ncr = np.where(seg == 1, 1, 0)
    seg_ed = np.where(seg == 2, 2, 0)
    seg_et = np.where(seg == 3, 3, 0)
    labels_ncr, n_ncr = cc3d.connected_components(seg_ncr, connectivity=26, return_N=True)
    labels_ed, n_ed = cc3d.connected_components(seg_ed, connectivity=26, return_N=True)
    labels_et, n_et = cc3d.connected_components(seg_et, connectivity=26, return_N=True)
    return labels_ncr, labels_ed, labels_et, n_ncr, n_ed, n_et


def remove_disconnected(
    seg_file: Path,
    out_file: Path,
    t_ncr: int = 50,
    t_ed: int = 50,
    t_et: int = 50
) -> Tuple[int, int, int, int, int, int]:
    """
    Removes disconnected small components from the segmentation based on thresholds.

    Args:
        seg_file (Path): Path to the input segmentation file.
        out_file (Path): Path to save the cleaned segmentation file.
        t_ncr (int, optional): Threshold for NCR voxel count. Defaults to 50.
        t_ed (int, optional): Threshold for ED voxel count. Defaults to 50.
        t_et (int, optional): Threshold for ET voxel count. Defaults to 50.

    Returns:
        Tuple[int, int, int, int, int, int]: 
            Number of removed NCR, total NCR, removed ED, total ED, removed ET, total ET.
    """
    seg_obj = nib.load(seg_file)
    labels_ncr, labels_ed, labels_et, n_ncr, n_ed, n_et = get_connected_labels(seg_file)
    
    # Process NCR
    vols = []
    for i in range(n_ncr):
        tmp = np.where(labels_ncr == i + 1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol < t_ncr:
            labels_ncr = np.where(labels_ncr == i + 1, 0, labels_ncr)
            vols.append(vol)
    removed_ncr = len(vols)
    
    # Process ED
    vols = []
    for i in range(n_ed):
        tmp = np.where(labels_ed == i + 1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol < t_ed:
            labels_ed = np.where(labels_ed == i + 1, 0, labels_ed)
            vols.append(vol)
    removed_ed = len(vols)
    
    # Process ET
    vols = []
    for i in range(n_et):
        tmp = np.where(labels_et == i + 1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol < t_et:
            labels_et = np.where(labels_et == i + 1, 0, labels_et)
            vols.append(vol)
    removed_et = len(vols)

    # Combine the cleaned labels
    new_ncr = np.where(labels_ncr != 0, 1, 0)
    new_ed = np.where(labels_ed != 0, 2, 0)
    new_et = np.where(labels_et != 0, 3, 0)
    new_seg = new_ncr + new_ed + new_et
    new_obj = nib.Nifti1Image(new_seg.astype(np.int8), seg_obj.affine)
    nib.save(new_obj, out_file)
    return removed_ncr, n_ncr, removed_ed, n_ed, removed_et, n_et


def remove_disconnected_from_dir(
    input_dir: Path,
    output_dir: Path,
    t_ncr: int = 50,
    t_ed: int = 50,
    t_et: int = 50
) -> Path:
    """
    Removes disconnected small components from all segmentation files in a directory.

    Args:
        input_dir (Path): Path to the input directory containing segmentation files.
        output_dir (Path): Path to the output directory to save cleaned segmentation files.
        t_ncr (int, optional): Threshold for NCR voxel count. Defaults to 50.
        t_ed (int, optional): Threshold for ED voxel count. Defaults to 50.
        t_et (int, optional): Threshold for ET voxel count. Defaults to 50.

    Returns:
        Path: Path to the output directory.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for seg1 in input_dir.iterdir():
        if seg1.name.endswith('.nii.gz'):
            casename = seg1.name
            seg2 = output_dir / casename
            removed_ncr, n_ncr, removed_ed, n_ed, removed_et, n_et = remove_disconnected(
                seg1, seg2, t_ncr=t_ncr, t_ed=t_ed, t_et=t_et
            )
            print(
                f"{casename} removed regions NCR {removed_ncr:03d}/{n_ncr:03d} "
                f"ED {removed_ed:03d}/{n_ed:03d} ET {removed_et:03d}/{n_et:03d}"
            )
        else:
            print("Wrong input file!")

    return output_dir
