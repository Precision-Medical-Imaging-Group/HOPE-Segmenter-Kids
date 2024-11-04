"""
### Weighted Ensemble for Pediatric Brain Tumor Segmentation

Provides functionalities related to running a weighted ensemble of predictions from SwinUNETR, nnUNet, and MedNeXt models.
"""


from pathlib import Path
import numpy as np
import nibabel as nib
import os
import subprocess
from tqdm import tqdm
from typing import List, Tuple, Optional, Union, Dict


def maybe_make_dir(path: Union[str, Path]) -> Path:
    """
    Creates a directory at the specified path if it does not exist.

    Args:
        path (Union[str, Path]): Path to the directory to be created.

    Returns:
        Path: The path to the created or existing directory.
    """
    os.makedirs(path, exist_ok=True)
    return Path(path)


def convert_npz_mednext(
    npz_path: Union[str, Path],
    pkl_path: Union[str, Path],
    save_nifti: bool = False,
    nifti_dir: Union[str, Path] = Path('./tmp_prob_nifti'),
    suffix: str = 'mednext'
) -> np.ndarray:
    """
    Converts MedNeXt .npz and .pkl files to a NumPy array and optionally saves as NIfTI files.

    Args:
        npz_path (Union[str, Path]): Path to the .npz file.
        pkl_path (Union[str, Path]): Path to the .pkl file.
        save_nifti (bool, optional): Whether to save the probabilities as NIfTI files. Defaults to False.
        nifti_dir (Union[str, Path], optional): Directory to save NIfTI files. Defaults to './tmp_prob_nifti'.
        suffix (str, optional): Suffix for the NIfTI file names. Defaults to 'mednext'.

    Returns:
        np.ndarray: Stacked probability array.
    """
    npz = np.load(npz_path, allow_pickle=True)
    pkl = np.load(pkl_path, allow_pickle=True)
    prob = npz[npz.files[0]].astype(np.float32)
    bbox = pkl['crop_bbox']
    shape_original_before_cropping = pkl['original_size_of_raw_data']
    out_list = []

    for i in range(prob.shape[0]):
        if bbox is not None:
            seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.float16)
            for c in range(3):
                bbox[c][1] = min(bbox[c][0] + prob[i].shape[c], shape_original_before_cropping[c])
            seg_old_size[
                bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1]
            ] = prob[i]
        else:
            seg_old_size = prob[i]
        
        out = np.swapaxes(seg_old_size, 0, 2)
        out_list.append(out)

        if save_nifti:
            nifti_dir_path = maybe_make_dir(nifti_dir)
            nifti_path = nifti_dir_path / f"{npz_path.name.split('.npz')[0]}_{i}_{suffix}.nii.gz"
            nib.save(
                nib.Nifti1Image(out.astype(np.float32), affine=np.eye(4)),
                nifti_path
            )

    return np.stack(out_list, axis=0).astype(np.float32)


def convert_npz_nnunet(
    npz_path: Union[str, Path],
    save_nifti: bool = False,
    nifti_dir: Union[str, Path] = Path('./tmp_prob_nifti'),
    suffix: str = 'nnunet'
) -> np.ndarray:
    """
    Converts nnUNet .npz files to a NumPy array and optionally saves as NIfTI files.

    Args:
        npz_path (Union[str, Path]): Path to the .npz file.
        save_nifti (bool, optional): Whether to save the probabilities as NIfTI files. Defaults to False.
        nifti_dir (Union[str, Path], optional): Directory to save NIfTI files. Defaults to './tmp_prob_nifti'.
        suffix (str, optional): Suffix for the NIfTI file names. Defaults to 'nnunet'.

    Returns:
        np.ndarray: Stacked probability array.
    """
    npz = np.load(npz_path, allow_pickle=True)
    prob = npz[npz.files[0]].astype(np.float32)
    out_list = []

    for i in range(prob.shape[0]):
        out = np.swapaxes(prob[i], 0, 2)
        out_list.append(out)

        if save_nifti:
            nifti_dir_path = maybe_make_dir(nifti_dir)
            nifti_path = nifti_dir_path / f"{npz_path.name.split('.npz')[0]}_{i}_{suffix}.nii.gz"
            nib.save(
                nib.Nifti1Image(out.astype(np.float32), affine=np.eye(4)),
                nifti_path
            )

    return np.stack(out_list, axis=0).astype(np.float32)


def convert_npz_swinunetr(
    npz_path: Union[str, Path],
    save_nifti: bool = False,
    nifti_dir: Union[str, Path] = Path('./tmp_prob_nifti'),
    suffix: str = 'swinunetr'
) -> np.ndarray:
    """
    Converts SwinUNETR .npz files to a NumPy array and optionally saves as NIfTI files.

    Args:
        npz_path (Union[str, Path]): Path to the .npz file.
        save_nifti (bool, optional): Whether to save the probabilities as NIfTI files. Defaults to False.
        nifti_dir (Union[str, Path], optional): Directory to save NIfTI files. Defaults to './tmp_prob_nifti'.
        suffix (str, optional): Suffix for the NIfTI file names. Defaults to 'swinunetr'.

    Returns:
        np.ndarray: Probability array.
    """
    npz = np.load(npz_path, allow_pickle=True)
    prob = npz[npz.files[0]].astype(np.float32)

    for i in range(prob.shape[0]):
        if save_nifti:
            nifti_dir_path = maybe_make_dir(nifti_dir)
            nifti_path = nifti_dir_path / f"{npz_path.name.split('.npz')[0]}_{i}_{suffix}.nii.gz"
            nib.save(
                nib.Nifti1Image(prob[i].astype(np.float32), affine=np.eye(4)),
                nifti_path
            )

    return prob.astype(np.float32)


def ped_ensemble(
    swinunetr_npz_path_list: List[Union[str, Path]],
    nnunet_npz_path_list: List[Union[str, Path]],
    mednext_npz_path_list: List[Union[str, Path]],
    mednext_pkl_path_list: List[Union[str, Path]],
    ensembled_path: Union[str, Path],
    input_img: Union[str, Path],
    weights: List[float] = [1.0, 1.0, 1.0]
) -> Union[Path, Tuple[Path, Path]]:
    """
    Performs ensemble of predictions from SwinUNETR, nnUNet, and MedNeXt models.

    Args:
        swinunetr_npz_path_list (List[Union[str, Path]]): List of paths to SwinUNETR .npz files.
        nnunet_npz_path_list (List[Union[str, Path]]): List of paths to nnUNet .npz files.
        mednext_npz_path_list (List[Union[str, Path]]): List of paths to MedNeXt .npz files.
        mednext_pkl_path_list (List[Union[str, Path]]): List of paths to MedNeXt .pkl files.
        ensembled_path (Union[str, Path]): Directory to save the ensembled predictions.
        input_img (Union[str, Path]): Path to the original input image.
        weights (List[float], optional): Weights for each model in the ensemble. Defaults to [1.0, 1.0, 1.0].

    Returns:
        Union[Path, Tuple[Path, Path]]: Path(s) to the saved ensembled NIfTI file(s).
    """
    ensembled_path = maybe_make_dir(ensembled_path)

    case = nnunet_npz_path_list[0].name.split('.npz')[0]
    print(f"Ensemble {case}")

    ensembled_nifti = ensembled_path / f"{case}.nii.gz"
    if ensembled_nifti.exists():
        print(f"File {ensembled_nifti} already exists. Skipping.")
        return ensembled_nifti

    # SwinUNETR
    prob_swinunetr = convert_npz_swinunetr(swinunetr_npz_path_list[0])
    for swin_npz in swinunetr_npz_path_list[1:]:
        prob_swinunetr += convert_npz_swinunetr(swin_npz)
    prob_swinunetr /= len(swinunetr_npz_path_list)
    print(f"Probabilities SwinUNETR: {prob_swinunetr.shape}")

    # nnUNet
    prob_nnunet = convert_npz_nnunet(nnunet_npz_path_list[0])
    for nnunet_npz in nnunet_npz_path_list[1:]:
        prob_nnunet += convert_npz_nnunet(nnunet_npz)
    prob_nnunet /= len(nnunet_npz_path_list)
    print(f"Probabilities nnUNet: {prob_nnunet.shape}")

    # MedNeXt
    prob_mednext = convert_npz_mednext(mednext_npz_path_list[0], mednext_pkl_path_list[0])
    for mednext_npz, mednext_pkl in zip(mednext_npz_path_list[1:], mednext_pkl_path_list[1:]):
        prob_mednext += convert_npz_mednext(mednext_npz, mednext_pkl)
    prob_mednext /= len(mednext_npz_path_list)
    print(f"Probabilities MedNeXt: {prob_mednext.shape}")

    # Weighted Ensemble
    prob = (
        weights[0] * prob_swinunetr +
        weights[1] * prob_nnunet +
        weights[2] * prob_mednext
    )
    prob /= sum(weights)

    # Generate segmentation by taking argmax
    seg = np.argmax(prob, axis=0)
    print(f"Segmentation shape: {seg.shape}")

    # Save the ensembled segmentation
    img = nib.load(input_img)
    nib.save(nib.Nifti1Image(seg.astype(np.int8), img.affine), ensembled_nifti)
    print(f"Saved ensembled segmentation to {ensembled_nifti}")

    return ensembled_nifti


def batch_ped_ensemble(
    swinunetr_pred_dirs: List[Union[str, Path]],
    nnunet_pred_dirs: List[Union[str, Path]],
    mednext_pred_dirs: List[Union[str, Path]],
    input_img_dir: Union[str, Path],
    ensembled_dir: Union[str, Path],
    weights: List[float] = [1.0, 1.0, 1.0],
    cv: bool = False
) -> None:
    """
    Performs ensemble of predictions for multiple cases either in cross-validation (cv) or validation mode.

    Args:
        swinunetr_pred_dirs (List[Union[str, Path]]): List of directories containing SwinUNETR predictions.
        nnunet_pred_dirs (List[Union[str, Path]]): List of directories containing nnUNet predictions.
        mednext_pred_dirs (List[Union[str, Path]]): List of directories containing MedNeXt predictions.
        input_img_dir (Union[str, Path]): Directory containing the original input images.
        ensembled_dir (Union[str, Path]): Directory to save the ensembled predictions.
        weights (List[float], optional): Weights for each model in the ensemble. Defaults to [1.0, 1.0, 1.0].
        cv (bool, optional): Flag indicating whether to perform cross-validation ensemble. Defaults to False.
    """
    ensembled_dir = maybe_make_dir(ensembled_dir)

    if cv:
        
        # Get files inside each of the nnunet_pred_dirs items
        cases = [[case_path.name[:-7] for case_path in pred_dir.iterdir() if str(case_path).endswith(".nii.gz")] for pred_dir in nnunet_pred_dirs]
        
        for f in range(len(cases)):
            for case in cases[f]:
                
                # swinunetr_npz_path_list = [swinunetr_pred_dirs[f] / f"{case}-t1n.npz"]
                swinunetr_npz_path_list = [swinunetr_pred_dirs[f] / f"{case}.npz"]
                nnunet_npz_path_list = [nnunet_pred_dirs[f] / f"{case}.npz"]
                mednext_npz_path_list = [mednext_pred_dirs[f] / f"{case}.npz"]
                mednext_pkl_path_list = [mednext_pred_dirs[f] / f"{case}.pkl"]

                saved_path = ped_ensemble(
                    swinunetr_npz_path_list, 
                    nnunet_npz_path_list, 
                    mednext_npz_path_list, 
                    mednext_pkl_path_list, 
                    ensembled_dir, 
                    input_img_dir / case / f"{case}-t1n.nii.gz", 
                    weights=weights
                )
                print(f"Saved {saved_path}")
        
    else:
    
        cases = [case_path.name for case_path in input_img_dir.iterdir() if case_path.is_dir()]

        for case in cases:
            swinunetr_npz_path_list = [pred / f"{case}-t1n.npz" for pred in swinunetr_pred_dirs]
            nnunet_npz_path_list = [pred / f"{case}.npz" for pred in nnunet_pred_dirs]
            mednext_npz_path_list = [pred / f"{case}.npz" for pred in mednext_pred_dirs]
            mednext_pkl_path_list = [pred / f"{case}.pkl" for pred in mednext_pred_dirs]

            saved_path = ped_ensemble(
                swinunetr_npz_path_list, 
                nnunet_npz_path_list, 
                mednext_npz_path_list, 
                mednext_pkl_path_list, 
                ensembled_dir, 
                input_img_dir / case / f"{case}-t1n.nii.gz", 
                weights=weights
            )
            print(f"Saved {saved_path}")


def main_cv():
    """
    Executes ensemble predictions for cross-validation (cv) mode.
    """
    swinunetr_pred_path = Path("/home/v363/v363397/media/output_cv/ped_stratified")
    swinunetr_pred_dirs = [
        swinunetr_pred_path / f'swinunetr_e650_f{i}_b1p4' for i in [0, 1, 2]
    ] + [
        swinunetr_pred_path / f'swinunetr_e1000_f{i}_b1p4' for i in [3, 4]
    ]

    nnunet_pred_path = Path("/home/v363/v363397/media/nnUNet_results/Dataset021_BraTS2024-PED/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres")
    nnunet_pred_dirs = [nnunet_pred_path / f'fold_{i}' / 'validation' for i in range(5)]

    mednext_pred_path = Path("/home/v363/v363397/media/nnUNet_trained_models/nnUNet/3d_fullres/Task021_BraTS2024-PEDs/nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit__nnUNetPlansv2.1_trgSp_1x1x1")
    mednext_pred_dirs = [mednext_pred_path / f'fold_{i}' / 'validation_raw' for i in range(5)]

    input_img_dir = Path("/home/v363/v363397/stay/brats2024/data/MICCAI-BraTS2024-PED/BraTS-PEDs2024_Training")
    ensembled_dir = Path("/home/v363/v363397/stay/brats2024/Task099a_postprocessed_cv/PEDs/ensembled_preds_cv")

    weights = [0.330911177, 0.330839468, 0.338249355]

    batch_ped_ensemble(
        swinunetr_pred_dirs,
        nnunet_pred_dirs,
        mednext_pred_dirs,
        input_img_dir,
        ensembled_dir,
        weights=weights,
        cv=True
    )


def main_val():
    """
    Executes ensemble predictions for validation mode.
    """
    swinunetr_pred_path = Path("/home/v363/v363397/media/output_val/ped_stratified")
    swinunetr_pred_dirs = [
        swinunetr_pred_path / f'swinunetr_e650_f{i}_b1p4' for i in [0, 1, 2]
    ] + [
        swinunetr_pred_path / f'swinunetr_e1000_f{i}_b1p4' for i in [3, 4]
    ]

    nnunet_pred_path = Path("/home/v363/v363397/stay/brats2024/Task012a_BraTS2024-PED_nnUNet_SS/6_predict/predicted/folds_independent")
    nnunet_pred_dirs = [nnunet_pred_path / f'fold_{i}' / "nnUNetTrainer_200epochs/3d_fullres" for i in range(5)]

    mednext_pred_path = Path("/home/v363/v363397/stay/brats2024/Task012b_BraTS2024-PED_MedNeXt_SS/6_predict/predicted/folds_independent")
    mednext_pred_dirs = [mednext_pred_path / f'fold_{i}' / "nnUNetTrainerV2_MedNeXt_M_kernel3_200epochs_StratifiedSplit/3d_fullres" for i in range(5)]

    input_img_dir = Path("/home/v363/v363397/stay/brats2024/data/MICCAI-BraTS2024-PED/BraTS_Validation_Data_backup")
    ensembled_dir = Path("/home/v363/v363397/stay/brats2024/Task099b_postprocessed_val/PEDs/ensembled_preds_val")

    weights = [0.330911177, 0.330839468, 0.338249355]

    batch_ped_ensemble(
        swinunetr_pred_dirs,
        nnunet_pred_dirs,
        mednext_pred_dirs,
        input_img_dir,
        ensembled_dir,
        weights=weights,
    )


if __name__ == '__main__':
    # Uncomment the desired mode to run
    # main_cv()
    main_val()
