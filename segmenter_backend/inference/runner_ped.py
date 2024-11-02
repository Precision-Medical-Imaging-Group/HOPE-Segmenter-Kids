from pathlib import Path
import os
import shutil
import tempfile
from typing import Tuple, List, Dict, Any

from nnunet.install_model import install_model_from_zip
from ensembler.ped_weighted_ensemble import ped_ensemble
from nnunet.runner import run_infer_nnunet
from mednext.runner import run_infer_mednext
from swinunetr.runner import run_infer_swinunetr
from pp_cluster.infer import get_cluster, get_cluster_artifacts
from radiomics.feature_extraction_v2 import extract_all, save_json, load_json
from postproc.postprocess_cc import remove_small_component
from postproc.postprocess_lblredef import label_redefinition
import pandas as pd

# Task identifier
TASK: str = 'BraTS-PED'

# Read the weights_cv.json
json_path: Path = Path('./ensembler/weights_cv.json')
ENSEMBLE_WEIGHTS: Dict[str, float] = load_json(json_path)[TASK]
NAME_MAPPER: Dict[str, str] = load_json(Path('./weights/name.json'))
CLUSTER_ARTIFACT: Any = get_cluster_artifacts(TASK)
THRESHOLD_FILE_CC: Path = Path('./postproc/cc_thresholds_cv.json')
THRESHOLD_FILE_LBLDEF: Path = Path('./postproc/lblredef_thresholds_cv.json')

# Configuration constants
CONSTANTS: Dict[str, str] = {
    'pyradiomics_paramfile': './radiomics/params.yaml',
    'nnunet_model_path': './weights/BraTS2024_PEDs_nnunetv2_model.zip',
    'mednext_model_path': './weights/BraTS2024_PEDs_MedNeXt_model.zip',
    'swinunetr_model_path': './weights/swinunetr_peds_trunc.zip',
}


def maybe_make_dir(path: Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path (Path): The path of the directory to create.

    Returns:
        Path: The path to the created directory.
    """
    os.makedirs(path, exist_ok=True)
    return Path(path)


def lbl_redefination(
    task: str,
    threshold_file: Path,
    input_dir: Path,
    df_radiomics: pd.DataFrame,
    out_dir: Path
) -> None:
    """Perform label redefinition by copying files from input to output directory.

    Args:
        task (str): The task identifier.
        threshold_file (Path): Path to the threshold file for label redefinition.
        input_dir (Path): Directory containing input files.
        df_radiomics (pd.DataFrame): DataFrame containing radiomics data.
        out_dir (Path): Directory to store output files.
    """
    # Copy files from input_dir to out_dir
    for file in input_dir.iterdir():
        shutil.copy(file, out_dir)


def postprocess_single(
    input_dir: Path,
    seg_dir: Path,
    out_dir: Path
) -> Path:
    """Postprocess a single segmentation case.

    This function performs radiomics extraction, cluster determination, removes small components,
    and redefines labels based on the processed data.

    Args:
        input_dir (Path): Directory containing the input data.
        seg_dir (Path): Directory containing the segmentation results.
        out_dir (Path): Directory to store the postprocessed segmentation.

    Returns:
        Path: Path to the postprocessed segmentation file.
    """
    case_name: str = input_dir.name
    seg_path: Path = seg_dir / f"{case_name}.nii.gz"
    out_path: Path = maybe_make_dir(out_dir) / f"{case_name}.nii.gz"

    # Compute radiomics features
    try:
        df_radiomics: pd.DataFrame = extract_all(
            CONSTANTS['pyradiomics_paramfile'],
            input_dir.parent,
            case_name,
            seg_path,
            1,
            region='wt',
            tmpp='/tmp/',
            seg_suffix='',
            sequences=['-t1n', '-t1c', '-t2w', '-t2f']
        )
        # Determine the cluster based on radiomics features
        cluster: int = get_cluster(df_radiomics, CLUSTER_ARTIFACT)[0]
    except Exception as e:
        print(f"Error in radiomics extraction: {e}")
        df_radiomics = pd.DataFrame([{"StudyID": case_name}])
        cluster = 4

    df_radiomics['cluster'] = int(cluster)

    # Remove disconnected regions from segmentation
    removed_cc_dir: Path = maybe_make_dir(seg_dir / 'remove_cc')
    remove_small_component(
        TASK,
        THRESHOLD_FILE_CC,
        seg_dir,
        df_radiomics,
        removed_cc_dir
    )

    # Redefine labels based on the postprocessed segmentation
    label_redefinition(
        TASK,
        THRESHOLD_FILE_LBLDEF,
        removed_cc_dir,
        df_radiomics,
        out_dir
    )

    return out_path


def infer_single(
    input_path: Path,
    out_dir: Path
) -> None:
    """Perform inference on a single input directory.

    This function sets up a temporary directory, copies input files, runs inference models,
    ensembles the predictions, and postprocesses the segmentation.

    Args:
        input_path (Path): Input directory containing the 4 nii.gz files.
        out_dir (Path): Output directory to store the segmentation results.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir: Path = Path(tmpdirname)
        print(f'Storing artifacts in temporary directory {temp_dir}')

        input_folder_raw: Path = maybe_make_dir(temp_dir / 'inp')
        name: str = input_path.name
        print(f'Processing case: {name}')

        shutil.copytree(input_path, input_folder_raw / name)

        # Rename files based on NAME_MAPPER
        for key, val in NAME_MAPPER.items():
            src: Path = input_folder_raw / name / f'{name}{key}'
            dest: Path = input_folder_raw / name / f'{name}{val}'
            os.rename(src, dest)
            one_image: Path = dest

        # Run inference using nnUNet
        nnunet_npz_path_list: List[Path] = run_infer_nnunet(
            input_folder_raw / name,
            maybe_make_dir(temp_dir / 'nnunet'),
            TASK,
            name,
            ensemble=False
        )

        # Run inference using MedNeXt
        mednext_npz_path_list: List[Path]
        mednext_pkl_path_list: List[Path]
        mednext_npz_path_list, mednext_pkl_path_list = run_infer_mednext(
            input_folder_raw / name,
            maybe_make_dir(temp_dir / 'mednext'),
            TASK,
            name,
            ensemble=False
        )

        # Run inference using SwinUNETR
        swinunetr_npz_path_list: List[Path] = run_infer_swinunetr(
            Path(input_path),
            maybe_make_dir(temp_dir / 'swinunetr'),
            TASK
        )

        # Ensemble the predictions
        ensemble_folder: Path = maybe_make_dir(input_folder_raw / 'ensemble')
        ensembled_pred_nii_path: Path = ped_ensemble(
            swinunetr_npz_path_list,
            nnunet_npz_path_list,
            mednext_npz_path_list,
            mednext_pkl_path_list,
            ensemble_folder,
            one_image,
            weights=[
                ENSEMBLE_WEIGHTS['SwinUNETR'],
                ENSEMBLE_WEIGHTS['nnUNetv2'],
                ENSEMBLE_WEIGHTS['MedNeXt']
            ]
        )

        # Postprocess the ensembled segmentation
        radiomics: Path = postprocess_single(
            input_path,
            ensemble_folder,
            out_dir
        )


def batch_processor(
    input_folder: str,
    output_folder: str
) -> None:
    """Process a batch of input directories for inference.

    Args:
        input_folder (str): Path to the folder containing input directories.
        output_folder (str): Path to the folder to store output segmentations.
    """
    for input_path in Path(input_folder).iterdir():
        infer_single(input_path, Path(output_folder))


def setup_model_weights() -> None:
    """Install model weights from zip files."""
    install_model_from_zip(CONSTANTS['nnunet_model_path'])
    install_model_from_zip(CONSTANTS['swinunetr_model_path'])
    install_model_from_zip(CONSTANTS['mednext_model_path'], mednext=True)


def batch_postprocess(
    input_folder: str,
    seg_folder: str,
    output_folder: str
) -> int:
    """Postprocess a batch of segmentation results.

    Args:
        input_folder (str): Path to the folder containing input directories.
        seg_folder (str): Path to the folder containing segmentation results.
        output_folder (str): Path to the folder to store postprocessed results.

    Returns:
        int: Number of processed cases.
    """
    l_radiomics: List[Path] = []
    l_path: List[Path] = [x for x in Path(input_folder).iterdir() if x.is_dir()]
    
    for input_path in l_path:
        radiomics: Path = postprocess_single(input_path, Path(seg_folder), Path(output_folder))
        l_radiomics.append(radiomics)
    
    save_json(output_folder / 'val_radiomics.json', l_radiomics)
    return len(list(Path(input_folder).iterdir()))


if __name__ == "__main__":
    setup_model_weights()
    batch_processor('./ins', './outs')
