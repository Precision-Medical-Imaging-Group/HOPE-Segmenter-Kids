import logging
import subprocess
import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import PIL
import gradio as gr
import os
from app_assets import logo

logger = logging.getLogger(__file__)
# Constants

# Always use dumy name to maintain annonimity 
DUMMY_DIR = "BraTS-PED-00019-000"
DUMMY_FILE_NAMES = {modality : f"{DUMMY_DIR}-{modality}.nii.gz" for modality in ["t1c", "t2f", "t1n", "t2w"]}


mydict = {}

def run_inference(image_t1c, image_t2f, image_t1n, image_t2w):
    """Run inference on the given image paths

    Args:
        image_paths (list): List of image paths

    Returns:
        path.like, path.like: inpuit path, output path
    """
    image_paths = {
        "t1c": image_t1c, 
        "t2f": image_t2f, 
        "t1n": image_t1n, 
        "t2w":image_t2w}
    input_path = Path(f'/tmp/peds-app/{DUMMY_DIR}')
    os.makedirs(input_path, exist_ok=True)
    output_folder = Path('./segmenter/mlcube/outs')
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder / f'seg_{Path(image_t1c).name}'
    fake_output_path = output_folder / f'{DUMMY_DIR}.nii.gz'
    # Create directories and move files
    
    for key, file in image_paths.items():
        subprocess.run(f"cp {file} {input_path}/{DUMMY_FILE_NAMES[key]}", shell=True)
    # delete original files
    for _, file in image_paths.items():
        subprocess.run(f"rm  {file}", shell=True)

    mlcube_cmd =f"docker run --shm-size=2gb --gpus=all -v {input_path.parent}:/input/ -v {output_folder.absolute()}:/output aparida12/brats-peds-2023:ped infer --data_path /input/ --output_path /output/"
    #mlcube_cmd = f"cd ./segmenter/mlcube; mlcube run --gpus device=1 --task infer data_path={input_path}/ output_path=../outs"
    print(mlcube_cmd)
    subprocess.run(mlcube_cmd, shell=True)
    subprocess.run(f"mv {fake_output_path} {output_path}", shell=True)

    return str(input_path), str(output_path)

def get_img_mask(image_path, mask_path):
    """_summary_

    Args:
        image_path (_type_): _description_
        mask_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    img_obj = sitk.ReadImage(image_path)
    mask_obj = sitk.ReadImage(mask_path)
    img = sitk.GetArrayFromImage(img_obj)
    mask = sitk.GetArrayFromImage(mask_obj)
    
    # Normalize image
    minval, maxval = np.min(img), np.max(img)
    img = ((img - minval) / (maxval - minval)).clip(0, 1) * 255

    return img, img_obj, mask

def calculate_volumes(mask, spacing_tuple):
    multiplier_ml = 0.001 * np.prod(spacing_tuple)
    unique, frequency = np.unique(mask, return_counts=True)
    volumes = {f'vol_lbl{lbl}': multiplier_ml * freq for lbl, freq in zip(unique, frequency)}
    return volumes


def render_slice(img, mask, x, view):
    if view == 'axial':
        slice_img, slice_mask = img[x, :, :], mask[x, :, :]
    elif view == 'coronal':
        slice_img, slice_mask = img[:, x, :], mask[:, x, :]
    elif view == 'saggital':
        slice_img, slice_mask = img[:, :, x], mask[:, :, x]
    
    slice_img = np.flipud(slice_img)
    slice_mask = np.flipud(slice_mask)
    return slice_img, slice_mask


def main_func(image_t1c, image_t2f, image_t1n, image_t2w):
    print(image_t1c)
    global mydict
    image_path = image_t1c
    image_path, mask_path = run_inference(Path(image_t1c), Path(image_t2f),Path(image_t1n), Path(image_t2w))
    image_path = glob.glob(image_path+'/*.nii.gz')
    mydict['img_path'] = image_path
    mydict['mask_path'] = mask_path
    img, img_obj, mask = get_img_mask(image_path[0], mask_path)
    
    mydict['img'] = img.astype(np.uint8)
    mydict['mask'] = mask.astype(np.uint8)

    print(img_obj.GetSpacing())
    spacing_tuple = img_obj.GetSpacing()
    multiplier_ml = 0.001 * spacing_tuple[0] * spacing_tuple[1] * spacing_tuple[2]
    unique, frequency = np.unique(mask, return_counts = True)
    for i, lbl in enumerate(unique):
        mydict[f'vol_lbl{lbl}'] = multiplier_ml * frequency[i]


    return mask_path, f"Segmentation done! Total tumor volume segmented {mydict.get('vol_lbl3', 0) + mydict.get('vol_lbl2', 0) + mydict.get('vol_lbl1', 0):.3f} ml; EDEMA {mydict.get('vol_lbl2', 0):.3f} ml; NECROSIS {mydict.get('vol_lbl3', 0):.3f} ml; ENHANCING TUMOR {mydict.get('vol_lbl1', 0):.3f} ml"

def main_func_example(file_obj):
    print(file_obj)
    mask_path, _ = main_func(file_obj)
    x, state = 10, 10
    im, another_output_text = render(x, state)
    return mask_path, im, another_output_text

def render(file_to_render, x, view):
    suffix = {'T2 Flair': 't2f', 'native T1': 't1n', 'post-contrast T1-weighted': 't1c', 'T2 weighted': 't2w'}
    if 'img_path' in mydict:
        get_file = [file for file in mydict['img_path'] if suffix[file_to_render] in file][0]
        img, _, mask = get_img_mask(get_file, get_file)
        
        x = max(0, min(x, img.shape[0 if view == 'axial' else (1 if view == 'coronal' else 2)] - 1))
        slice_img, slice_mask = render_slice(img, mask, x, view)
        
        im = PIL.Image.fromarray(slice_img.astype(np.uint8))
        annotations = [
            (slice_mask == 1, f"enhancing tumor: {mydict.get('vol_lbl1', 0):.3f} ml"),
            (slice_mask == 2, f"edema: {mydict.get('vol_lbl2', 0):.3f} ml"),
            (slice_mask == 3, f"necrosis: {mydict.get('vol_lbl3', 0):.3f} ml")
        ]
        return im, annotations
    else:
        return np.zeros((10, 10)), []

def render_axial(file_to_render, x):
    return render(file_to_render, x, 'axial')
def render_coronal(file_to_render, x):
    return render(file_to_render, x, 'coronal')
def render_saggital(file_to_render, x):
    return render(file_to_render, x, 'saggital')
# Gradio UI
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
        """
        # CNMC PMI Pediatric Brain Tumor Segmentation
        """)
    gr.HTML(value=f"<p style='margin-top: 1rem, margin-bottom: 1rem'> <img src='{logo.logo}' alt='Childrens National Logo' style='display: inline-block'/></p>")

    with gr.Row():
        image_t1c = gr.File(label="upload t1 contrast enhanced here:", file_types=["nii.gz"])
        image_t2f = gr.File(label="upload t2 flair here:", file_types=["nii.gz"])
        image_t1n = gr.File(label="upload t1 pre-contrast here:", file_types=["nii.gz"])
        image_t2w = gr.File(label="upload t2 weighted here:", file_types=["nii.gz"])
    with gr.Row():
        with gr.Column():
             gr.Button("", render=False)
        with gr.Column():
            btn = gr.Button("start segmentation")
        with gr.Column():
             gr.Button("", render=False)
    with gr.Column():
        out_text = gr.Textbox(label='Status', placeholder="Volumetrics will be updated here.")

    with gr.Row():
        with gr.Column():
             gr.Button("", render=False)
        with gr.Column():
            file_to_render = gr.Dropdown(['T2 Flair','native T1', 'post-contrast T1-weighted', 'T2 weighted'], label='choose file to overlay')
        with gr.Column():
             gr.Button("", render=False)

    with gr.Row():
        height = "20vw"
        myimage_axial = gr.AnnotatedImage(label="axial view", height=height)
        myimage_coronal = gr.AnnotatedImage(label="coronal view",height=height)
        myimage_saggital = gr.AnnotatedImage(label="saggital view",height=height)
    with gr.Row():
        slider_axial = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_axial = gr.State(value=75)
        slider_coronal = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_coronal = gr.State(value=75)
        slider_saggital = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_saggital = gr.State(value=75)

    with gr.Row():
        mask_file = gr.File(label="download annotation", height="vw" )

    example_1 = [[
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t1c.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t2f.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t1n.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t2w.nii.gz",
    ]]
    example_2 =[
    [
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t1c.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t2f.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t1n.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t2w.nii.gz",
    ]]


    gr.HTML(value=f"<center><font size='2'> The software provided 'as is', without any warranty or liability.  For research use only and not intended for medical diagnosis. We do not store or access any information uploaded to the platform. This version is v20240815</font></center>")
    # gr.Examples(
    #     examples=[example_1, example_2],
    #     inputs=[image_folder],
    #     outputs=[mask_file,out_text],
    #     fn=main_func,
    #     cache_examples=False,
    #     label="Preloaded BraTS 2023 examples"
    # )
    
    btn.click(fn=main_func, 
        inputs=[image_t1c, image_t2f, image_t1n, image_t2w], outputs=[mask_file, out_text],
    )
    file_to_render.select(render_axial,
        inputs=[file_to_render, state_axial],
        outputs=[myimage_axial])
    file_to_render.select(render_coronal,
        inputs=[file_to_render, state_coronal],
        outputs=[myimage_coronal])
    file_to_render.select(render_saggital,
        inputs=[file_to_render, state_saggital],
        outputs=[myimage_saggital])

    slider_axial.change(
        render_axial,
        inputs=[file_to_render, slider_axial],
        outputs=[myimage_axial],
        api_name="axial_slider"
    )
    slider_coronal.change(
        render_coronal,
        inputs=[file_to_render, slider_coronal],
        outputs=[myimage_coronal],
        api_name="hohoho"
    )
    slider_saggital.change(
        render_saggital,
        inputs=[file_to_render, slider_saggital],
        outputs=[myimage_saggital],
        api_name="hohoho"
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")