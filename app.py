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
    docker = "aparida12/brats-peds-2024:v20240827"
    mlcube_cmd =f"docker run --shm-size=2gb --gpus=all -v {input_path.parent}:/input/ -v {output_folder.absolute()}:/output {docker} infer --data_path /input/ --output_path /output/"
    #mlcube_cmd = f"cd ./segmenter/mlcube; mlcube run --gpus device=1 --task infer data_path={input_path}/ output_path=../outs"
    print(mlcube_cmd)
    subprocess.run(mlcube_cmd, shell=True)
    subprocess.run(f"mv -f {fake_output_path} {output_path}", shell=True)

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

def render_slice(img, mask, x, view):
    if view == 'axial':
        slice_img, slice_mask = img[x, :, :], mask[x, :, :]
    elif view == 'coronal':
        slice_img, slice_mask = img[:, x, :], mask[:, x, :]
    elif view == 'sagittal':
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
    total_sum = 0
    for i, lbl in enumerate(unique):
        ml_vol = multiplier_ml * frequency[i]
        mydict[f'vol_lbl{int(lbl)}'] = ml_vol
        if lbl != 0:
            total_sum += ml_vol

        mydict[f'vol_total'] = total_sum
    return mask_path, f"Segmentation done! Total tumor volume segmented {mydict.get('vol_total', 0):.3f} ml; EDEMA(ED) {mydict.get('vol_lbl4', 0):.3f} ml; ENHANCING TUMOR(ET) {mydict.get('vol_lbl1', 0):.3f} ml; NON-ENHANCING TUMOR CORE(NETC) {mydict.get('vol_lbl2', 0):.3f} ml; CYSTIC COMPONENT(CC) {mydict.get('vol_lbl3', 0):.3f} ml"


def render(file_to_render, x, view):
    suffix = {'T2 Flair': 't2f', 'native T1': 't1n', 'post-contrast T1-weighted': 't1c', 'T2 weighted': 't2w'}
    if 'img_path' in mydict:
        get_file = [file for file in mydict['img_path'] if suffix[file_to_render] in file][0]
        img, _, mask = get_img_mask(get_file, mydict['mask_path'])
        
        x = max(0, min(x, img.shape[0 if view == 'axial' else (1 if view == 'coronal' else 2)] - 1))
        slice_img, slice_mask = render_slice(img, mask, x, view)
        
        im = PIL.Image.fromarray(slice_img.astype(np.uint8))
        annotations = [
            (slice_mask == 1, f"ET: {mydict.get('vol_lbl1', 0):.3f} ml"),
            (slice_mask == 2, f"NETC: {mydict.get('vol_lbl2', 0):.3f} ml"),
            (slice_mask == 3, f"CC: {mydict.get('vol_lbl3', 0):.3f} ml"),
            (slice_mask == 4, f"ED: {mydict.get('vol_lbl4', 0):.3f} ml")
        ]
        return im, annotations
    else:
        return np.zeros((10, 10)), []

def render_axial(file_to_render, x):
    return render(file_to_render, x, 'axial')
def render_coronal(file_to_render, x):
    return render(file_to_render, x, 'coronal')
def render_sagittal(file_to_render, x):
    return render(file_to_render, x, 'sagittal')
# Gradio UI
with gr.Blocks() as demo:

    gr.HTML(value=f"<center><font size='6'><bold> Children's National Pediatric Brain Tumor Segmenter</bold></font></center>")
    gr.HTML(value=f"<p style='margin-top: 1rem, margin-bottom: 1rem'> <img src='{logo.logo}' alt='Childrens National Logo' style='display: inline-block'/></p>")
    # gr.HTML(value=f"<justify><font size='4'> Welcome to the pediatric brain tumor segmentation app that won the prestigious <a href='https://www.synapse.org/Synapse:syn51156910/wiki/627802'>Pediatric Brain Tumor Segmentation Challenge(BraTS) 2023</a>! Our advanced segmentation model, recognized for its exceptional accuracy and reliability, is designed to automate the early detection and precise segmentation of brain tumors in pediatric patients. With this app, you can effortlessly upload pediatric brain MRI sequences and receive detailed, accurate segmentation results in just minutes. The idea is to simplify the analysis process by providing an web based interaction with the segmentation model.</font></justify>")
    # gr.HTML(value=f"<justify><font size='4'> We also provide couple of samples at the bottom of the page for you see the performance of the model. To stay updated with the different model updates sign up <a href='https://forms.gle/e634eJzoimhHnJ7W9'>here</a>. If you would like to receive a dockerized version of the model reach out to <a href='mailto:mlingura@childrensnational.org'>Marius George Linguraru</a> with details how you would like to use the docker container.</font></justify>")
    gr.HTML(value=f"<justify><font size='4'> Welcome to the pediatric brain tumor segmenter. Please read the <a href='https://precision-medical-imaging-group.github.io/SegmenterApp-Segmenter-Peds'>instructions</a> before using the application. </font></justify>")
    with gr.Row():
        image_t1c = gr.File(label="upload t1 contrast enhanced here:", file_types=["nii.gz"])
        image_t2f = gr.File(label="upload t2 flair here:", file_types=["nii.gz"])
        image_t1n = gr.File(label="upload t1 pre-contrast here:", file_types=["nii.gz"])
        image_t2w = gr.File(label="upload t2 weighted here:", file_types=["nii.gz"])
    with gr.Row():
        with gr.Column():
             gr.Button("", render=False)
        with gr.Column():
            btn = gr.Button("Start Segmentation")
        with gr.Column():
             gr.Button("", render=False)
    with gr.Column():
        out_text = gr.Textbox(label='Status', placeholder="Volumetrics will be updated here.")

    with gr.Row():
        with gr.Column():
             gr.Button("", render=False)
        with gr.Column():
            file_to_render = gr.Dropdown(['T2 Flair','native T1', 'post-contrast T1-weighted', 'T2 weighted'], label='choose the scan to overlay the segmentation on')
        with gr.Column():
             gr.Button("", render=False)

    with gr.Row():
        height = "20vw"
        myimage_axial = gr.AnnotatedImage(label="axial view", height=height)
        myimage_coronal = gr.AnnotatedImage(label="coronal view",height=height)
        myimage_sagittal = gr.AnnotatedImage(label="sagittal view",height=height)
    with gr.Row():
        slider_axial = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_axial = gr.State(value=75)
        slider_coronal = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_coronal = gr.State(value=75)
        slider_sagittal = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_sagittal = gr.State(value=75)

    with gr.Row():
        mask_file = gr.File(label="Download Segmentation File", height="vw" )

    example_dir = '/home/pmilab/Abhijeet/examples/'
    generate_examples = glob.glob(example_dir + '*')
    order_list = ['-t1c.nii.gz', '-t2f.nii.gz', '-t1n.nii.gz', '-t2w.nii.gz']
    example_list = [[os.path.join(path, str(Path(path).name)+ending) for ending in order_list] for path in generate_examples]

    
    # gr.HTML(value=f"<center><font size='2'> The software is provided 'as is', without any warranties or liabilities.  For research use only and not intended for medical diagnosis. We do not store or access any information uploaded to the platform. This version is v20240827.</font></center>")
    gr.Examples(
        examples=example_list,
        inputs=[image_t1c, image_t2f, image_t1n, image_t2w],
        outputs=[mask_file,out_text],
        fn=main_func,
        cache_examples=False,
        label="Preloaded BraTS 2023 examples"
    )
    
    btn.click(fn=main_func, 
        inputs=[image_t1c, image_t2f, image_t1n, image_t2w], outputs=[mask_file, out_text],
    )
    file_to_render.select(render_axial,
        inputs=[file_to_render, state_axial],
        outputs=[myimage_axial])
    file_to_render.select(render_coronal,
        inputs=[file_to_render, state_coronal],
        outputs=[myimage_coronal])
    file_to_render.select(render_sagittal,
        inputs=[file_to_render, state_sagittal],
        outputs=[myimage_sagittal])

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
    slider_sagittal.change(
        render_sagittal,
        inputs=[file_to_render, slider_sagittal],
        outputs=[myimage_sagittal],
        api_name="hohoho"
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")
