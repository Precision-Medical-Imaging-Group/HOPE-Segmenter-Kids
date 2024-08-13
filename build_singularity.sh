
docker_img=aparida12/brats-peds-2023:ped
img_tag=ped_runner
inp_folder=/media/abhijeet/DataThunder1/BraTS2024-SSA-Challenge-ValidationData/test
out_folder=/media/abhijeet/DataThunder1/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/test_outs
singularity pull "${img_tag}.sif" "docker://${docker_img}"
singularity build --sandbox "${img_tag}" "docker://${docker_img}"
mkdir "${img_tag}/input/"
mkdir "${img_tag}/output/"
singularity exec --tmp-sandbox --nv -w --bind  "${inp_folder}:/input/" --bind "${out_folder}:/output" --pwd "${pwd}/mlcube_project/" "${img_tag}" python3 mlcube.py infer --data_path /input/ --output_path /output/