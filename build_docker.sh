 source ../ISBI2024-BraTS-GOAT/.mlcubes/bin/activate
cd segmenter/mlcube
mlcube configure -Pdocker.build_strategy=always
docker push aparida12/brats-peds-2023:ped-v2