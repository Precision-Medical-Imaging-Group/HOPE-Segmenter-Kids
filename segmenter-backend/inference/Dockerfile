FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install gcc
RUN apt-get -y install nano
COPY ./weights/swinunetr_peds_trunc.zip /mlcube_project/weights/
COPY ./weights/BraTS2024_PEDs_MedNeXt_model.zip /mlcube_project/weights/
COPY ./weights/BraTS2024_PEDs_nnunetv2_model.zip /mlcube_project/weights/

COPY ./kmeans-cluster-artifacts /mlcube_project/kmeans-cluster-artifacts
COPY ./requirements.txt /mlcube_project/requirements.txt
RUN pip3 install --no-cache-dir -r /mlcube_project/requirements.txt
COPY ./install_nnunet.sh /mlcube_project/install_nnunet.sh
COPY ./install_mednext.sh /mlcube_project/install_mednext.sh
RUN bash /mlcube_project/install_nnunet.sh
RUN bash /mlcube_project/install_mednext.sh
#RUN pip3 install --upgrade --force-reinstall numpy

COPY ./mlcube /mlcube_project/mlcube
COPY ./ensembler /mlcube_project/ensembler
COPY ./nnunet /mlcube_project/nnunet
COPY ./mednext /mlcube_project/mednext
COPY ./postproc /mlcube_project/postproc
COPY ./pp_cluster /mlcube_project/pp_cluster
COPY ./radiomics /mlcube_project/radiomics
COPY ./swinunetr /mlcube_project/swinunetr

# MODIFY here
# mlcube>> mlcube.yaml>>L11
COPY ./runner_ped.py /mlcube_project/runner_ped.py
COPY ./mlcube.py /mlcube_project/mlcube.py
COPY ./weights/name.json /mlcube_project/weights/name.json

ENV LANG C.UTF-8
WORKDIR /mlcube_project/
ENTRYPOINT ["python3", "mlcube.py"]