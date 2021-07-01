<<<<<<< HEAD
FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo unzip ninja-build && \
    rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user torchvision torchtext==0.9.1 tensorboard cython cmake pyyaml==5.1 opencv gdown
RUN pip install --user torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo

# copy files from local repo
COPY configs/ /home/appuser/detectron2_repo/configs
COPY finetune_net_dla.py /home/appuser/detectron2_repo

# download, decompress the training dataset used
RUN gdown https://drive.google.com/uc?id=1NAgGrYoQJwdNXE-nFzx0yEJEoqxMcqh2
RUN unzip siemens.zip -d /home/appuser/detectron2_repo/datasets
RUN rm -r siemens.zip

# download, pretrained models
RUN mkdir /home/appuser/detectron2_repo/pretrained_models
RUN wget https://www.dropbox.com/sh/wgt9skz67usliei/AADGw0h1y7K5vO0akulyXm-qa/model_final.pth -P /home/appuser/detectron2_repo/pretrained_models/

RUN pip install --user 'git+https://github.com/facebookresearch/detectron2.git'

# Command line options
ENTRYPOINT [ "python3", "finetune_net_dla.py" ]  
=======
FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install apt-utils nano build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
	python3-opencv ca-certificates python3-dev git wget sudo unzip ninja-build && \
    rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

# Detectron2 prerequisites
RUN pip install --user torchtext==0.9.1
RUN pip install --user torchvision torchtext==0.9.1 tensorboard cython cmake pyyaml==5.1 gdown
RUN pip install --user torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Copy files from repo
RUN mkdir /home/appuser/detectron2_repo
COPY configs/ /home/appuser/detectron2_repo/configs
COPY finetune_net_dla.py /home/appuser/detectron2_repo

WORKDIR /home/appuser/detectron2_repo

# Install dependencies for Detectron2
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# Detectron2 - CPU copy
RUN python3 -m pip install --user detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

# Development packages
RUN pip install --user pdf2image gunicorn Pillow opencv-python

# download, decompress the training dataset used
RUN mkdir /home/appuser/detectron2_repo/datasets

RUN gdown https://drive.google.com/uc?id=1NAgGrYoQJwdNXE-nFzx0yEJEoqxMcqh2
RUN unzip siemens.zip -d /home/appuser/detectron2_repo/datasets
RUN rm -r siemens.zip

# download, pretrained models
RUN mkdir /home/appuser/detectron2_repo/pretrained_models
RUN wget https://www.dropbox.com/sh/wgt9skz67usliei/AADGw0h1y7K5vO0akulyXm-qa/model_final.pth -P /home/appuser/detectron2_repo/pretrained_models/

# Command line options
ENTRYPOINT [ "python3", "finetune_net_dla.py" ]  

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
>>>>>>> 567fc550c8d3cf649f3fb001887199bcd091070a
