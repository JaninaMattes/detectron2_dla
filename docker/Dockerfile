FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget \
	sudo unzip ninja-build apt-utils && \
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
RUN pip install --user torchvision torchtext==0.9.1 tensorboard cython cmake pyyaml==5.1 gdown
RUN pip install --user torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# install detectroon2
RUN mkdir /home/appuser/detectron2_repo
RUN git clone https://github.com/facebookresearch/detectron2.git /home/appuser/detectron2_repo
ENV FORCE_CUDA="1"

# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install --user -e /home/appuser/detectron2_repo

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

RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Command line options
ENTRYPOINT [ "python3", "finetune_net_dla.py" ]  
