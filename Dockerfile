FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev && \
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
RUN pip install --user torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Copy files from repo
RUN mkdir /home/appuser/detectron2_repo
COPY configs/ /home/appuser/detectron2_repo/configs
COPY finetune_net_dla.py /home/appuser/detectron2_repo

WORKDIR /home/appuser/detectron2_repo

# Install dependencies for Detectron2
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'

# Detectron2 - CPU copy
RUN python -m pip install --user detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

# Development packages
RUN pip install --user pdf2image gunicorn Pillow opencv-python

# download, decompress the training dataset used
RUN mkdir /home/appuser/detectron2/datasets

RUN gdown https://drive.google.com/uc?id=1NAgGrYoQJwdNXE-nFzx0yEJEoqxMcqh2
RUN unzip siemens.zip -d /home/appuser/detectron2/datasets
RUN rm -r siemens.zip

# download, pretrained models
RUN mkdir /home/appuser/detectron2/pretrained_models
RUN wget https://www.dropbox.com/sh/wgt9skz67usliei/AADGw0h1y7K5vO0akulyXm-qa/model_final.pth -P /home/appuser/detectron2/pretrained_models/

# Command line options
ENTRYPOINT [ "python3", "finetune_net_dla.py" ]  

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl