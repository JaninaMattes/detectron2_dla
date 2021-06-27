# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

ENV APP /app

WORKDIR ${APP}

ENV PYTHONUSERBASE ${APP}
ENV PATH "${PYTHONUSERBASE}/bin:$PATH"
ENV User appuser
ENV MPLCONFIGDIR /tmp/

RUN apt-get update -y \
    && apt-get -y install \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Detectron2 prerequisites
RUN pip install --user --no-cache-dir pyyaml==5.1
RUN pip install --user --no-cache-dir cmake
RUN pip install --user --no-cache-dir torchtext==0.9.1
RUN pip install --user --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --user --no-cache-dir cython
RUN pip install --user --no-cache-dir pycocotools

# Install pip requirements
ADD requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Detectron2 - GPU copy
RUN pip install --user --no-cache-dir detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

COPY . .

# download, decompress the data
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NAgGrYoQJwdNXE-nFzx0yEJEoqxMcqh2' -O siemens.zip # exchange
RUN unzip siemens.zip -d datasets
RUN rm -r siemens.zip
RUN ls 'datasets/siemens'

# download pre-trained model
RUN mkdir 'pretrained_models'
RUN wget https://www.dropbox.com/sh/wgt9skz67usliei/AADGw0h1y7K5vO0akulyXm-qa/model_final.pth -P pretrained_models
RUN ls 'pretrained_models'

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd ${User} && chown -R ${User} ${APP}
USER ${User}

# CMD ["gunicorn", "--config", "gunicorn.conf.py", "dla:app"]
CMD ["finetuning/finetune_net_dla.py"] 

ENTRYPOINT ["python3"]
