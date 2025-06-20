FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y git vim libgl1 libglib2.0-0 ffmpeg build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

RUN python --version

RUN pip install --upgrade pip && \
    pip install numpy matplotlib opencv-python yacs tqdm path open3d tensorboardX transforms3d

ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH=/data/inference:/data/data_gen:$PYTHONPATH

WORKDIR /workspace
