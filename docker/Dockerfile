FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /project

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    wget \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git \
    ffmpeg \
    libnvidia-gl-560 \
    libvulkan1 \
    vulkan-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install octo/SimplerEnv python requirements
RUN wget https://raw.githubusercontent.com/simpler-env/SimplerEnv/refs/heads/main/requirements_full_install.txt \
    && pip install tensorflow==2.15.0 \
    && pip install -r requirements_full_install.txt && rm requirements_full_install.txt \
    && pip install tensorflow[and-cuda]==2.15.1 \
    && pip install git+https://github.com/nathanrooy/simulated-annealing \
    && pip install numpy==1.24.4 \
    && pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip install debugpy

# Upgrade pip so we can install SimplerEnv later
RUN pip3 install -U pip 
