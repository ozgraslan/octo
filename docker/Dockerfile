FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /project

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    software-properties-common\
    build-essential\
    libspdlog-dev\
    libeigen3-dev lsb-release ffmpeg libsm6 libxext6 zstd\
    wget \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git \
    ffmpeg \
    libvulkan1 \
    vulkan-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniconda 
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH /root/miniconda3/bin:$PATH

# create conda environment
RUN conda create -n "robot" python=3.10
SHELL ["conda", "run", "-n", "robot", "/bin/bash", "-c"]

# install the zed sdk
ARG UBUNTU_RELEASE_YEAR=22
ARG ZED_SDK_MAJOR=4
ARG ZED_SDK_MINOR=0
ARG CUDA_MAJOR=12
ARG CUDA_MINOR=1

RUN echo "Europe/Paris" > /etc/localtime ; echo "CUDA Version ${CUDA_MAJOR}.${CUDA_MINOR}.0" > /usr/local/cuda/version.txt

# setup the ZED SDK
RUN apt-get update -y || true ; apt-get install --no-install-recommends lsb-release wget less udev sudo zstd build-essential cmake python3 python3-pip libpng-dev libgomp1 -y && \ 
    python3 -m pip install numpy opencv-python && \
    wget -q -O ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/cu${CUDA_MAJOR}${CUDA_MINOR%.*}/ubuntu${UBUNTU_RELEASE_YEAR} && \
    chmod +x ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run && \
    ./ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run -- silent skip_tools skip_cuda && \
    ln -sf /lib/x86_64-linux-gnu/libusb-1.0.so.0 /usr/lib/x86_64-linux-gnu/libusb-1.0.so && \
    rm ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run && \
    rm -rf /var/lib/apt/lists/*
RUN conda install -c conda-forge libstdcxx-ng requests # required for pyzed
RUN python /usr/local/zed/get_python_api.py && python -m pip install --ignore-installed pyzed


# Install octo/SimplerEnv/droid python requirements
     
# RUN pip install tensorflow==2.15.0 
RUN pip install numpy==1.24.4 
RUN pip install tensorflow[and-cuda]==2.15.1 
RUN pip install git+https://github.com/nathanrooy/simulated-annealing 
RUN pip install debugpy
RUN pip install dm-robotics-moma==0.5.0 --no-deps  
RUN pip install dm-robotics-transformations==0.5.0 --no-deps 
RUN pip install dm-robotics-agentflow==0.5.0 --no-deps 
RUN pip install dm-robotics-geometry==0.5.0 --no-deps 
RUN pip install dm-robotics-manipulation==0.5.0 --no-deps 
RUN pip install dm-robotics-controllers==0.5.0 --no-deps 

# # Upgrade pip so we can install SimplerEnv later
RUN pip install --upgrade pip

RUN wget https://raw.githubusercontent.com/simpler-env/SimplerEnv/refs/heads/main/requirements_full_install.txt 
RUN wget https://raw.githubusercontent.com/simpler-env/ManiSkill2_real2sim/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37/requirements.txt
RUN pip install -r requirements_full_install.txt && rm requirements_full_install.txt 
RUN pip install -r requirements.txt && rm requirements.txt
RUN pip install -U "jax[cuda12]"