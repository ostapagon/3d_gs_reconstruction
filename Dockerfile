FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

ARG COLMAP_GIT_COMMIT=main

#80 for L4 video card, for more information visit: https://developer.nvidia.com/cuda-gpus
ARG CUDA_ARCHITECTURES=75
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \ 
    libsm6 \
    libxext6 \
    ffmpeg \
    libxrender-dev \
    zip \
    unzip \
    git \
    openssh-client \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-dev

RUN git config --global --add safe.directory /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

RUN python3 -m pip install --upgrade pip setuptools cmake

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m pip install --no-cache-dir --upgrade pip

# COPY requirements.txt /app/requirements.txt 
# RUN pip install -r /app/requirements.txt
COPY submodules/Depth-Anything/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install Cython

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
ENV PATH=$PATH:${CUDA_HOME}/bin

RUN git clone https://github.com/NVlabs/nvdiffrast.git
RUN cd /app/nvdiffrast && pip install .


RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

COPY . .

RUN cd /app/submodules/co3d && pip install -e .

RUN pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt200/download.html

# RUN cd /app/submodules/gaussian-splatting/submodules/diff-gaussian-rasterization && pip install -e . 
# RUN cd /app/submodules/gaussian-splatting/submodules/simple-knn && pip install -e . 