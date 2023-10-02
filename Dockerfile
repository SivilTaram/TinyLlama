FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

USER root:root

ARG IMAGE_NAME=None
ARG BUILD_NUMBER=None

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends &&\
    # Others
    apt-get install -y \
    libksba8 \
    openssl \
    libssl3 \
    libaio-dev \
    git \
    wget && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* 

ENV MINICONDA_VERSION py310_23.3.1-0
ENV PATH /opt/miniconda/bin:$PATH
ENV CONDA_PACKAGE 23.5.0
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda install -n base conda-libmamba-solver -y && \
    conda config --set solver libmamba && \
    conda clean -ay && \
    conda install conda=${CONDA_PACKAGE} -y && \
    conda install wheel=0.38.1 setuptools=65.5.1 cryptography=41.0.3 requests=2.31.0 -c conda-forge -y && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Install Pytorch 2.0.1
# RUN conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
#     conda clean -ay && \
#     rm -rf /opt/miniconda/pkgs && \
#     find / -type d -name __pycache__ | xargs rm -rf

# RUN pip install deepspeed==0.10.2
RUN pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
RUN pip uninstall ninja -y && pip install ninja -U
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
RUN git clone https://github.com/Dao-AILab/flash-attention
RUN cd flash-attention && \
    python setup.py install && \
    cd csrc/rotary && pip install . && \
    cd ../layer_norm && pip install . && \
    cd ../xentropy && pip install . && \ 
    cd ../.. && rm -rf flash-attention
RUN pip install bitsandbytes==0.40.0 transformers==4.31.0 peft==0.4.0 accelerate==0.21.0 einops==0.6.1 evaluate==0.4.0 scikit-learn==1.2.2 sentencepiece==0.1.99 wandb==0.15.3 tokenizers
RUN pip install git+https://github.com/Lightning-AI/lightning@master jsonargparse[signatures] pandas pyarrow tokenizers wandb zstd
