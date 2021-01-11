FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget unzip cmake git \
    python3-dev python3-pip python3-setuptools \
    build-essential ca-certificates

RUN ln -sf $(which python3) /usr/bin/python && \
    ln -sf $(which pip3) /usr/bin/pip

ENV     LD_LIBRARY_PATH /usr/local/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV     PYTHONIOENCODING=utf-8

ENV     NCCL_ROOT_DIR /usr/local/cuda
ENV     TH_BINARY_BUILD 1
ENV     TORCH_CUDA_ARCH_LIST "3.5;5.0+PTX;5.2;6.0;6.1;7.0;7.5"
ENV     TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV     DESIRED_CUDA 101

RUN     pip3 install torch==1.3.0

RUN     cd /opt && \
        git clone https://github.com/NVIDIA/apex && cd apex && \
        pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
        rm -rf /opt/apex

RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r pip_requirements.txt

COPY . $BUILD_DIR


RUN git clone https://github.com/pytorch/fairseq.git && \
    cd fairseq && \
    git checkout 2b68e91f231a2b7997664e1418f30b808d889963 && \
    pip install --editable . && \
    wget https://dl.fbaipublicfiles.com/unlikelihood/wikitext-103-bpe_v0.tar.gz && \
    tar -xzvf wikitext-103-bpe_v0.tar.gz && \
    mv wikitext-103-bpe_v0 data-bin/ && \
    cd ..
    
RUN git clone https://github.com/facebookresearch/unlikelihood_training.git && \
    export FAIRSEQ_DIR=fairseq && \
    export UNLIKELIHOOD_DIR=unlikelihood_training && \
    cp -r $UNLIKELIHOOD_DIR/custom $FAIRSEQ_DIR/fairseq

    
ENV NAME World
