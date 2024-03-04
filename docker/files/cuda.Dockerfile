FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

LABEL description="Docker container for DUSt3R with dependencies installed. CUDA VERSION"

ENV DEBIAN_FRONTEND=noninteractive
ENV DEVICE="cpu"

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    # Required for Anaconda
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh 

# Clone the dust3r repository and its submodules
RUN git clone --recursive https://github.com/naver/dust3r /dust3r

WORKDIR /dust3r

RUN conda create -y -n dust3r python=3.11 cmake=3.14.0 \
    && echo "source activate dust3r" > ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

RUN conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

RUN pip install -r requirements.txt

# Download pre-trained model
RUN mkdir -p checkpoints/ \
    && wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["--local_network"]
