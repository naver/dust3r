FROM python:3.11-slim

LABEL description="Docker container for DUSt3R with dependencies installed. CPU VERSION"

ENV DEVICE="cpu"
ENV MODEL="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
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

RUN git clone --recursive https://github.com/naver/dust3r /dust3r
WORKDIR /dust3r

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt
RUN pip install opencv-python==4.8.0.74

WORKDIR /dust3r

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
