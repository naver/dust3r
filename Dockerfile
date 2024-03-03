FROM nvcr.io/nvidia/pytorch:24.01-py3
ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /workdir/requirements.txt
RUN pip install -r /workdir/requirements.txt

RUN mkdir /workdir/checkpoints/ 
RUN wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P /workdir/checkpoints/

COPY croco /workdir/croco
RUN cd /workdir/croco/models/curope/ ; python setup.py build_ext --inplace

WORKDIR /workdir
COPY demo.py /workdir/demo.py
COPY dust3r /workdir/dust3r
RUN pip install opencv-python==4.8.0.74
CMD python3 /workdir/demo.py --weights /workdir/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --server_name 0.0.0.0

# Use --image_size to select the correct resolution for your checkpoint. 512 (default) or 224
# Use --local_network to make it accessible on the local network, or --server_name to specify the url manually
# Use --server_port to change the port, by default it will search for an available port starting at 7860
# Use --device to use a different device, by default it's "cuda"


#wget progress...
# --no-verbose --show-progress --progress=dot:mega
