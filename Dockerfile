FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV TERM=xterm-256color
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post2 https://github.com/camenduru/wheels/releases/download/torch-2.5.0-cu124/flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl && \
    pip install opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install torchsde einops diffusers transformers accelerate && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone -b dev https://github.com/camenduru/ComfyUI-MochiWrapper /content/ComfyUI/custom_nodes/ComfyUI-MochiWrapper && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/kijai/ComfyUI-KJNodes /content/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /content/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/ciri/comfyui-model-downloader /content/ComfyUI/custom_nodes/comfyui-model-downloader && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kijai/Mochi_preview_comfy/resolve/main/mochi_preview_dit_bf16.safetensors -d /content/ComfyUI/models/diffusion_models/mochi -o mochi_preview_dit_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kijai/Mochi_preview_comfy/resolve/main/mochi_preview_vae_bf16.safetensors -d /content/ComfyUI/models/vae/mochi -o mochi_preview_vae_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/model.safetensors -d /content/ComfyUI/models/clip -o google_t5-v1_1-xxl_encoderonly-fp16.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py