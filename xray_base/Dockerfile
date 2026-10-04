FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU from official index
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Rest of deps from PyPI
RUN pip install --no-cache-dir \
    scikit-learn \
    numpy \
    tqdm \
    pillow \
    adversarial-robustness-toolbox \
    packaging

COPY adv_cli.py attack_core.py helper.py /app/
COPY models /app/models
COPY labeled-chest-xray-images/chest_xray/test /app/labeled-chest-xray-images/chest_xray/test

RUN mkdir -p /app/adversarial-examples

ENTRYPOINT ["python", "adv_cli.py"]
