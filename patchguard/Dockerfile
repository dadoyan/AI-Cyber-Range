FROM python:3.11-slim
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

COPY patchguard/requirements.txt /tmp/patchguard-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
       --index-url ${PYTORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r /tmp/patchguard-requirements.txt

COPY range_device.py /usr/local/lib/python3.11/site-packages/range_device.py
COPY patchguard /app/patchguard
COPY assets/mnist/processed/ /opt/mnist/MNIST/processed/
COPY tests /opt/tests

EXPOSE 8000
CMD ["uvicorn", "patchguard.evaluator.app:app", "--host", "0.0.0.0", "--port", "8000"]
