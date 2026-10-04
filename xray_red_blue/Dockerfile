ARG XRAY_BASE_IMAGE=adv_demo
FROM ${XRAY_BASE_IMAGE}
ARG PYTORCH_INDEX_URL

WORKDIR /app
RUN python -m pip install --no-cache-dir --disable-pip-version-check mlflow-skinny==3.16.1
RUN if [ -n "$PYTORCH_INDEX_URL" ]; then \
    python -m pip install --no-cache-dir --upgrade torch==2.13.0 torchvision==0.28.0 --index-url "$PYTORCH_INDEX_URL"; \
    fi
RUN python -m pip install --no-cache-dir --disable-pip-version-check flask==3.1.2 gunicorn==23.0.0
COPY range_device.py /app/range_device.py
COPY xray_red_blue/redblue_app.py /app/redblue_app.py
COPY xray_red_blue/mlflow_audit.py /app/mlflow_audit.py
COPY xray_red_blue/defenses.py /app/xray_defenses.py
EXPOSE 5000
ENTRYPOINT []
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "0", "redblue_app:app"]
