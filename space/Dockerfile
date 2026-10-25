FROM lmsysorg/sglang:v0.5.19-cu129-runtime
# HF Spaces run the container as uid 1000: keep every cache in writable /tmp.
ENV HOME=/tmp HF_HOME=/tmp/hf TRITON_CACHE_DIR=/tmp/triton FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer \
    XDG_CACHE_HOME=/tmp/cache PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir "gradio==5.49.1" pyyaml ninja
WORKDIR /app
COPY app.py ps8_common.py examples.json /app/
EXPOSE 7860
CMD ["python3", "app.py"]
