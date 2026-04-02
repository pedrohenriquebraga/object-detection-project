FROM tensorflow/tensorflow:2.16.1-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# OpenCV/TensorFlow runtime dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && grep -viE '^tensorflow(\[.*\])?$' requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir --force-reinstall numpy==1.26.4 ml-dtypes==0.3.2 pillow==10.4.0 \
    && pip install --no-cache-dir -r requirements.docker.txt \
    && pip install --no-cache-dir --force-reinstall numpy==1.26.4 ml-dtypes==0.3.2

COPY build.py ./

RUN mkdir -p /app/models /app/data
VOLUME ["/app/models", "/app/data", "/app/logs"]

EXPOSE 6006

CMD ["python", "build.py"]