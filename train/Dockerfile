FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENV MODEL_DIR=/models

ENTRYPOINT ["python", "train.py", "--model-dir", "/models"]
