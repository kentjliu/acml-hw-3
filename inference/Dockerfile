FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask gunicorn

COPY inference.py .

ENV MODEL_DIR=/models
ENV MODEL_NAME=model.pt
ENV PORT=5000

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 inference:app
