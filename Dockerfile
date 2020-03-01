FROM python:3

WORKDIR /usr/src/app

COPY app.py .
COPY templates/about.html templates/
COPY templates/gemstones.html templates/
COPY static/stone_image.jpg static/
COPY data/gemstones/label_encoder.pkl data/gemstones/
COPY data/gemstones/resnet_model_weights.pth data/gemstones/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]
