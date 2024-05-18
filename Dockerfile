FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
COPY mobilenet_glasses_classifier.h5 .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app.py .

EXPOSE 5000

CMD ["python", "flask_app.py"]
