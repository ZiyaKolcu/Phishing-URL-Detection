FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ml_model /app/ml_model

COPY . .

CMD ["python", "app.py"]

