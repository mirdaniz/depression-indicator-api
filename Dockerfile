FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries=5 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]