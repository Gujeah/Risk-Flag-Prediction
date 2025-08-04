
FROM python:3.9.13-slim


WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=flask_api/main.py
ENV FLASK_RUN_HOST=0.0.0.0


CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "flask_api.main:app"]