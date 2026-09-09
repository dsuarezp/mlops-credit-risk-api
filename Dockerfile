FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
# separately from the application code, which changes more often.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model.
COPY app.py .
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
