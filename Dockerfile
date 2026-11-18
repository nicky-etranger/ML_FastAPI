FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY api/ api/
COPY core/ core/
COPY ml/ ml/
COPY schemas/ schemas/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
