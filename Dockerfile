FROM python:3.11-slim

WORKDIR /app

COPY requirements-serve.txt .
# torch는 CPU 전용 wheel로 따로 설치 — CUDA 포함 기본 wheel은 수 GB라
# Render 무료 플랜 빌드/실행에 부적합하다.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY web/ web/

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn web.api:app --host 0.0.0.0 --port ${PORT}"]
