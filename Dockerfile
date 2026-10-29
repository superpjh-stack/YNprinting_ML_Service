FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY web/package*.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YN_RUNTIME=/data
WORKDIR /app
COPY requirements-tested.txt ./
RUN pip install --no-cache-dir -r requirements-tested.txt \
    && useradd --uid 10001 --create-home app \
    && mkdir /data && chown app:app /data
COPY api/ ./api/
COPY docs/ ./docs/
COPY output/ ./output/
COPY --from=frontend /build/dist ./web/dist
USER app
EXPOSE 8042
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8042/api/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8042"]
