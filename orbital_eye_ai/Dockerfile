# Public demo container (Hugging Face Spaces, Docker SDK). Also works on any Docker host.
FROM python:3.12-slim

# Hugging Face Spaces run containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt -r requirements-ml.txt

COPY --chown=user . .

# Public demo defaults. Set MODEL_URL and SECRET_KEY as Space *secrets* (never in this file).
ENV PUBLIC_DEMO=1 FLASK_DEBUG=0 PORT=7860
EXPOSE 7860

# --preload: download the model and seed the demo data once, before the worker starts
CMD ["gunicorn", "--preload", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "4", "--timeout", "120", "app:create_app()"]
