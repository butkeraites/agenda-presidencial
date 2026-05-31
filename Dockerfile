# Imagem multiuso: scheduler (scraper + processor) e dashboard rodam a partir dela.
# O comando é definido por docker-compose, por serviço.
#
# --platform=linux/amd64: as deps pinadas (pandas==1.2.4 etc.) só têm wheels
# para amd64. Em Apple Silicon, Docker Desktop emula via Rosetta. Quando as
# pins forem atualizadas para versões com wheels ARM64, remova esta linha.

FROM --platform=linux/amd64 python:3.8-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data

# postgresql-client -> pg_dump (snapshots)
# git              -> commit/push do CSV dump
# gcc/libpq-dev    -> build do wheel de psycopg2-binary se necessário
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        postgresql-client \
        gcc \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache de dependências: requirements mudam raramente, código muda muito.
COPY requirements.txt requirements-pipeline.txt ./
RUN pip install -r requirements-pipeline.txt -r requirements.txt

# Pre-baixa stopwords do NLTK para o scrape não depender da rede.
RUN python -m nltk.downloader -d $NLTK_DATA stopwords

# O código-fonte vem por bind-mount em runtime (docker-compose monta .:/app),
# então não há COPY do projeto aqui. Isso mantém o build rápido e permite
# editar código sem rebuild.

CMD ["python", "-c", "print('Override the command in docker-compose.yml')"]
