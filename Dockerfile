FROM julia:1.10

WORKDIR /app

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

COPY Project.toml .
COPY src/ src/
COPY python/ python/

RUN pip install --no-cache-dir -r python/requirements.txt

RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'
RUN julia --project=. -e 'using IngExuity'

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]