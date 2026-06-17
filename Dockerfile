FROM julia:1.12

WORKDIR /app

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

COPY Project.toml .
COPY src/ src/
COPY python/ python/

# Install Python deps and export GPT-2 weights to Julia binary
RUN mkdir -p models
RUN pip install --no-cache-dir --break-system-packages -r python/requirements.txt
COPY scripts/export_gpt2_weights.py scripts/
RUN python scripts/export_gpt2_weights.py && rm scripts/export_gpt2_weights.py

RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'
RUN julia --project=. -e 'using IngExuity'

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]