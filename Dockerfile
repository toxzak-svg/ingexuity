FROM julia:1.10

WORKDIR /root

# Copy Project.toml
COPY Project.toml .

# Install Julia dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Copy source
COPY src/ src/

# Precompile the package
RUN julia --project=. -e 'using IngExuity'

EXPOSE 8000

CMD ["julia", "--project=.", "-e", "using IngExuity; IngExuity.start()"]
