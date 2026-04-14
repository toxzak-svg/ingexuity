FROM julia:1.10

WORKDIR /root

# Copy Project.toml
COPY Project.toml .

# Install Julia dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Copy source
COPY src/ src/
COPY genie/ genie/

# Precompile the package
RUN julia --project=. -e 'using IngExuity; using Genie'

EXPOSE 8000

CMD ["julia", "--project=.", "-e", "using IngExuity; IngExuity.start()"]
