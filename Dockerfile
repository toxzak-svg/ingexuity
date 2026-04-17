FROM julia:1.10

WORKDIR /root

ENV PORT=8000

# Copy Project.toml
COPY Project.toml .

# Install Julia dependencies (force redownload to break stale cache)
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); using HTTP, Flux; @info "Deps ready"'

# Copy source
COPY src/ src/

# Precompile the package
RUN julia --project=. -e 'using IngExuity'

EXPOSE 8000

CMD ["julia", "--project=.", "-e", "using IngExuity; IngExuity.start()"]
