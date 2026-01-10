FROM python:3.11-slim

WORKDIR /app

# Copy only what we need first (better caching)
COPY pyproject.toml /app/
COPY sharpe_optimizer /app/sharpe_optimizer

# Install your package (and its dependencies)
RUN pip install --no-cache-dir .

# Run the CLI command by default
CMD ["sharpe-opt"]
