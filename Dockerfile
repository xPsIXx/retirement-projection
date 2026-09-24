FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY firemodel/ firemodel/
COPY default_plans/ default_plans/
COPY config.py web_app.py ./
COPY templates/ templates/

# Sync dependencies
RUN uv sync --no-dev --frozen

# Expose port
EXPOSE 8177

# Run the web app
CMD ["uv", "run", "web_app.py"]
