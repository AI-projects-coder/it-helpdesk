# syntax=docker/dockerfile:1
# IT Helpdesk: Streamlit dashboard + `helpdesk` CLI in one image.
# No secrets are baked in: OPENAI_API_KEY is provided at runtime (env / Secret Manager).

# ---- build stage: install dependencies into a venv -------------------------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

# ---- runtime stage ---------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HELPDESK_ROOT=/app \
    RUNTIME_DIR=/var/lib/helpdesk \
    PORT=8080

RUN useradd --create-home --uid 10001 helpdesk \
    && mkdir -p /var/lib/helpdesk \
    && chown helpdesk:helpdesk /var/lib/helpdesk

COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=helpdesk:helpdesk config ./config
COPY --chown=helpdesk:helpdesk data ./data
COPY --chown=helpdesk:helpdesk scripts ./scripts
COPY --chown=helpdesk:helpdesk .streamlit ./.streamlit
COPY --chown=helpdesk:helpdesk streamlit_app.py ./
COPY --chown=helpdesk:helpdesk docker/entrypoint.sh /usr/local/bin/entrypoint.sh

USER helpdesk
VOLUME ["/var/lib/helpdesk"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

ENTRYPOINT ["entrypoint.sh"]
CMD ["ui"]
