FROM tiagopeixoto/graph-tool:latest

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    git \
    wget \
    curl \
    tree \
    base-devel

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

RUN echo "Contents of the build context:" && ls -la /workspace


# WORKDIR /workspace
# COPY ../pyproject.toml ../poetry.lock ./
# RUN poetry install --no-root