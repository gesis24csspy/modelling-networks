FROM tiagopeixoto/graph-tool:latest

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    git \
    wget \
    curl \
    tree \
    python-pip \
    base-devel

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="$PYTHONPATH:/usr/local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages"

RUN curl -sS https://starship.rs/install.sh | sh -s -- -y \
    && echo 'eval "$(starship init bash)"' >> /root/.bashrc \
    && mkdir -p /workspace

WORKDIR /workspace
COPY pyproject.toml poetry.lock ./
RUN poetry install