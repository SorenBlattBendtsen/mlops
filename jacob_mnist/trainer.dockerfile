# Base image
FROM --platform=linux/arm64/v8 python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY jacob_mnist/requirements.txt requirements.txt
COPY jacob_mnist/pyproject.toml pyproject.toml
COPY jacob_mnist/jacob_mnist/ jacob_mnist/
COPY jacob_mnist/models/ models/
COPY jacob_mnist/reports/ reports/


WORKDIR /
RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "jacob_mnist/train_model.py"]
