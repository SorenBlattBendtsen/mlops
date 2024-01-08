# Base image
FROM --platform=linux/arm64/v8 python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY jacob_mnist/ jacob_mnist/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/


WORKDIR /
RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "jacob_mnist/predict_model.py"]
