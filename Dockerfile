FROM debian:bullseye-slim
WORKDIR /app
COPY packages.txt .
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* packages.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["python3", "app.py"]
