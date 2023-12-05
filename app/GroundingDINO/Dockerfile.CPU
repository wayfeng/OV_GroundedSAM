FROM ubuntu:22.04 AS builder
RUN apt update && apt install -y  --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    python3-wheel build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /home/gradio/venv
ENV PATH="/home/gradio/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM ubuntu:22.04 AS runner
RUN apt update && apt install --no-install-recommends -y \
    python3.10 python3-venv && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home gradio
COPY --from=builder /home/gradio/venv /home/gradio/venv

USER gradio
RUN mkdir -p /home/gradio/app
WORKDIR /home/gradio/app
COPY app.py .
COPY models models
EXPOSE 7579
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/home/gradio/venv
ENV PATH="/home/gradio/venv/bin:$PATH"
CMD ["python","app.py"]