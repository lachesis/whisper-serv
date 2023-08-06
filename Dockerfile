FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg wget curl python-is-python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN cd /app/whisper.cpp \
    && make clean \
    && WHISPER_CUBLAS=1 make -j

RUN cd /app/whisper.cpp \
    && test -e models/ggml-large.bin || ./models/download-ggml-model.sh large

RUN python3 -m pip install -r /app/requirements.txt

CMD python3 ./server.py
EXPOSE 8080
