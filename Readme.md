# Transcriber API for the Podology app

This is a simple FastAPI server that transcribes audio files and sends back a diarized JSON transcript. It is prepared to run on a machine with an Nvidia GPU and installed Nvidia drivers.

## Getting started

> First, get an API token for [`pyannote/speaker-diarization`](https://huggingface.co/pyannote/speaker-diarization) at huggingface.co. It is necessary to use the diarization feature in the whisperX transcription model. Keep this token, see below for how to use it.

Once cloned to the host system,

1. set `.env` content,
2. build docker image,
3. run.

### 1. `.env`

This file is necessary for the API to run and it contains two values:

    API_TOKEN=...
    HF_TOKEN=...

The `API_TOKEN` can be anything, and you will paste it also in the main Podology `.env`. This is to limit access to the API to your own app.

The `HF_TOKEN` is the one you got at huggingface.co, see above.

### 2. Build image

With `.env` in place, enter

    docker build -t transcriber .

Wait for the image to build, then run it.

### 3. run

Run the docker image using

    docker run --gpus all -d --rm --name transcriber -p 8001:8001 transcriber

If you want to watch the API work by keeping a terminal open and watching log messages, use

    docker logs -f transcriber

That should be it.
