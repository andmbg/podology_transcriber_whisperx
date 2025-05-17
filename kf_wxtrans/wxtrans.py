import os
import json
import uuid
import subprocess
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

load_dotenv(find_dotenv())

HFTOKEN = os.getenv("HFTOKEN")

# Create the FastAPI app
app = FastAPI()

# Directory to store uploaded audio files temporarily
UPLOAD_DIR = Path("/tmp/audio_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Endpoint to handle audio file uploads and transcribe them using WhisperX.
    """
    # Save the uploaded file to the temporary directory
    file_id = str(uuid.uuid4())  # Generate a unique ID for the file
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Run WhisperX on the saved file
    try:
        # transcription = run_whisperx(file_path)
        transcription = run_dummy(file_path)  # Use dummy function for testing
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WhisperX transcription failed: {str(e)}")
    finally:
        # Clean up the uploaded file after processing
        file_path.unlink(missing_ok=True)

    # Return the transcription result
    return JSONResponse(content=transcription)


def run_whisperx(audio_path: Path) -> dict:
    """
    Run WhisperX on the given audio file and return the transcription result.
    """
    # Example command to run WhisperX

    command_cpu = [
        "whisperx",
        str(audio_path),
        "--batch_size 4",
        "--compute_type int8",
        "--diarize",
        "--align_model WAV2VEC2_ASR_LARGE_LV60K_960H",
        f"--hf_token {HFTOKEN}",
        "--output_format json",
        "--min_speakers 3",
        "--threads 8"
    ]

    # Run the command and capture the output
    result = subprocess.run(command_cpu, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"WhisperX failed: {result.stderr}")

    # Parse the JSON output from WhisperX
    try:
        json_file = Path.cwd() / audio_path.name.replace(".mp3", ".json")
        print(json_file)

        with open(json_file, "r") as f:
            return json.load(f)

    except Exception as e:
        raise RuntimeError(f"Failed to parse WhisperX output: {str(e)}")


def run_dummy(audio_path: Path) -> dict:
    """
    Dummy function to simulate transcription.
    """
    return {
        "task": "transcribe",
        "language": "en",
        "duration": 2.24,
        "segments": [
            {
                "id": 0,
                "text": "Andy in Kansas, you're on the air. Thanks for holding.",
                "start": 0.5,
                "end": 2.24,
                "avg_logprob": -0.10296630859375,
                "language": "en",
                "speaker": "SPEAKER_03",
                "words": [
                    {
                        "word": "Andy",
                        "start": 0.5,
                        "end": 0.5,
                        "score": 0,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "in",
                        "start": 0.5,
                        "end": 0.68,
                        "score": 0.08,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "Kansas,",
                        "start": 0.68,
                        "end": 1.14,
                        "score": 0.32,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "you're",
                        "start": 1.14,
                        "end": 1.28,
                        "score": 0.76,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "on",
                        "start": 1.28,
                        "end": 1.36,
                        "score": 0.9,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "the",
                        "start": 1.36,
                        "end": 1.44,
                        "score": 0.92,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "air.",
                        "start": 1.44,
                        "end": 1.72,
                        "score": 0.33,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "Thanks",
                        "start": 1.72,
                        "end": 1.72,
                        "score": 0.29,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "for",
                        "start": 1.72,
                        "end": 1.82,
                        "score": 0.86,
                        "speaker": "SPEAKER_03"
                    },
                    {
                        "word": "holding.",
                        "start": 1.82,
                        "end": 2.24,
                        "score": 0.54,
                        "speaker": "SPEAKER_03"
                    }
                ]
            }
        ]
    }


@app.get("/")
def root():
    """
    Root endpoint to verify the server is running.
    """
    return {"message": "WhisperX transcription server is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wxtrans:app", host="127.0.0.1", port=8001)
