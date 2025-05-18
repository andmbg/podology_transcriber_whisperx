import os
import json
import uuid
import subprocess
from pathlib import Path

import uvicorn
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException


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
        transcription = run_whisperx(file_path)
        # transcription = run_dummy(file_path)  # Use dummy function for testing
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WhisperX transcription failed: {str(e)}")
    finally:
        # Clean up the uploaded file after processing
        file_path.unlink(missing_ok=True)

    # Return the transcription result
    return JSONResponse(content=transcription)


def run_whisperx(audio_path: Path) -> dict:
    """
    Run WhisperX as a command-line process on the given audio file and return the transcription result.
    """
    logger.debug(f"Running WhisperX on {audio_path}")

    # Define the output directory for WhisperX
    output_dir = audio_path.parent / "whisperx_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the WhisperX command
    command = [
        "whisperx",
        str(audio_path),
        "--output_dir", str(output_dir),
        "--output_format", "json",
        "--hf_token", os.getenv("HFTOKEN", ""),
        "--batch_size", "4",
        "--compute_type", "int8",
        "--diarize",
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "--threads", "8"
    ]

    # Run the command
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"WhisperX command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"WhisperX failed with error: {e.stderr}")
        raise RuntimeError(f"WhisperX transcription failed: {e.stderr}")

    # Locate the JSON output file
    json_file = output_dir / f"{audio_path.stem}.json"
    if not json_file:
        raise RuntimeError(f"JSON output file {json_file} found in the WhisperX output directory")

    logger.debug(f"WhisperX output JSON file: {json_file}")

    # Read and parse the JSON file
    try:
        with open(json_file, "r") as f:
            transcription_result = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse WhisperX output JSON file: {str(e)}")

    logger.info(f"WhisperX transcription completed for {audio_path}")
    return transcription_result


@app.get("/")
def root():
    """
    Root endpoint to verify the server is running.
    """
    return {"message": "WhisperX transcription server is running"}


if __name__ == "__main__":
    uvicorn.run("wxtrans:app", host="127.0.0.1", port=8001)
