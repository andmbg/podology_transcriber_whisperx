import os
import json
import time
import uuid
import secrets
import subprocess
from pathlib import Path

import uvicorn
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import JSONResponse
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Request,
    Depends,
)
from wxtrans.assets import dummy_result

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HFTOKEN")
API_TOKEN = os.getenv("API_TOKEN")

# Create the FastAPI app
app = FastAPI()

# Directory to store uploaded audio files temporarily
UPLOAD_DIR = Path("/tmp/audio_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Store job statuses and results (for demo; use a DB or persistent store in production)
JOBS = {}

print(API_TOKEN)


def generate_job_id(length=8):
    # Generates a random 8-character hex string
    return f"jid_{secrets.token_hex(length // 2)}"


def check_api_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    """
    Endpoint to handle audio file uploads and transcribe them using WhisperX.
    """
    # Save the uploaded file to the temporary directory
    jid = str(uuid.uuid4())  # Generate a unique ID for the file
    file_path = UPLOAD_DIR / f"{jid}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    JOBS[jid] = {"status": "processing", "result": None}
    background_tasks.add_task(process_transcription, job_id=jid, file_path=file_path)
    return {"job_id": jid}


def process_transcription(job_id, file_path, threads: int = 8):
    try:
        result = run_whisperx(file_path, threads=threads)
        JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/dummytranscribe")
async def dummytranscribe(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    """
    Endpoint to handle dummy transcription requests.
    """
    jid = generate_job_id()
    JOBS[jid] = {"status": "processing", "result": None}
    logger.debug(f"{jid}: Received a dummy transcription request")
    background_tasks.add_task(process_dummy, job_id=jid)
    return {"job_id": jid}


def process_dummy(job_id):
    time.sleep(5)
    logger.debug("Slept 10s")
    JOBS[job_id] = {"status": "done", "result": dummy_result}


@app.get("/transcribe/status/{job_id}")
def get_status(
    job_id: str,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    logger.debug(f"Checking status for job {job_id}")
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job["status"]}


@app.get("/transcribe/result/{job_id}")
def get_result(
    job_id: str,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Transcription not finished")

    logger.debug("Sending out transcription result")
    return JSONResponse(content=job["result"])


def run_whisperx(audio_path: Path, threads: int) -> dict:
    """
    Run WhisperX as a command-line process on the given audio file and return the transcription result.
    """
    logger.debug(f"Running WhisperX on {audio_path}")

    # Define the output directory for WhisperX
    output_dir = audio_path.parent / "whisperx_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the WhisperX command
    # fmt: off
    command = [
        "whisperx", str(audio_path),
        "--output_dir", str(output_dir),
        "--output_format", "json",
        "--hf_token", os.getenv("HFTOKEN", ""),
        "--batch_size", "4",
        "--compute_type", "int8",
        "--model", "large-v2",
        "--diarize",
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "--threads", f"{threads}",
    ]
    # fmt: on

    # Run the command
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.debug(f"WhisperX command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"WhisperX failed with error: {e.stderr}")
        raise RuntimeError(f"WhisperX transcription failed: {e.stderr}")

    # Locate the JSON output file
    json_file = output_dir / f"{audio_path.stem}.json"
    if not json_file:
        raise RuntimeError(
            f"JSON output file {json_file} found in the WhisperX output directory"
        )

    logger.debug(f"WhisperX output JSON file: {json_file}")

    # Read and parse the JSON file
    try:
        with open(json_file, "r") as f:
            transcription_result = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse WhisperX output JSON file: {str(e)}")

    # Clean up the result:
    transcription_result = fix_missing_speakers(transcription_result)
    if "word_segments" in transcription_result:
        del transcription_result["word_segments"]

    logger.info(f"WhisperX transcription completed for {audio_path}")
    return transcription_result


def fix_missing_speakers(transcription_result: dict) -> dict:
    for i, segment in enumerate(transcription_result["segments"]):
        if "speaker" not in segment:
            transcription_result["segments"][i] = "UNKNOWN"

    return transcription_result


@app.get("/")
def root():
    """
    Root endpoint to verify the server is running.
    """
    return {"message": "WhisperX transcription server is running"}


if __name__ == "__main__":
    uvicorn.run("wxtrans:app", host="0.0.0.0", port=8001)
