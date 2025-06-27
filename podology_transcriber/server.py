import os
import shelve
import sys
import json
import time
import uuid
import secrets
import subprocess
from pathlib import Path
import sqlite3

import uvicorn
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import FileResponse, JSONResponse
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Request,
    Depends,
)
from podology_transcriber.assets import dummy_result

logger.remove()
logger.add(sys.stderr, level="DEBUG")

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
API_TOKEN = os.getenv("API_TOKEN")
JOBS_DB_PATH = "jobs.db"


def init_jobs_db():
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                path TEXT
            )
        """
        )


# Create the FastAPI app
app = FastAPI()
init_jobs_db()

# Directory to store uploaded audio files temporarily
UPLOAD_DIR = Path("/tmp/audio_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def set_job(job_id, status, path=None):
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, path) VALUES (?, ?, ?)",
            (job_id, status, path),
        )


def get_job(job_id):
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        row = conn.execute(
            "SELECT status, path FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row:
            status, path = row
            return {"status": status, "path": path}
        return None


def generate_job_id(length=8):
    # Generates a random 8-character hex string
    return f"jid_{secrets.token_hex(length // 2)}"


def check_api_token(request: Request):
    logger.info("Checking API token for request")
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")


@app.post("/transcribe")
async def transcribe(
    audiofile: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    """
    Endpoint to handle audio file uploads and transcribe them using WhisperX.
    """
    # Save the uploaded file to the temporary directory
    jid = generate_job_id()  # Generate a unique ID for the file
    logger.debug(f"{jid}: Received a transcription request")

    audio_path = UPLOAD_DIR / f"{jid}_{audiofile.filename}"
    set_job(jid, "processing")

    with open(audio_path, "wb") as f:
        f.write(await audiofile.read())

    background_tasks.add_task(process_transcription, job_id=jid, audio_path=audio_path)
    return {"job_id": jid}


def process_transcription(job_id, audio_path, threads: int = 8):
    transcript_path = audio_path.with_suffix(".json")

    try:
        result = run_whisperx(audio_path, threads=threads)
        with open(transcript_path, "w") as f:
            json.dump(result, f, indent=2)
        set_job(job_id, "done", str(transcript_path))
    
    except Exception as e:
        set_job(job_id, "failed", {"error": str(e)})
    
    finally:
        audio_path.unlink(missing_ok=True)


@app.post("/dummytranscribe")
async def dummytranscribe(
    audiofile: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    """
    Endpoint to handle dummy transcription requests.
    """
    jid = generate_job_id()
    set_job(jid, "processing")
    logger.debug(f"{jid}: Received a dummy transcription request")
    background_tasks.add_task(process_dummy, job_id=jid)

    return {"job_id": jid}


def process_dummy(job_id):
    logger.info(f"{job_id}: Processing dummy transcription")
    transcript_path = UPLOAD_DIR / f"{job_id}_dummy.json"
    time.sleep(10)
    logger.debug("Slept 10s")

    with open(transcript_path, "w") as f:
        json.dump(dummy_result, f, indent=2)
    set_job(job_id, "done", str(transcript_path))


@app.get("/status/{job_id}")
def get_status(
    job_id: str,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    logger.debug(f"Checking status for job {job_id}")
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job["status"]
    response = {"status": status}
    if status == "done":
        base_url = str(request.base_url).rstrip("/")
        response["download_url"] = f"{base_url}/download/{job_id}"

    return JSONResponse(content=response)


@app.get("/download/{job_id}")
def download_transcript(
    job_id: str,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    job = get_job(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not found or not done")

    transcript_path = job["path"]
    if not transcript_path or not Path(transcript_path).exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    return FileResponse(
        transcript_path, media_type="application/json", filename=f"payload.json"
    )


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
        "--hf_token", HF_TOKEN,
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
            transcription_result["segments"][i]["speaker"] = "UNKNOWN"

    return transcription_result


@app.get("/")
def root():
    """
    Root endpoint to verify the server is running.
    """
    return {"message": "WhisperX transcription server is running"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, workers=1, log_level="debug")
