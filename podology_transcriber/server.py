import os
import sys
import json
import time
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
    Form,
)

from podology_transcriber.assets import dummy_result

logger.remove()
logger.add(sys.stderr, level="DEBUG")

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
API_TOKEN = os.getenv("API_TOKEN")
JOBS_DB_PATH = Path.cwd() / "podology_transcriber" / "jobs.db"
UPLOAD_DIR = Path.cwd() / "podology_transcriber" / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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


def set_job(job_id, status, path=None):
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, path) VALUES (?, ?, ?)",
            (job_id, status, path),
        )


def get_job(job_id):
    try:
        with sqlite3.connect(JOBS_DB_PATH) as conn:
            row = conn.execute(
                "SELECT status, path FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row:
                status, path = row
                logger.debug(f"Job {job_id} found with status: {status}")
                return {"status": status, "path": path}
            logger.debug(f"Job {job_id} not found in database.")
            return None
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching job {job_id} from database"
        )


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
    job_id: str = Form(None),
    background_tasks: BackgroundTasks = None,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    """
    Endpoint to handle audio file uploads and transcribe them using WhisperX.
    """
    if not job_id:
        return HTTPException(
            status_code=400, detail="Job ID must be provided in the request"
        )
    logger.info(f"{job_id}: Received a transcription request")

    # Save the uploaded file to the temporary directory

    audio_path = UPLOAD_DIR / f"{audiofile.filename}"
    set_job(job_id, "processing")

    with open(audio_path, "wb") as f:
        f.write(await audiofile.read())

    background_tasks.add_task(
        process_transcription, job_id=job_id, audio_path=audio_path
    )

    return {"job_id": job_id, "status": "processing"}


def process_transcription(job_id, audio_path):
    transcript_path = audio_path.with_suffix(".json")

    try:
        result = run_whisperx(audio_path)
        logger.debug(f"Writing transcription result to {transcript_path}")

        with open(transcript_path, "w") as f:
            json.dump(result, f, indent=2)

        set_job(job_id, "done")
        logger.info(f"{job_id}: Transcription completed successfully")

    except Exception as e:
        logger.error(f"{job_id}: Transcription failed: {e}")
        set_job(job_id, "failed", str(e))  # Fix: Store error as string, not dict

        # Clean up failed transcript file
        if transcript_path.exists():
            transcript_path.unlink()

    finally:
        # Clean up audio file
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
        logger.debug(f"{job_id}: Cleaned up temporary files")


@app.post("/dummytranscribe")
async def dummytranscribe(
    audiofile: UploadFile = File(...),
    job_id: str = None,
    background_tasks: BackgroundTasks = None,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    """
    Endpoint to handle dummy transcription requests.
    """
    set_job(job_id, "processing")
    logger.debug(f"{job_id}: Received a dummy transcription request")
    background_tasks.add_task(process_dummy, job_id=job_id)

    return {"job_id": job_id}


def process_dummy(job_id):
    logger.info(f"{job_id}: Processing dummy transcription")
    transcript_path = UPLOAD_DIR / f"{job_id}.json"
    time.sleep(10)
    logger.debug("Slept 10s")

    with open(transcript_path, "w") as f:
        json.dump(dummy_result, f, indent=2)
    set_job(job_id, "done")


@app.get("/status/{job_id}")
def get_status(
    job_id: str,
    request: Request = None,
    _: None = Depends(check_api_token),
):
    logger.debug(f"Getting status for job {job_id}")
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    logger.debug(f"Job {job_id} found with status: {job['status']}")

    status = job["status"]
    response = {"status": status}

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

    logger.debug(f"Downloading transcript for job {job_id}, status: {job['status']}")

    transcript_path = UPLOAD_DIR / f"{job_id}.json"
    logger.debug(f"Transcript path: {transcript_path}")

    if not transcript_path or not Path(transcript_path).exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    return FileResponse(
        transcript_path, media_type="application/json", filename=f"payload.json"
    )


def run_whisperx(audio_path: Path) -> dict:
    """
    Run WhisperX as a command-line process on the given audio file and return the transcription result.
    """
    logger.debug(f"Running WhisperX on {audio_path}")

    # Define the output directory for WhisperX
    output_dir = audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the WhisperX command
    # fmt: off
    command = [
        "whisperx", str(audio_path),
        "--output_dir", str(output_dir),
        "--output_format", "json",
        "--hf_token", HF_TOKEN,
        #"--batch_size", "4",
        # "--compute_type", "int8",
        "--model", "large-v2",
        "--diarize",
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
        # "--threads", "8",
    ]
    # fmt: on

    # Run the command
    try:
        logger.info(f"Running WhisperX command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.debug(f"WhisperX command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"WhisperX failed with error: {e.stderr}")
        raise RuntimeError(f"WhisperX transcription failed: {e.stderr}")

    # Locate the JSON output file
    json_file = output_dir / audio_path.with_suffix(".json")
    if not json_file.exists():
        raise RuntimeError(
            f"JSON output file {json_file} not found in the WhisperX output directory"
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
