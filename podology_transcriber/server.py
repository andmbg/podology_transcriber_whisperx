import os
import sys
import json
import time
import subprocess
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import traceback  # Add this import at the top

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

# Create logs directory
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)

# Setup global logging with both console and file output
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="DEBUG")  # Console logging
logger.add(
    LOGS_DIR / "transcriber.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

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
                path TEXT,
                error_message TEXT
            )
        """
        )


# Create the FastAPI app
app = FastAPI()
init_jobs_db()


def set_job(job_id, status, path=None, error_message=None):
    with sqlite3.connect(JOBS_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, path, error_message) VALUES (?, ?, ?, ?)",
            (job_id, status, path, error_message),
        )


def get_job(job_id):
    try:
        with sqlite3.connect(JOBS_DB_PATH) as conn:
            row = conn.execute(
                "SELECT status, path, error_message FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row:
                status, path, error_message = row
                logger.debug(f"Job {job_id} found with status: {status}")
                return {"status": status, "path": path, "error_message": error_message}
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


@contextmanager
def job_logging(job_id: str):
    """Context manager for job-specific logging"""
    log_file = LOGS_DIR / f"{job_id}.log"

    # Add job-specific file handler
    handler_id = logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )

    try:
        yield log_file
    finally:
        logger.remove(handler_id)


def process_transcription(job_id, audio_path):
    transcript_path = audio_path.with_suffix(".json")

    try:
        logger.info(f"Job {job_id}: Starting transcription for {audio_path}")
        result = run_whisperx(audio_path)

        logger.debug(f"Job {job_id}: Writing transcription result to {transcript_path}")
        with open(transcript_path, "w") as f:
            json.dump(result, f, indent=2)

        # DEBUG
        set_job(job_id, "failed", error_message="+++ Simulated failure for testing purposes +++")
        logger.error(f"Job {job_id}: +++ Simulated failure for testing purposes +++")
        # set_job(job_id, "done", path=str(transcript_path))
        # logger.info(f"Job {job_id}: Transcription completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id}: Transcription failed: {e}")
        logger.exception(f"Job {job_id}: Full exception details:")

        # Capture the full traceback for the error message
        full_traceback = traceback.format_exc()
        error_message = f"Exception: {str(e)}\n\nFull traceback:\n{full_traceback}"
        
        set_job(job_id, "failed", error_message=error_message)

        if transcript_path.exists():
            transcript_path.unlink()

    finally:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
        logger.debug(f"Job {job_id}: Cleaned up temporary files")


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
    include_logs: bool = False,
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

    # Always include logs for failed jobs, optionally for others
    if status == "failed" or include_logs:
        # Read from the global log file and filter for this job
        log_file = LOGS_DIR / "transcriber.log"
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    all_logs = f.read()

                # Filter logs for this specific job
                job_logs = []
                for line in all_logs.split("\n"):
                    if f"Job {job_id}:" in line:
                        job_logs.append(line)

                if job_logs:
                    response["logs"] = "\n".join(job_logs)
                else:
                    response["logs"] = f"No logs found for job {job_id}"

                logger.debug(f"Included filtered logs for job {job_id}")
            except Exception as e:
                logger.error(f"Failed to read log file for job {job_id}: {e}")
                response["logs"] = f"Error reading log file: {e}"
        else:
            response["logs"] = "No log file found"

    # Include error message for failed jobs
    if status == "failed":
        response["error_message"] = job.get("error_message", "Unknown error")

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
    Run WhisperX with logging
    """
    logger.debug(f"Running WhisperX on {audio_path}")

    # Define the output directory for WhisperX
    output_dir = audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct the WhisperX command
    # fmt: off
    command = [
        sys.executable, "-m", "whisperx",  str(audio_path),
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

        # Use Popen for real-time output capture
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1 hour timeout
        )

        logger.debug(f"WhisperX stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"WhisperX stderr: {result.stderr}")

        logger.info("WhisperX process completed successfully")

    except subprocess.CalledProcessError as e:
        error_msg = f"WhisperX failed with return code {e.returncode}\n"
        error_msg += f"Command: {' '.join(command)}\n"
        error_msg += f"Stdout: {e.stdout}\n"
        error_msg += f"Stderr: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        logger.error(f"Unexpected error running WhisperX: {e}")
        raise RuntimeError(f"Unexpected error running WhisperX: {str(e)}")

    # Locate the JSON output file
    json_file = output_dir / f"{audio_path.stem}.json"
    if not json_file.exists():
        logger.error(f"JSON output file not found: {json_file}")
        files_in_dir = list(output_dir.glob("*"))
        logger.debug(f"Files in output directory: {files_in_dir}")
        raise RuntimeError(f"JSON output file {json_file} not found")

    logger.debug(f"WhisperX output JSON file: {json_file}")

    # Read and parse the JSON file
    try:
        with open(json_file, "r") as f:
            transcription_result = json.load(f)
        logger.debug(
            f"Successfully parsed JSON with {len(transcription_result.get('segments', []))} segments"
        )
    except Exception as e:
        logger.error(f"Failed to parse WhisperX output JSON file: {str(e)}")
        raise RuntimeError(f"Failed to parse WhisperX output JSON file: {str(e)}")

    # Clean up the result
    transcription_result = fix_missing_speakers(transcription_result)
    if "word_segments" in transcription_result:
        del transcription_result["word_segments"]

    logger.info("WhisperX transcription processing completed")
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
