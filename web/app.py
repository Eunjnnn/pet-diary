"""pet-diary web app: upload videos, get your pet's diary back.

Run:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/uvicorn web.app:app --host 0.0.0.0 --port 8899

The local VLM is loaded lazily on the first job and kept in memory. Jobs run
one at a time (single GPU) in FastAPI's background thread pool.
"""

import shutil
import threading
import uuid
from pathlib import Path

import markdown
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pet_diary.pipeline import VIDEO_EXTS, generate_diary

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR.parent / "web_jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="pet-diary")

_jobs: dict[str, dict] = {}
_backend = None
_backend_lock = threading.Lock()
_gpu_lock = threading.Lock()  # one job at a time on the GPU


def get_backend():
    global _backend
    with _backend_lock:
        if _backend is None:
            from pet_diary.local_backend import LocalVLMBackend
            _backend = LocalVLMBackend()
    return _backend


def run_job(job_id: str, input_path: Path, out_dir: Path, date: str, lang: str) -> None:
    job = _jobs[job_id]

    def log(msg: str) -> None:
        job["log"].append(msg)

    try:
        log("Loading the model (takes ~1 min on first run)...")
        backend = get_backend()
        with _gpu_lock:
            log("Starting the pipeline...")
            result = generate_diary(
                input_path, backend, out_dir, date_label=date, lang=lang, log=log,
            )
        job["diary"] = result["diary"]
        job["diary_html"] = markdown.markdown(result["diary"])
        job["captions"] = result["captions"]
        job["keyframes"] = {
            clip: [f"/files/{job_id}/out/keyframes/{p.name}" for p in paths]
            for clip, paths in result["keyframes"].items()
        }
        job["time_labels"] = result["time_labels"]
        job["status"] = "done"
        log("Done! 🐾")
    except ValueError as e:
        job["status"] = "error"
        job["error"] = str(e)
    except Exception as e:  # surface unexpected failures to the UI
        job["status"] = "error"
        job["error"] = f"unexpected error: {e}"


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    date: str = Form("오늘"),
    lang: str = Form("ko"),
):
    valid = [f for f in files if Path(f.filename or "").suffix.lower() in VIDEO_EXTS]
    if not valid:
        raise HTTPException(400, "Please upload video files (.mp4/.webm/.mov/.mkv/.avi)")

    job_id = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True)

    for f in valid:
        dest = input_dir / Path(f.filename).name
        with dest.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)

    # One file => long-recording mode (motion events); several => clip mode.
    saved = list(input_dir.iterdir())
    input_path = saved[0] if len(saved) == 1 else input_dir

    _jobs[job_id] = {"status": "running", "log": [], "queued": _gpu_lock.locked()}
    background_tasks.add_task(run_job, job_id, input_path, job_dir / "out", date, lang)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


app.mount("/files", StaticFiles(directory=JOBS_DIR), name="files")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
