"""Measurements for the backend: structure (hops), time (flows) and memory, with the real app in-process.

Run it with the project's virtual environment from anywhere:

    python backend/tools/measure.py hops [--frames 12 36] [--json out.json]
    python backend/tools/measure.py flows [--frames-dir DIR] [--json out.json]
    python backend/tools/measure.py memory --frames-dir DIR [--second-dir DIR] [--async-loading] [--no-offload]

hops     Fake models, no GPU. For every endpoint: backend files touched, Python calls, hops (calls that cross
         from one backend file into another) and the deepest nesting. With two frame counts it also reports
         propagation's cost per frame.
flows    Real SAM2 and CoTracker. Wall time of each flow (init, second init, clicks, propagate, track, save,
         reopen) with the time inside each model call, the job stages and the peak GPU memory.
memory   Real models. Resident memory (RSS) and GPU memory around the flows that hold or move the most:
         open a video, open a second one, click, track, click again, propagate. A watchdog exits if free system
         RAM falls below --min-free-gb. Use --no-offload to compare tracking without moving SAM2's weights.

This is a measurement tool, not a test: it needs the GPU and real models for `flows` and `memory`. Differences
under about 10% between single runs are noise.
"""
import argparse
import collections
import contextlib
import functools
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

DEFAULT_FRAMES_DIR = BACKEND / "tests" / "bedroom"


# ---- shared: the app in-process ---------------------------------------------------------------------------------------
class Harness:
    """The real app behind a TestClient, with its working and saved folders in a temporary directory."""

    def __init__(self):
        from fastapi.testclient import TestClient
        from app.factory import create_app
        from core.context import AppContext

        self.temp = Path(tempfile.mkdtemp(prefix="measure_"))
        self.context = AppContext()
        self.patches = [patch("video.io.SESSION_CACHE_ROOT", self.temp / "sessions"),
                        patch("video.save.SAVED_ROOT", self.temp / "saved")]
        for active in self.patches:
            active.start()
        self.client = TestClient(create_app(self.context), raise_server_exceptions=False)

    def close(self):
        for active in self.patches:
            active.stop()
        shutil.rmtree(self.temp, ignore_errors=True)

    def headers(self):
        session = self.context.session
        return {"X-Session-ID": str(session.active_session_id),
                "X-Annotation-Revision": str(session.annotations.revision)}

    def wait_for_job(self, pause=0.01):
        jobs = self.context.jobs
        while jobs._job is not None and jobs._job["status"] in ("queued", "running"):
            time.sleep(pause)

    def job_status(self):
        return (self.context.jobs._job or {}).get("status")


def init_body(frames_dir, **overrides):
    return {"video_frames_dir": str(frames_dir), "online_mode": True, "batch_size": 32,
            "offload_video_to_cpu": True, "offload_state_to_cpu": False, **overrides}


def center_point(frames_dir):
    import cv2
    first = sorted(Path(frames_dir).glob("*.jpg"))[0]
    height, width = cv2.imread(str(first)).shape[:2]
    return [width // 2, height // 2]


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---- hops: structure, with fake models ---------------------------------------------------------------------------------
def run_hops(frames):
    import cv2
    import numpy as np
    import torch
    import co_tracker

    height, width = 24, 32

    class FakeMasker:
        online_mode = True
        default_batch_size = 32
        offload_video_to_cpu = True
        offload_state_to_cpu = False
        state_loaded = True
        device = type("Device", (), {"type": "cpu"})()

        def __init__(self, *, progress_callback=None, reuse=None):
            pass

        def init_state(self, video_dir, *, online_mode, batch_size, offload_video_to_cpu, offload_state_to_cpu,
                       async_loading_frames=False, progress_callback=None):
            self.state_loaded = True

        def release_state(self):
            self.state_loaded = False

        def offload_weights(self):
            pass

        def reload_weights(self):
            pass

        def reset_state(self):
            pass

        def add_new_points_or_box(self, *, frame_idx, obj_id, points=None, labels=None, clear_old_points=True, box=None):
            return frame_idx, [obj_id], [torch.ones((1, height, width))]

        def add_new_mask(self, *, frame_idx, obj_id, mask):
            pass

        def clear_all_prompts_in_frame(self, frame_idx, obj_id):
            pass

        def remove_object(self, obj_id):
            pass

        def propagate_in_video(self, *, start_frame_idx, max_frame_num_to_track, reverse=False, batch_size=None,
                               online_mode=None, collect_segments=False, frame_callback=None, **_):
            for index in range(start_frame_idx, start_frame_idx + max_frame_num_to_track):
                frame_callback(index, {1: np.ones((height, width), dtype=bool)})

    class FakeTracker:
        model_name = co_tracker.DEFAULT_COTRACKER_MODEL
        device = "cpu"
        model_resolution = (height, width)

        def __init__(self, *args, **kwargs):
            pass

        def prepare_video(self, video):
            return video

        def track(self, video, *, queries, add_support_grid):
            count = len(video)
            return np.zeros((len(queries), count, 2), dtype=np.float32), np.ones((len(queries), count), dtype=bool)

    skipped = {"tests", "cache", "saved", ".venv", "__pycache__", "tools"}
    relative_cache = {}

    def relative(filename):
        if filename not in relative_cache:
            try:
                parts = Path(filename).resolve().relative_to(BACKEND).parts
                relative_cache[filename] = None if skipped & set(parts) else "/".join(parts)
            except (ValueError, OSError):
                relative_cache[filename] = None
        return relative_cache[filename]

    label = [None]
    local = threading.local()
    stats = collections.OrderedDict()

    def profiler(frame, event, arg):
        current = label[0]
        if current is None or event not in ("call", "return"):
            return
        where = relative(frame.f_code.co_filename)
        if where is None:
            return
        if event == "return":
            local.depth = getattr(local, "depth", 1) - 1
            return
        entry = stats[current]
        entry["files"].add(where)
        entry["calls"] += 1
        caller = frame.f_back
        caller_where = relative(caller.f_code.co_filename) if caller is not None else None
        if caller_where is not None and caller_where != where:
            entry["hops"] += 1
        local.depth = getattr(local, "depth", 0) + 1
        entry["depth"] = max(entry["depth"], local.depth)

    work = Path(tempfile.mkdtemp(prefix="measure_hops_"))
    frames_dir = work / "frames"
    frames_dir.mkdir()
    for index in range(frames):
        cv2.imwrite(str(frames_dir / f"{index:05d}.jpg"), np.full((height, width, 3), index * 10, dtype=np.uint8))

    with patch("video.prompts.svm.SAM2VideoMasker", FakeMasker), patch("co_tracker.CoTracker", FakeTracker):
        harness = Harness()
        threading.setprofile(profiler)
        sys.setprofile(profiler)
        try:
            def call(name, method, path, wait=False, **kwargs):
                stats[name] = {"files": set(), "calls": 0, "hops": 0, "depth": 0, "status": None}
                label[0] = name
                response = harness.client.request(method, path, **kwargs)
                if wait:
                    harness.wait_for_job(0.005)
                label[0] = None
                stats[name]["status"] = response.status_code
                return response

            mask = np.ones((height, width), dtype=bool).tolist()
            call("GET /health", "GET", "/health")
            call("init_state (frames folder)", "POST", "/video/init_state", wait=True, json={"video_frames_dir": str(frames_dir)})
            call("GET /video/info", "GET", "/video/info")
            call("POST /video/objects", "POST", "/video/objects", headers=harness.headers())
            call("PUT /video/objects/{id}", "PUT", "/video/objects/2", headers=harness.headers(), json={"label": "cup", "color": "#00ff00"})
            call("add_new_points_or_box", "POST", "/video/add_new_points_or_box", headers=harness.headers(),
                 json={"frame_idx": 0, "obj_id": 1, "points": [[5, 5]], "labels": [1]})
            call("GET /video/mask_data/{n}", "GET", "/video/mask_data/0")
            call("add_new_mask", "POST", "/video/add_new_mask", headers=harness.headers(), json={"frame_idx": 1, "obj_id": 2, "mask": mask})
            call("clear_all_prompts_in_frame", "POST", "/video/clear_all_prompts_in_frame", headers=harness.headers(),
                 params={"frame_idx": 1, "obj_id": 2})
            call("propagate_in_video (job)", "POST", "/video/propagate_in_video", wait=True, headers=harness.headers(),
                 json={"use_tracked_points": False})
            call("track_prompt_points (job)", "POST", "/tracking/track_prompt_points", wait=True, headers=harness.headers(), json={})
            result_id = harness.context.session.session_metadata.get("latest_tracking_result_id", "missing")
            call("GET /tracking/results/{id}", "GET", f"/tracking/results/{result_id}")
            call("remove_object", "POST", "/video/remove_object", headers=harness.headers(), params={"obj_id": 2})
            call("reset_state", "POST", "/video/reset_state", headers=harness.headers())
            saved = call("save", "POST", "/video/save", headers=harness.headers(), json={"name": "hops"})
            saved_path = saved.json().get("saved_path") if saved.status_code == 200 else str(work / "missing")
            call("init_state (reopen a checkpoint)", "POST", "/video/init_state", wait=True, json={"video_frames_dir": saved_path})
        finally:
            sys.setprofile(None)
            threading.setprofile(None)
            harness.close()
            shutil.rmtree(work, ignore_errors=True)
    return [{"call": name, "files": len(entry["files"]), "calls": entry["calls"], "hops": entry["hops"],
             "depth": entry["depth"], "status": entry["status"]} for name, entry in stats.items()]


def command_hops(args):
    runs = {frames: run_hops(frames) for frames in args.frames}
    first = args.frames[0]
    print(f"{'call':38} {'files':>5} {'calls':>6} {'hops':>5} {'depth':>5}  (propagation clip: {first} frames)")
    for row in runs[first]:
        flag = "" if row["status"] == 200 else f"  HTTP {row['status']}"
        print(f"{row['call']:38} {row['files']:5d} {row['calls']:6d} {row['hops']:5d} {row['depth']:5d}{flag}")
    edits = [row for row in runs[first] if row["call"] in (
        "POST /video/objects", "PUT /video/objects/{id}", "add_new_points_or_box", "add_new_mask",
        "clear_all_prompts_in_frame", "remove_object", "reset_state")]
    print(f"edit calls, median: files {statistics.median(r['files'] for r in edits):g}, "
          f"calls {statistics.median(r['calls'] for r in edits):g}, hops {statistics.median(r['hops'] for r in edits):g}")
    if len(args.frames) >= 2:
        low, high = args.frames[0], args.frames[1]
        find = lambda frames: next(r for r in runs[frames] if r["call"] == "propagate_in_video (job)")
        a, b = find(low), find(high)
        per_calls = (b["calls"] - a["calls"]) / (high - low)
        per_hops = (b["hops"] - a["hops"]) / (high - low)
        print(f"propagation per frame: {per_calls:.1f} calls, {per_hops:.1f} hops; "
              f"fixed: {a['calls'] - per_calls * low:.0f} calls, {a['hops'] - per_hops * low:.0f} hops")
    if args.json:
        Path(args.json).write_text(json.dumps({str(k): v for k, v in runs.items()}, indent=2), encoding="utf-8")


# ---- flows: time, with the real models ---------------------------------------------------------------------------------
def command_flows(args):
    import torch
    import co_tracker
    import sam2_video_masker
    import tracking.prompt_tracking as prompt_tracking
    import video.propagation as propagation
    import video.runtime as video_runtime
    from core.jobs import JobHandle

    frames_dir = Path(args.frames_dir)
    events, stage_events, lock = [], [], threading.Lock()

    def timed(name, function):
        @functools.wraps(function)
        def wrapper(*a, **kw):
            started = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                sync()
                with lock:
                    events.append((name, started, time.perf_counter()))
        return wrapper

    masker_class = sam2_video_masker.SAM2VideoMasker
    for owner, attribute, name in [
        (masker_class, "__init__", "sam2: load model weights"),
        (masker_class, "init_state", "sam2: init_state (load frames)"),
        (masker_class, "add_new_points_or_box", "sam2: add points"),
        (masker_class, "add_new_mask", "sam2: add mask"),
        (masker_class, "reset_state", "sam2: reset_state"),
        (masker_class, "propagate_in_video", "sam2: propagate"),
        (masker_class, "offload_weights", "sam2: weights to CPU"),
        (masker_class, "reload_weights", "sam2: weights to GPU"),
        (co_tracker.CoTracker, "__init__", "cotracker: load model weights"),
        (co_tracker.CoTracker, "prepare_video", "cotracker: prepare video"),
        (co_tracker.CoTracker, "track", "cotracker: track"),
        (video_runtime.VideoMaskerRuntime, "initialize", "runtime: build + init_state + replay prompts"),
        (video_runtime.VideoMaskerRuntime, "rebuild_inputs", "runtime: reset and replay the accepted inputs"),
        (propagation, "open_run", "propagation: open_run (outputs, manifest)"),
        (propagation, "publish_run", "propagation: publish_run (manifest write)"),
        (prompt_tracking, "load_tracking_video_from_current_video_state", "tracking: load every frame into memory"),
    ]:
        setattr(owner, attribute, timed(name, getattr(owner, attribute)))

    real_enter_stage = JobHandle.enter_stage

    def enter_stage(self, stage, *a, **kw):
        with lock:
            stage_events.append((time.perf_counter(), stage))
        return real_enter_stage(self, stage, *a, **kw)

    JobHandle.enter_stage = enter_stage
    harness = Harness()
    results = []

    def flow(name, method, path, wait=False, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        with lock:
            first_event, first_stage = len(events), len(stage_events)
        started = time.perf_counter()
        response = harness.client.request(method, path, **kwargs)
        if wait:
            harness.wait_for_job()
        sync()
        ended = time.perf_counter()
        with lock:
            window = [(n, s, e) for n, s, e in events[first_event:] if s >= started and e <= ended]
            stages = [(t, s) for t, s in stage_events[first_stage:] if started <= t <= ended]
        by_name = collections.OrderedDict()
        for n, s, e in sorted(window, key=lambda item: item[1]):
            entry = by_name.setdefault(n, [0, 0.0])
            entry[0] += 1
            entry[1] += e - s
        stage_seconds = collections.OrderedDict()
        for index, (t, stage) in enumerate(stages):
            following = stages[index + 1][0] if index + 1 < len(stages) else ended
            stage_seconds[stage] = stage_seconds.get(stage, 0.0) + (following - t)
        peak = torch.cuda.max_memory_allocated() / 1e6
        results.append({"flow": name, "status": response.status_code, "wall_s": ended - started, "peak_gpu_mb": peak,
                        "calls": {n: {"count": c, "seconds": t} for n, (c, t) in by_name.items()},
                        "stages": dict(stage_seconds)})
        print(f"\n{name}: {ended - started:.2f} s wall, peak GPU {peak:,.0f} MB (HTTP {response.status_code})")
        for n, (c, t) in by_name.items():
            print(f"    {t:7.2f} s  {c:4d}x  {n}")
        if stage_seconds:
            print("    job stages: " + ", ".join(f"{k} {v:.1f}s" for k, v in stage_seconds.items() if v >= 0.05))
        return response

    try:
        body = init_body(frames_dir)
        point = center_point(frames_dir)
        flow("init_state, first in this process", "POST", "/video/init_state", wait=True, json=body)
        flow("init_state, again on the same video (weights shared)", "POST", "/video/init_state", wait=True, json=body)
        clicks = []
        for index in range(5):
            started = time.perf_counter()
            response = harness.client.post(
                "/video/add_new_points_or_box", headers=harness.headers(),
                json={"frame_idx": 0, "obj_id": 1, "points": [[point[0] + 12 * index, point[1]]], "labels": [1]})
            sync()
            clicks.append((time.perf_counter() - started) * 1000)
            print(f"click {index + 1}: {clicks[-1]:.0f} ms (HTTP {response.status_code})")
        print(f"interactive click latency: median {statistics.median(clicks):.0f} ms, max {max(clicks):.0f} ms")
        results.append({"flow": "interactive clicks (ms)", "samples": clicks})
        flow("propagate, whole clip", "POST", "/video/propagate_in_video", wait=True, headers=harness.headers(),
             json={"use_tracked_points": False, "online_mode": True, "batch_size": 32})
        flow("prompt tracking (in memory, no support grid)", "POST", "/tracking/track_prompt_points", wait=True,
             headers=harness.headers(), json={"add_support_grid": False})
        saved = flow("save", "POST", "/video/save", headers=harness.headers(), json={"name": "measure"})
        saved_path = saved.json().get("saved_path") if saved.status_code == 200 else str(harness.temp / "missing")
        flow("reopen the checkpoint", "POST", "/video/init_state", wait=True, json=init_body(saved_path))
    finally:
        JobHandle.enter_stage = real_enter_stage
        harness.close()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\nwrote", args.json)


# ---- memory: resident memory and GPU memory around the heavy flows -------------------------------------------------------
def command_memory(args):
    try:
        import psutil
    except ImportError:
        sys.exit("memory needs psutil in the project's virtual environment")
    import torch
    import sam2_video_masker
    import video.runtime as video_runtime

    if args.async_loading:
        real_init = sam2_video_masker.SAM2VideoMasker.init_state

        @functools.wraps(real_init)
        def async_init(self, *a, **kw):
            kw["async_loading_frames"] = True
            return real_init(self, *a, **kw)

        sam2_video_masker.SAM2VideoMasker.init_state = async_init
    if args.no_offload:
        video_runtime.VideoMaskerRuntime.offloaded = lambda self: contextlib.nullcontext()

    process = psutil.Process()
    timeline, phase, stop = [], ["start"], threading.Event()
    started_at = time.perf_counter()

    def sampler():
        while not stop.is_set():
            timeline.append((time.perf_counter() - started_at, process.memory_info().rss / 1e6,
                             torch.cuda.memory_allocated() / 1e6, phase[0]))
            if psutil.virtual_memory().available < args.min_free_gb * 1e9:
                print(f"WATCHDOG: free RAM below {args.min_free_gb} GB during '{phase[0]}', exiting", flush=True)
                os._exit(3)
            time.sleep(0.05)

    threading.Thread(target=sampler, daemon=True).start()
    harness = Harness()

    def settle(limit=60):
        """Wait until resident memory stops changing, e.g. after a background load; returns MB."""
        last, changed, begun = process.memory_info().rss / 1e6, time.perf_counter(), time.perf_counter()
        while time.perf_counter() - begun < limit:
            time.sleep(0.25)
            now = process.memory_info().rss / 1e6
            if abs(now - last) > 60:
                last, changed = now, time.perf_counter()
            elif time.perf_counter() - changed > 3:
                break
        return process.memory_info().rss / 1e6

    rows = []

    def step(name, action):
        phase[0] = name
        torch.cuda.reset_peak_memory_stats()
        begun = time.perf_counter()
        response = action()
        harness.wait_for_job()
        sync()
        wall = time.perf_counter() - begun
        settled = settle()
        samples = [(rss, gpu) for _, rss, gpu, p in timeline if p == name]
        row = {"step": name, "http": response.status_code, "job": harness.job_status(), "wall_s": round(wall, 2),
               "rss_settled_mb": round(settled), "rss_peak_mb": round(max(r for r, _ in samples)) if samples else None,
               "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
               "gpu_now_mb": round(torch.cuda.memory_allocated() / 1e6)}
        rows.append(row)
        print(f"{name:34} {wall:7.2f} s  HTTP {response.status_code:3d}  RSS settled {row['rss_settled_mb']:6,} MB  "
              f"peak {row['rss_peak_mb']:6,} MB  GPU peak {row['gpu_peak_mb']:6,} MB", flush=True)

    first = Path(args.frames_dir)
    current = first
    try:
        print(f"RSS before anything: {process.memory_info().rss / 1e6:,.0f} MB")
        step(f"open {first.name}", lambda: harness.client.post("/video/init_state", json=init_body(first)))
        if args.second_dir:
            current = Path(args.second_dir)
            step(f"open {current.name} (a different video)",
                 lambda: harness.client.post("/video/init_state", json=init_body(current)))
        point = center_point(current)
        harness.client.post("/video/add_new_points_or_box", headers=harness.headers(),
                            json={"frame_idx": 0, "obj_id": 1, "points": [point], "labels": [1]})
        step("track prompts", lambda: harness.client.post(
            "/tracking/track_prompt_points", headers=harness.headers(), json={"add_support_grid": False}))
        step("click after tracking", lambda: harness.client.post(
            "/video/add_new_points_or_box", headers=harness.headers(),
            json={"frame_idx": 0, "obj_id": 1, "points": [point], "labels": [1]}))
        step("propagate after tracking", lambda: harness.client.post(
            "/video/propagate_in_video", headers=harness.headers(),
            json={"use_tracked_points": False, "online_mode": True, "batch_size": 32}))
    finally:
        stop.set()
        time.sleep(0.2)
        harness.close()
    overall = round(max(rss for _, rss, _, _ in timeline))
    print(f"overall peak RSS: {overall:,} MB")
    if args.json:
        Path(args.json).write_text(json.dumps({"steps": rows, "overall_peak_rss_mb": overall,
                                               "async_loading": args.async_loading, "offload": not args.no_offload},
                                              indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    hops = commands.add_parser("hops", help="calls and hops per endpoint, with fake models")
    hops.add_argument("--frames", type=int, nargs="+", default=[12], help="clip lengths; two give a per-frame cost")
    hops.add_argument("--json")
    hops.set_defaults(run=command_hops)

    flows = commands.add_parser("flows", help="time per flow, real models")
    flows.add_argument("--frames-dir", default=str(DEFAULT_FRAMES_DIR), help="folder of .jpg frames")
    flows.add_argument("--json")
    flows.set_defaults(run=command_flows)

    memory = commands.add_parser("memory", help="resident and GPU memory around the heavy flows, real models")
    memory.add_argument("--frames-dir", required=True, help="folder of .jpg frames")
    memory.add_argument("--second-dir", help="a different video to open second")
    memory.add_argument("--async-loading", action="store_true", help="use SAM2's async frame loading")
    memory.add_argument("--no-offload", action="store_true", help="keep SAM2's weights on the GPU during tracking")
    memory.add_argument("--min-free-gb", type=float, default=3.5, help="exit if free system RAM drops below this")
    memory.add_argument("--json")
    memory.set_defaults(run=command_memory)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
