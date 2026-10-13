"""Run a MiniMaxH3Infinity API workflow one window per prompt, so every window starts from freshly loaded models.

Usage:
    python tools/run_per_window.py workflow_api.json --windows 9 [--run-folder <existing run folder>]
                                   [--output-dir <ComfyUI output dir>] [--server http://127.0.0.1:8188]

The workflow needs the MiniMaxH3Infinity node (found by class_type) and ideally a FreeMemoryAfterRun node.
The script sets windows_per_run=1, submits the prompt, waits for it, and queues it again with resume_from set to
the run folder until <windows> chunk files exist. Pass --run-folder to continue an interrupted run.
"""
import argparse
import json
import os
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("workflow")
parser.add_argument("--windows", type=int, required=True, help="total number of windows of the plan")
parser.add_argument("--run-folder", default="")
parser.add_argument("--output-dir", default="output", help="ComfyUI output directory")
parser.add_argument("--server", default="http://127.0.0.1:8188")
args = parser.parse_args()

partial = os.path.join(args.output_dir, "MiniMaxH3_Infinity_partial")
wf = json.load(open(args.workflow, encoding="utf-8"))
infinity = next(n for n in wf.values() if n["class_type"] == "MiniMaxH3Infinity")
infinity["inputs"]["windows_per_run"] = 1


def get(path):
    return json.load(urllib.request.urlopen(args.server + path, timeout=120))


def submit():
    req = urllib.request.Request(args.server + "/prompt", data=json.dumps({"prompt": wf}).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]


def chunks(run):
    return len([f for f in os.listdir(run) if f.startswith("chunk_")]) if run and os.path.isdir(run) else 0


run = args.run_folder
while chunks(run) < args.windows:
    known = set(os.listdir(partial)) if os.path.isdir(partial) else set()
    infinity["inputs"]["resume_from"] = os.path.abspath(run) if run else ""
    pid = submit()
    print(time.strftime("%H:%M:%S"), f"submitted {pid} ({chunks(run)} windows finished)", flush=True)
    while pid not in (history := get("/history/" + pid)):
        time.sleep(15)
    status = history[pid]["status"]
    if status["status_str"] != "success":
        raise SystemExit("prompt failed: " + json.dumps(status)[:800])
    if not run:
        run = os.path.join(partial, max(set(os.listdir(partial)) - known, key=lambda d: os.path.getmtime(os.path.join(partial, d))))
    print(time.strftime("%H:%M:%S"), f"{chunks(run)}/{args.windows} windows finished in {run}", flush=True)
print("all windows finished:", run)
