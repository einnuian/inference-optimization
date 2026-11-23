#!/usr/bin/env python3
"""Show saved trial progress through a local, read-only browser interface."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit


CONFIGURATIONS = (
    ("ar-sglang", "Autoregressive · SGLang"),
    ("eagle3", "EAGLE-3 · SGLang"),
    ("dflash", "DFlash · SGLang"),
    ("ar-vllm-v1", "Autoregressive · vLLM V1"),
    ("peagle", "P-EAGLE · H-Spec reproduction"),
    ("ar-vllm-v2", "Autoregressive · vLLM V2"),
    ("dspark", "DSpark · fixed-length H-Spec reproduction"),
)
BENCHMARKS = {
    "mt-bench": "MT-Bench", "alpaca": "Alpaca", "arena-hard-v2": "Arena-Hard v2",
    "cnn-dm": "CNN/DailyMail", "gsm8k": "GSM8K", "math500": "MATH-500",
    "aime25": "AIME2025", "humaneval": "HumanEval", "mbpp": "MBPP",
    "livecodebench": "LiveCodeBench",
}


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def load_prompts(path):
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = bundle["prompts"]
    if len(rows) != 30 or len({row["id"] for row in rows}) != 30:
        raise ValueError("The browser view requires 30 distinct prepared prompts.")
    if Counter(row["benchmark"] for row in rows) != Counter({name: 3 for name in BENCHMARKS}):
        raise ValueError("Each of the ten benchmarks must contribute three prompts.")
    counts = Counter()
    prompts = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError("Each prepared prompt must contain one user message.")
        counts[row["benchmark"]] += 1
        prompts.append({"index": index, "id": row["id"], "benchmark": row["benchmark"],
                        "benchmark_label": BENCHMARKS[row["benchmark"]],
                        "label": "%s · prompt %d" % (BENCHMARKS[row["benchmark"]], counts[row["benchmark"]]),
                        "text": messages[0]["content"], "input_tokens": len(row["input_token_ids"])})
    return prompts


class TrialStore:
    """Cache immutable results and incrementally read the recorder's progress files."""

    def __init__(self, prepared, results, refresh_seconds=1.0):
        self.prompts = load_prompts(prepared)
        self.prompt_ids = {row["id"] for row in self.prompts}
        self.prompt_benchmarks = {row["id"]: row["benchmark"] for row in self.prompts}
        self.root = Path(results).resolve()
        self.refresh_seconds = refresh_seconds
        self.lock = threading.Lock()
        self.last_refresh = float("-inf")
        self.records = {}
        self.events = {}
        self.metadata = {}
        self.warmups = set()
        self.read_files = set()
        self.live_files = {}
        self.progress = {}
        self.progress_signature = None

    def _safe_file(self, path):
        try:
            return path.is_file() and path.resolve().is_relative_to(self.root)
        except (OSError, RuntimeError):
            return False

    def _json(self, path):
        if not self._safe_file(path):
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, UnicodeError):
            return None

    def _read_live(self, configuration, path):
        if not self._safe_file(path):
            return
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            old = self.live_files.get(configuration)
            if old is None or old["identity"] != identity or stat.st_size < old["offset"]:
                old = {"identity": identity, "offset": 0, "pending": b"", "mtime": None}
                self.live_files[configuration] = old
                self.events = {key: value for key, value in self.events.items() if key[0] != configuration}
            if stat.st_size == old["offset"] and stat.st_mtime_ns == old["mtime"]:
                return
            with path.open("rb") as handle:
                handle.seek(old["offset"])
                data = handle.read(4 * 1024 * 1024)
                old["offset"] = handle.tell()
            parts = (old["pending"] + data).split(b"\n")
            old["pending"] = parts.pop()
            if len(old["pending"]) > 1024 * 1024:
                old["pending"] = b""
            old["mtime"] = stat.st_mtime_ns
            for line in parts:
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(event, dict) or event.get("configuration") != configuration:
                    continue
                if event.get("event") not in {"submitted", "completed"}:
                    continue
                repetition, identifier = event.get("repetition"), event.get("prompt_id")
                if type(repetition) is not int or repetition not in {0, 1, 2, 3} or not isinstance(identifier, str):
                    continue
                if repetition != 0 and identifier not in self.prompt_ids:
                    continue
                if repetition == 0 and event["event"] == "completed":
                    self.warmups.add((configuration, identifier))
                key = (configuration, identifier, repetition)
                current = self.events.setdefault(key, {})
                old_event = current.get(event["event"], {})
                observed = finite_number(event.get("unix_ns")) or 0
                previous = finite_number(old_event.get("unix_ns")) or 0
                if observed >= previous:
                    current[event["event"]] = event
        except OSError:
            return

    def _read_results(self, configuration):
        directory = self.root / configuration
        if not directory.is_dir():
            return
        metadata_path = directory / "run.json"
        if configuration not in self.metadata:
            metadata = self._json(metadata_path)
            if isinstance(metadata, dict):
                self.metadata[configuration] = {key: metadata.get(key) for key in (
                    "configuration", "backend", "model", "hostname", "slurm_job_id", "gpu_uuid", "target_revision")}
        warmup_path = directory / "warmup.json"
        if warmup_path not in self.read_files:
            warmups = self._json(warmup_path)
            if isinstance(warmups, list):
                for index, row in enumerate(warmups):
                    identifier = row.get("prompt_id", "warmup-%d" % (index + 1)) if isinstance(row, dict) else None
                    if identifier:
                        self.warmups.add((configuration, identifier))
                self.read_files.add(warmup_path)
        for repetition in range(1, 4):
            for benchmark in BENCHMARKS:
                path = directory / ("%02d-%s.json" % (repetition, benchmark))
                if path in self.read_files:
                    continue
                block = self._json(path)
                if not isinstance(block, dict) or block.get("repetition") != repetition or block.get("benchmark") != benchmark:
                    continue
                rows = block.get("requests")
                if not isinstance(rows, list) or len(rows) != 3:
                    continue
                staged = {}
                for row in rows:
                    if (not isinstance(row, dict) or self.prompt_benchmarks.get(row.get("prompt_id")) != benchmark
                            or not isinstance(row.get("output_text"), str)):
                        break
                    identifier = row["prompt_id"]
                    staged[(configuration, identifier, repetition)] = {
                        "request_id": row.get("request_id"), "output_text": row.get("output_text", ""),
                        "output_tokens": finite_number(row.get("output_tokens")),
                        "ttft_seconds": finite_number(row.get("ttft_seconds")),
                        "end_to_end_seconds": finite_number(row.get("end_to_end_seconds")),
                        "output_rate": finite_number(row.get("output_rate")),
                        "finish_reason": row.get("finish_reason"),
                    }
                else:
                    if len(staged) == 3:
                        self.records.update(staged)
                        self.read_files.add(path)

    def _refresh(self):
        now = time.monotonic()
        if now - self.last_refresh < self.refresh_seconds:
            return
        self.last_refresh = now
        path = self.root / "progress.json"
        try:
            stat = path.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if signature != self.progress_signature:
                value = self._json(path)
                if isinstance(value, dict):
                    self.progress = {key: str(value.get(key, "")) for key in ("phase", "message")}
                    self.progress_signature = signature
        except OSError:
            pass
        for configuration, _ in CONFIGURATIONS:
            self._read_live(configuration, self.root / configuration / "live.jsonl")
            self._read_results(configuration)

    def snapshot(self, prompt_index=None):
        with self.lock:
            self._refresh()
            completed = set(self.records)
            completed.update(key for key, value in self.events.items() if key[2] in {1, 2, 3} and "completed" in value)
            prompt_counts = Counter(key[1] for key in completed)
            pending = []
            for key, value in self.events.items():
                submitted = value.get("submitted")
                finished = value.get("completed")
                warmup_saved = key[2] == 0 and (key[0], key[1]) in self.warmups
                if submitted and key not in self.records and not warmup_saved and (not finished or submitted.get("request_id") != finished.get("request_id")):
                    pending.append({"configuration": key[0], "prompt_id": key[1], "repetition": key[2],
                                    "unix_ns": finite_number(submitted.get("unix_ns")) or 0})
            result = {"observed_at": datetime.now(timezone.utc).isoformat(), "progress": self.progress.copy(),
                      "completed_requests": len(completed), "saved_requests": len(self.records), "total_requests": 630,
                      "warmup_completed": len(self.warmups), "total_warmups": 21,
                      "current": max(pending, key=lambda row: row["unix_ns"]) if pending else None,
                      "prompts": [{"index": row["index"], "label": row["label"], "completed": prompt_counts[row["id"]]}
                                  for row in self.prompts], "configurations": self.metadata.copy()}
            if prompt_index is not None:
                prompt = self.prompts[prompt_index]
                result["prompt"] = prompt.copy()
                result["rows"] = []
                for configuration, label in CONFIGURATIONS:
                    for repetition in range(1, 4):
                        key = (configuration, prompt["id"], repetition)
                        record = self.records.get(key)
                        events = self.events.get(key, {})
                        finished, submitted = events.get("completed", {}), events.get("submitted", {})
                        values = record or finished
                        result["rows"].append({
                            "key": "%s/%d" % (configuration, repetition), "configuration": configuration,
                            "label": label, "repetition": repetition,
                            "status": "saved" if record else "completed" if finished else "submitted" if submitted else "waiting",
                            "submitted_unix_ns": finite_number(submitted.get("unix_ns")),
                            "completed_unix_ns": finite_number(finished.get("unix_ns")),
                            "output_tokens": finite_number(values.get("output_tokens")),
                            "ttft_seconds": finite_number(values.get("ttft_seconds")),
                            "end_to_end_seconds": finite_number(values.get("end_to_end_seconds")),
                            "output_rate": finite_number(values.get("output_rate")),
                            "finish_reason": values.get("finish_reason"),
                            "output_text": record.get("output_text", "") if record else None,
                        })
            return result


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HPML trial</title><style nonce="__NONCE__">
:root{font-family:Arial,Helvetica,sans-serif;font-size:16px;color:#171717;background:#fff;color-scheme:light}
*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:32px 24px 64px}
h1{font-size:2rem;line-height:1.2;margin:20px 0}h2{font-size:1.35rem;margin:30px 0 14px}
p,li{line-height:1.6}p{max-width:90ch}a{color:inherit;text-decoration:underline;text-underline-offset:3px}
nav{display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid #aaa;padding-bottom:16px}
.status{padding:16px 20px;background:#f3f3f3;border:1px solid #ccc;margin:24px 0}.status p{margin:4px 0}
.note{color:#444}.prompt{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.6;border:1px solid #bbb;padding:20px;max-height:400px;overflow:auto}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:920px;font-size:1rem}
th,td{text-align:left;vertical-align:top;border-bottom:1px solid #ccc;padding:12px 10px;line-height:1.4}
th{background:#eee;font-weight:600}tbody tr:nth-child(even){background:#fafafa}
button{font:inherit;color:inherit;background:#fff;border:1px solid #555;border-radius:3px;padding:7px 10px;cursor:pointer}
button:disabled{color:#666;border-color:#bbb;cursor:default}button:focus-visible,a:focus-visible{outline:2px solid #111;outline-offset:3px}
.choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;padding:0;list-style:none}
.choices li{border:1px solid #ccc;padding:16px}.choices span{display:block;margin-top:6px;color:#444}
#connection{font-weight:600}.response{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.6;padding:20px;background:#f7f7f7;border:1px solid #ccc;max-height:650px;overflow:auto}
[hidden]{display:none!important}small{font-size:1rem}footer{border-top:1px solid #aaa;margin-top:32px;padding-top:16px}
@media(max-width:600px){main{padding:20px 16px}h1{font-size:1.7rem}.status{padding:12px}}
</style></head><body><main>
<nav><a href="/">All trial prompts</a><span id="location">The trial contains 30 prepared prompts.</span></nav>
<h1 id="title">HPML trial progress</h1>
<p id="intro">The recorder submits one request at a time. Each prompt has three measured repetitions across seven method and reference configurations.</p>
<section class="status" aria-label="Trial progress"><p id="connection" role="status">The page is reading the saved trial progress.</p>
<p id="progress"></p><p id="warmups"></p><p id="current"></p><p id="phase"></p><p id="refreshed" class="note"></p></section>
<section id="index-section"><h2>Choose a prompt</h2><p>Each link opens a page for one prepared prompt. You can keep each prompt page in its own Safari tab.</p><ul id="prompt-list" class="choices"></ul></section>
<section id="detail-section" hidden><h2>Exact prompt</h2><p id="prompt-info" class="note"></p><pre id="prompt" class="prompt"></pre>
<h2>Measured repetitions</h2><p>Each row shows one request. Time to first token (TTFT) is the time between submitting the prompt and receiving the first generated text. The recorder measures both events on the model's compute node. Completion time covers the full response.</p>
<p>Request output rate divides that request's returned output tokens by its completion time. Output tokens per second (OTPS) uses the complete timed group of three requests. The saved group result contains OTPS.</p>
<p class="note">The recorder saves the complete response text when the current group of three prompts finishes. Submission and completion events can appear before that saved response.</p>
<div class="scroll"><table><thead><tr><th scope="col">Method and reference</th><th scope="col">Repetition</th><th scope="col">Progress</th><th scope="col">TTFT (ms)</th><th scope="col">Completion time (s)</th><th scope="col">Output tokens</th><th scope="col">Request output rate (tokens/s)</th><th scope="col">Response</th></tr></thead><tbody id="results"></tbody></table></div>
<h2 id="response-title">Response text</h2><p id="response-info">Select a saved response to read the generated text.</p><pre id="response" class="response" hidden></pre></section>
<footer><p>The model's compute node records the measurement timestamps. These browser pages read the recorded progress and results. The pages send no model requests.</p></footer>
</main><script nonce="__NONCE__" type="application/json" id="boot">__BOOT__</script><script nonce="__NONCE__">
'use strict';
const boot=JSON.parse(document.getElementById('boot').textContent);
const el=id=>document.getElementById(id);
const labels=Object.fromEntries(boot.configurations);
let busy=false,selected=null,lastRows=[],lastSignature=null,lastRowSignature=null;
const promptNodes=new Map(),rowNodes=new Map();
const number=(value,scale=1,digits=2)=>typeof value==='number'&&Number.isFinite(value)?(value*scale).toFixed(digits):'';
const localTime=ns=>typeof ns==='number'?new Date(ns/1e6).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}):'';
function element(tag,text){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;}
function text(node,value){if(node.textContent!==value)node.textContent=value;}
function showResponse(){const row=lastRows.find(item=>item.key===selected);if(!row||row.output_text===null)return;
text(el('response-title'),row.label+' · repetition '+row.repetition);
const stop=row.finish_reason==='length'?'The response reached the output limit.':row.finish_reason==='stop'?'The model reached its stopping condition.':'The saved record contains the response below.';
text(el('response-info'),stop);text(el('response'),typeof row.output_text==='string'?row.output_text:'');el('response').hidden=false;}
function renderRows(rows){const signature=JSON.stringify(rows);if(signature===lastRowSignature)return;lastRowSignature=signature;lastRows=rows;const body=el('results');
const meanings={waiting:'The request is waiting.',submitted:'The request is running.',completed:'The request has completed. The response record is being saved.',saved:'The response is saved.'};
for(const row of rows){let nodes=rowNodes.get(row.key);if(!nodes){const tr=element('tr'),status=element('td');nodes={status:element('div'),submitted:element('div'),completed:element('div'),ttft:element('td'),latency:element('td'),tokens:element('td'),rate:element('td'),button:element('button','Read response')};
status.append(nodes.status,nodes.submitted,nodes.completed);const cell=element('td');nodes.button.type='button';nodes.button.addEventListener('click',()=>{selected=row.key;showResponse();el('response-title').scrollIntoView({block:'start',behavior:'smooth'});});cell.append(nodes.button);tr.append(element('td',row.label),element('td',String(row.repetition)),status,nodes.ttft,nodes.latency,nodes.tokens,nodes.rate,cell);body.append(tr);rowNodes.set(row.key,nodes);}
text(nodes.status,meanings[row.status]);text(nodes.submitted,row.submitted_unix_ns?'Submitted at '+localTime(row.submitted_unix_ns)+'.':'');text(nodes.completed,row.completed_unix_ns?'Completed at '+localTime(row.completed_unix_ns)+'.':'');
text(nodes.ttft,number(row.ttft_seconds,1000,1));text(nodes.latency,number(row.end_to_end_seconds,1,3));text(nodes.tokens,row.output_tokens===null?'':String(row.output_tokens));text(nodes.rate,number(row.output_rate));nodes.button.disabled=row.output_text===null;}
showResponse();}
function connection(message){if(el('connection').textContent!==message)el('connection').textContent=message;}
function render(state){connection('The trial view is connected.');
el('progress').textContent=state.completed_requests+' of '+state.total_requests+' measured requests have completed. '+state.saved_requests+' complete response records are saved.';
el('warmups').textContent=state.warmup_completed+' of '+state.total_warmups+' separate warmup requests have completed.';
const current=state.current;el('current').textContent=current?(labels[current.configuration]||current.configuration)+' is processing '+current.prompt_id+(current.repetition?' in repetition '+current.repetition+'.':' during warmup.'):'The recorder has no pending submission in the saved progress record.';
el('phase').textContent=state.progress.message||'';el('phase').hidden=!state.progress.message;
el('refreshed').textContent='This view last updated at '+new Date(state.observed_at).toLocaleString()+'.';
if(boot.prompt_index===null){const list=el('prompt-list');for(const prompt of state.prompts){let count=promptNodes.get(prompt.index);if(!count){const li=element('li'),a=element('a',prompt.label);a.href='/prompt/'+prompt.index;count=element('span');li.append(a,count);list.append(li);promptNodes.set(prompt.index,count);}text(count,prompt.completed+' of 21 measured requests have completed.');}}
else renderRows(state.rows);}
if(boot.prompt_index!==null){const prompt=boot.prompt;document.title=prompt.label+' | HPML trial';el('title').textContent=prompt.label;el('location').textContent='Prompt '+(prompt.index+1)+' of 30';el('index-section').hidden=true;el('detail-section').hidden=false;el('prompt').textContent=prompt.text;el('prompt-info').textContent='Source ID: '+prompt.id+'. Input length: '+prompt.input_tokens+' tokens.';}
async function refresh(){if(document.hidden||busy)return;busy=true;try{const suffix=boot.prompt_index===null?'':'?prompt='+boot.prompt_index;const response=await fetch('/api/state'+suffix,{cache:'no-store'});if(!response.ok)throw new Error('The trial view is unavailable.');const state=await response.json();const {observed_at,...content}=state;const signature=JSON.stringify(content);if(signature!==lastSignature){render(state);lastSignature=signature;}else{connection('The trial view is connected.');}}catch(error){connection('The connection to the trial view is interrupted. The page will retry automatically.');}finally{busy=false;}}
document.addEventListener('visibilitychange',refresh);setInterval(refresh,2000);refresh();
</script></body></html>'''


def page_bytes(prompts, prompt_index, nonce):
    data = {"prompt_index": prompt_index, "configurations": CONFIGURATIONS,
            "prompt": prompts[prompt_index] if prompt_index is not None else None}
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return PAGE.replace("__BOOT__", encoded).replace("__NONCE__", nonce).encode("utf-8")


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HPMLTrialView/1"

        def log_message(self, format_string, *args):
            if len(args) > 1 and str(args[1]) != "200":
                print("Browser request: " + format_string % args, flush=True)

        def _send(self, body, content_type, status=200, nonce=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            if nonce:
                self.send_header("Content-Security-Policy", "default-src 'none'; connect-src 'self'; style-src 'nonce-%s'; script-src 'nonce-%s'; base-uri 'none'; frame-ancestors 'none'" % (nonce, nonce))
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            try:
                hostname = urlsplit("http://" + self.headers.get("Host", "")).hostname
            except ValueError:
                hostname = None
            if hostname not in {"127.0.0.1", "localhost", "::1"}:
                self._send(b"Use the local forwarded browser address.\n", "text/plain; charset=utf-8", 403)
                return
            route = urlsplit(self.path)
            if route.path == "/api/state":
                query = parse_qs(route.query)
                value = query.get("prompt", [None])
                if len(value) != 1 or (value[0] is not None and not re.fullmatch(r"(?:[0-9]|[12][0-9])", value[0])):
                    self._send(b"Choose a prompt number from 0 through 29.\n", "text/plain; charset=utf-8", 400)
                    return
                index = int(value[0]) if value[0] is not None else None
                body = json.dumps(store.snapshot(index), ensure_ascii=False, allow_nan=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
                return
            match = re.fullmatch(r"/prompt/(0|[1-9]|[12][0-9])", route.path)
            if route.path == "/" or match:
                index = int(match.group(1)) if match else None
                nonce = secrets.token_urlsafe(18)
                self._send(page_bytes(store.prompts, index, nonce), "text/html; charset=utf-8", nonce=nonce)
                return
            self._send(b"Open the trial overview or one of its prompt links.\n", "text/plain; charset=utf-8", 404)

        do_HEAD = do_GET

        def do_POST(self):
            self._send(b"The browser view accepts read-only requests.\n", "text/plain; charset=utf-8", 405)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31001)
    args = parser.parse_args()
    store = TrialStore(args.prepared, args.results)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store))
    server.daemon_threads = True
    base = "http://127.0.0.1:%d" % args.port
    print("The read-only trial view is running at " + base + "/", flush=True)
    print("Keep this foreground terminal open. Press Ctrl+C to stop the browser view.", flush=True)
    for prompt in store.prompts:
        print("%s: %s/prompt/%d" % (prompt["label"], base, prompt["index"]), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping the trial browser view.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
