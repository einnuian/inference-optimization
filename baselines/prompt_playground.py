# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt the repository's Gradio chatbot example for the prepared HPML methods.

Source: external/vllm/examples/applications/chatbot/gradio_openai_chatbot_webserver.py
Changes: one user message, fixed trial settings, streaming display, local-only
hosting, explicit method identity, prepared-prompt preview and allocation records.
"""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
from openai import OpenAI
from trial_recorder import CONFIGURATIONS


LABELS = {"ar-sglang": "Autoregressive · SGLang", "eagle3": "EAGLE-3 · SGLang",
          "dflash": "DFlash · SGLang", "ar-vllm-v1": "Autoregressive · vLLM V1",
          "peagle": "P-EAGLE · H-Spec reproduction", "ar-vllm-v2": "Autoregressive · vLLM V2",
          "dspark": "DSpark · fixed-length H-Spec reproduction"}
BENCHMARK_LABELS = {"mt-bench": "MT-Bench", "alpaca": "Alpaca", "arena-hard-v2": "Arena-Hard v2",
                    "cnn-dm": "CNN/DailyMail", "gsm8k": "GSM8K", "math500": "MATH-500",
                    "aime25": "AIME2025", "humaneval": "HumanEval", "mbpp": "MBPP",
                    "livecodebench": "LiveCodeBench"}


def allocation_view(path):
    if not path.exists():
        return "Run the cluster accounting command to load the allocation record.", []
    record = json.loads(path.read_text())
    totals = record["project_totals"]
    description = (
        "The project uses **%.4f allocated GPU-hours** and **%.4f allocated CPU-core-hours** across **%d jobs**. "
        "The record covers allocations returned by Slurm from **%s** and was collected at **%s**.\n\n"
        "One GPU-hour means one GPU allocated for one hour. One CPU-core-hour means one CPU core allocated for one hour. "
        "Allocated time includes setup, model loading, generation and idle time within each job. "
        "Completed, failed and cancelled jobs all contribute their elapsed allocation time.\n\n"
        "Project jobs have names beginning `hpml-`. The preparation job `14056242` is also included. "
        "The table includes every returned job and identifies which jobs contribute to the project totals. "
        "The record uses each parent allocation once. The figures describe whole-job allocations returned for the date range."
    ) % (totals["allocated_gpu_hours"], totals["allocated_cpu_core_hours"], totals["jobs"],
         record["since"], record["observed_at"])
    rows = [[row["JobIDRaw"], row["JobName"], "Yes" if row["project_job"] else "No", row["State"],
             row["Start"], round(row["elapsed_seconds"] / 60, 2), row["allocated_gpus"],
             round(row["allocated_gpu_hours"], 4), row["allocated_cpus"],
             round(row["allocated_cpu_core_hours"], 4)] for row in reversed(record["jobs"])]
    return description, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--usage", type=Path, default=Path("measurements/compute-usage.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Show prompts and compute records while the model is stopped.")
    mode.add_argument("--configuration", choices=CONFIGURATIONS)
    parser.add_argument("--model-url", default="http://127.0.0.1:31100/v1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if urlsplit(args.model_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Use the local SSH-forwarded model port.")
    bundle = json.loads(args.prepared.read_text())
    prompts = {row["id"]: row for row in bundle["prompts"]}
    client = OpenAI(api_key="EMPTY", base_url=args.model_url, max_retries=0, timeout=900)
    label = LABELS.get(args.configuration, "The interface is in preview mode.")
    if args.configuration:
        models = [model.id for model in client.models.list().data]
        if "hpml-target" not in models:
            raise RuntimeError("The forwarded server must expose the prepared hpml-target model.")

    def predict(message, history):
        if args.preview:
            raise gr.Error("Start a prepared model server and reopen the interface with its configuration.")
        print("\nManual prompt | " + label + "\n" + message, flush=True)
        response = ""
        with client.chat.completions.create(
            model="hpml-target", messages=[{"role": "user", "content": message}],
            temperature=0, top_p=1, max_tokens=2048, seed=0, stream=True,
            extra_body={"stop_token_ids": bundle["stop_token_ids"], "repetition_penalty": 1},
        ) as stream:
            for chunk in stream:
                if chunk.choices:
                    response += chunk.choices[0].delta.content or ""
                    yield response
        print("The response is complete.\n" + response, flush=True)

    def prompt_view(identifier):
        row = prompts[identifier]
        return row["messages"][0]["content"], "Benchmark: %s. Prompt ID: %s. Input length: %d tokens." % (
            BENCHMARK_LABELS[row["benchmark"]], identifier, len(row["input_token_ids"]))

    initial = next(iter(prompts))
    prompt_text, prompt_description = prompt_view(initial)
    usage_description, usage_rows = allocation_view(args.usage)
    connection_status = ("The interface is in preview mode. Connect a prepared model server to enable your own prompts."
                         if args.preview else "The chat interface is connected to " + label + ".")
    with gr.Blocks(title="HPML prompts and compute", analytics_enabled=False) as demo:
        gr.Markdown("# HPML prompts and compute\nYou can inspect the trial prompts, enter your own prompts, and review cluster allocations.")
        with gr.Tab("Overview"):
            gr.Markdown("## Current progress\n"
                        "The 30 trial prompts are prepared. Each benchmark contributes three prompts. "
                        + connection_status + "\n\n"
                        "**Trial prompts** shows the exact text and input length for each prompt. "
                        "**Your prompts** provides the chat interface for a running method. "
                        "**Cluster compute** shows the recorded cluster allocations and their totals.\n\n"
                        "## How timing works\n"
                        "The measurement program records the time when it submits a prompt and when the first generated text arrives. "
                        "The difference is time to first token, or TTFT. For example, a request submitted at 10.000 seconds "
                        "with its first text arriving at 10.420 seconds has a TTFT of 420 milliseconds.\n\n"
                        "The program records these times on the cluster computer that runs the model. "
                        "Safari provides the view you follow.\n\n"
                        "## Preparation still in progress\n"
                        "The next checks will verify how each model server sends its generated text and token IDs. "
                        "Additional recording code will count the proposed tokens that the main model accepts. "
                        "That code will also measure the time spent drafting and checking proposals, "
                        "and the memory used inside each model process. The trial needs a browser view of each submission and its progress.")
        with gr.Tab("Trial prompts"):
            gr.Markdown("The trial uses the first three prompts from each of ten benchmarks. Each request contains one user message. "
                        "Select a prompt below to read the exact text that the model will receive.")
            choices = [("%s · prompt %d" % (BENCHMARK_LABELS[row["benchmark"]], index % 3 + 1), identifier)
                       for index, (identifier, row) in enumerate(prompts.items())]
            selector = gr.Dropdown(choices=choices, value=initial, label="Select a trial prompt")
            detail = gr.Markdown(prompt_description)
            content = gr.Textbox(value=prompt_text, label="Exact prompt", lines=12, interactive=False)
            selector.change(prompt_view, selector, [content, detail])
        with gr.Tab("Your prompts"):
            gr.Markdown("## " + label)
            gr.Markdown("The interface uses temperature 0 and a limit of 2,048 output tokens. "
                        "Each submission sends your latest message as a single turn. Previous messages stay visible as a record. "
                        "The browser displays the response as text arrives. "
                        "The cluster recorder collects benchmark timing on the compute node.")
            if args.preview:
                gr.Markdown("Open `baselines/measurement-runbook.md` in VS Code for the server and interface launch commands. "
                            "Start the model server before entering a prompt here.")
            gr.ChatInterface(fn=predict, textbox=gr.Textbox(label="Your prompt", interactive=not args.preview),
                             submit_btn=False if args.preview else "Send prompt")
        with gr.Tab("Cluster compute"):
            explanation = gr.Markdown(usage_description)
            table = gr.Dataframe(value=usage_rows,
                                headers=["Job", "Job name", "Project job", "State", "Started", "Minutes",
                                         "GPUs", "GPU-hours", "CPU cores", "CPU-core-hours"],
                                interactive=False)
            gr.Markdown("Run the accounting refresh commands in the visible VS Code terminals, then select **Reload the saved record**.")
            gr.Button("Reload the saved record").click(lambda: allocation_view(args.usage), outputs=[explanation, table])
    print("Opening the local interface at http://127.0.0.1:%d. Press Ctrl+C in this terminal to stop." % args.port, flush=True)
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=False, theme=gr.themes.Monochrome())


if __name__ == "__main__":
    main()
