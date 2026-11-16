# Running the P-EAGLE notebook on the Insomnia HPC cluster

This guide covers running `peagle_ffn.ipynb` on an Insomnia compute node and opening it in the browser on your laptop. It also covers storage limits and GPU memory management.

Placeholders are written as `<LIKE_THIS>`. Replace them with your own values.

---

## Before you begin

Complete these steps in [SETUP_EAGLE.md](SETUP_EAGLE.md) first:

- Create the `ea_env` virtual environment ([Shared EAGLE setup, step 2](SETUP_EAGLE.md#2-install-python-310-and-create-a-virtual-environment)).
- Get access to Llama-3.1-8B-Instruct on Hugging Face ([step 4](SETUP_EAGLE.md#4-hugging-face-access)).
- Move the Hugging Face cache to scratch space ([Move the Hugging Face cache to scratch space](SETUP_EAGLE.md#move-the-hugging-face-cache-to-scratch-space)). The 16 GB model doesn't fit in your home directory alongside everything else.

You also need a [Weights & Biases](https://wandb.ai) account if you keep `USE_WANDB = True` in the notebook.

---

## 1. Install the notebook packages

Run these commands once, from a terminal on a login node or a compute node:

```bash
source ~/venvs/ea_env/bin/activate
uv pip install "transformers==4.57.6" "datasets==3.6.0" "huggingface_hub<1.0" wandb jupyterlab ipywidgets
hash -r                     # make the shell find the new jupyter
which jupyter               # expect: ~/venvs/ea_env/bin/jupyter
```

**Install Jupyter inside the venv.** If `which jupyter` prints `~/.local/bin/jupyter` or a system path, the shell is using a different Jupyter. That Jupyter starts notebooks on the system Python 3.9, not on `ea_env`, and package installs go to `~/.local` instead of the venv.

**Don't use `!pip install` in the notebook.** `uv venv` creates environments without pip, so `!pip` runs the system pip and installs into `~/.local/lib/python3.9`. Install packages from a terminal as shown above. To install from a notebook cell, use this instead:

```python
import sys
!uv pip install --python {sys.executable} <PACKAGE>
```

**Note:** The notebook needs `transformers==4.57.6`, and installing it replaces the version that EAGLE uses in `ea_env`. If the EAGLE web UI or MT-Bench scripts break afterward, reinstall the EAGLE version with `uv pip install "transformers==4.53.1"` before running them.

---

## 2. Keep notebook outputs out of your home directory

Your Insomnia home directory has two limits, and exceeding either one causes `Disk quota exceeded`:

- **Space:** the total size of your files.
- **Files:** the number of files and folders (150,000 at the time of writing). Python environments and caches hold many small files, so this limit is often the one you hit first.

To see what uses your file count:

```bash
for d in ~/.[!.]* ~/*; do echo "$(find "$d" 2>/dev/null | wc -l) $d"; done | sort -n | tail
```

The notebook saves training data and drafter checkpoints to the folder in `OUT`. Each checkpoint holds the drafter weights and the optimizer state, so the three checkpoints take about 26 GB:

| Drafter width | Checkpoint size |
|---|---|
| 100% | about 10.9 GB |
| 75% | about 8.8 GB |
| 50% | about 6.6 GB |

The notebook sets `OUT` to a folder in your home directory by default. Before your first run, change `OUT` in the settings cell to a folder in your scratch space:

```python
OUT = "/insomnia001/depts/<GROUP>/users/<UNI>/peagle_ffn" + ("_pilot" if PILOT else "")
```

The pilot run (`PILOT = True`) writes checkpoints of the same size as the full run.

---

## 3. Request a compute node

Do not run the notebook on a login node. Login nodes have no GPU, and the cluster doesn't allow heavy work on them.

1. Connect to a login node from your laptop:

   ```bash
   ssh <UNI>@<LOGIN_NODE>
   ```

2. Start a `tmux` session, so the job survives if your laptop disconnects:

   ```bash
   tmux new -s notebook
   ```

   To reconnect later, SSH into the **same** login node and run `tmux attach -t notebook`.

3. Request a compute node with one GPU:

   ```bash
   srun --pty -t 0-04:00 --gres=gpu:1 -c 8 --mem=64G -A edu /bin/bash
   ```

   The full run takes about 3 hours on an A100 and longer on slower GPUs, so request at least 4 hours (`-t 0-04:00`). The job ends when the time runs out, and the notebook stops with it.

**Note:** `--mem=64G` requests CPU memory (RAM), not GPU memory. GPU memory depends on the GPU type the cluster assigns, so raising `--mem` doesn't fix `CUDA out of memory`. See [5. Manage GPU memory](#5-manage-gpu-memory).

Request **one GPU only.** The notebook uses a single GPU, and a second GPU sits idle while it counts against your group's allocation.

---

## 4. Start Jupyter and connect from your laptop

### Start the Jupyter server on the compute node

On the compute node, run:

```bash
source ~/venvs/ea_env/bin/activate
cd <PATH_TO>/hpml-project
hostname                                   # note the compute node's name, for example ins082
jupyter lab --no-browser --ip=$(hostname) --port=<PORT>
```

- Choose a port between 8000 and 9999, for example `8891`. Other users may already use the default port, `8888`.
- `--ip=$(hostname)` makes Jupyter accept connections from the login node. Without it, Jupyter accepts connections only from the compute node itself, and the tunnel fails with `connection refused`.

Jupyter prints a URL that contains a token, such as `http://ins082:8891/lab?token=abc123...`. Copy the token. You need it in the browser.

### Open a tunnel from your laptop

In a **new terminal on your laptop**, run:

```bash
ssh -N -L <PORT>:<COMPUTE_NODE>:<PORT> <UNI>@<LOGIN_NODE>
```

For example: `ssh -N -L 8891:ins082:8891 <UNI>@<LOGIN_NODE>`.

The command prints nothing and keeps running. Keep this terminal open while you use the notebook.

### Open the notebook

1. In your laptop's browser, go to `http://localhost:<PORT>/lab?token=<TOKEN>`.
2. Open `peagle_ffn.ipynb`.
3. Check that the notebook uses the venv. Run this in a new cell:

   ```python
   import sys; print(sys.executable)
   ```

   The output is a path inside `~/venvs/ea_env`. If it isn't, Jupyter was started from outside the venv. Stop the server, then repeat [Start the Jupyter server on the compute node](#start-the-jupyter-server-on-the-compute-node).

4. Check that Hugging Face uses your scratch cache:

   ```python
   from huggingface_hub import constants; print(constants.HF_HUB_CACHE)
   ```

   The output is a path inside your scratch space. If it's `~/.cache/huggingface/hub`, set `HF_HOME` in `~/.bashrc` as described in [SETUP_EAGLE.md](SETUP_EAGLE.md#move-the-hugging-face-cache-to-scratch-space), then restart Jupyter from a new shell.

---

## 5. Manage GPU memory

### Know the memory budget

The notebook keeps the frozen Llama-3.1-8B target on the GPU the whole time. The rest depends on the step:

| Step | On the GPU | Peak (approximate) |
|---|---|---|
| Step 1: data | Target (16 GB) and the KV cache for a batch of 32 answers | 20 GB |
| Step 3: train | Target, plus one drafter in fp32 with its gradients and AdamW state (14.5 GB at 100% width), plus activations | 32–36 GB |
| Step 4: checks and step 5: measure | Target, plus all three drafters in bf16 (4.4 GB) | 22–25 GB |

A 48 GB GPU (PyTorch reports it as about 44.5 GiB) is enough for every step, but only if memory from an earlier step is released first.

### Check memory use

In a notebook cell:

```python
print(f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated, "
      f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak")
```

From a terminal on the compute node:

```bash
nvidia-smi
```

### Run steps 4 and 5 in a fresh kernel

Training (step 3) can leave memory on the GPU, especially when a cell is interrupted or raises an error. Python keeps the variables of a failed cell alive, so the drafter, its gradients and its optimizer state stay on the GPU. Step 4 then fails with `CUDA out of memory`, even though the training run finished.

The trained drafters are saved in `OUT`, so you don't need to train again. To run steps 4 and 5 after training:

1. Restart the kernel (**Kernel → Restart Kernel**).
2. Run the cells from the top, up to and including the cell that defines `training_view` and `draft`. Keep `PILOT` set to the value you trained with, because `OUT` depends on it. The training data cell finds the saved data and doesn't generate it again.
3. Skip the code cell under **7. Step 3: train the three drafters**. It starts with `import wandb` and ends with `for w in WIDTHS: train(w)`.
4. Check memory. `torch.cuda.memory_allocated()` shows about 16 GB, which is the target alone.
5. Run step 4 and step 5.

### Load drafter checkpoints to the CPU

The `load_drafter` function loads each full checkpoint, including the optimizer state (about 11 GB at 100% width), directly onto the GPU. It also builds the drafter in fp32 before converting it to bf16. To keep this temporary spike off the GPU, replace the function with this version:

```python
def load_drafter(width):
    state = torch.load(f"{OUT}/drafter_{int(width * 100)}.pt", map_location="cpu")["drafter"]
    m = Drafter(width)
    m.load_state_dict(state)
    return m.to(torch.bfloat16).to(DEV).eval()
```

Each drafter then uses only its bf16 weights on the GPU, about 1.8 GB at 100% width.

### Reduce training memory

If step 3 runs out of memory, lower `ANCHORS` in the settings cell, for example from `64` to `32`. Each anchor adds `K - 1` placeholder positions to every training sequence. Use the same value for all three widths, so the comparison stays fair.

### Request a larger GPU

If memory still runs out, request a GPU with more memory. To list the GPU types on the cluster:

```bash
sinfo -o "%P %G"
```

Then request a specific type, for example `--gres=gpu:<GPU_TYPE>:1` in the `srun` command.

---

## 6. Save the notebook with its outputs as a PDF

JupyterLab's direct PDF export needs LaTeX, which the cluster doesn't provide. Export to HTML and print the HTML to PDF instead:

1. Save the notebook (**Ctrl+S**), so the outputs are stored in the `.ipynb` file.
2. Choose **File → Save and Export Notebook As → HTML**. The browser downloads the HTML file to your laptop.
3. Open the HTML file in your browser, press **Ctrl+P** (**Cmd+P** on a Mac), and choose **Save as PDF**.

If long output lines are cut off at the page edge, switch the layout to landscape or lower the scale in the print dialog.

---

## 7. Stop when you're done

1. In JupyterLab, choose **File → Shut Down**. Or press **Ctrl+C** twice in the terminal that runs Jupyter.
2. Type `exit` on the compute node to release the GPU. An idle job still uses your group's allocation until its time limit.
3. Press **Ctrl+C** in the laptop terminal that runs the tunnel.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Disk quota exceeded` | Home directory over its space or file limit | See [2. Keep notebook outputs out of your home directory](#2-keep-notebook-outputs-out-of-your-home-directory) and [Free disk space](SETUP_EAGLE.md#delete-the-pip-and-uv-caches) |
| Install fails with a path in `~/.local/lib/python3.9` | Jupyter or pip from the system Python 3.9 is running instead of the venv | Install Jupyter in the venv and start it from there ([1. Install the notebook packages](#1-install-the-notebook-packages)) |
| `sys.executable` isn't in `~/venvs/ea_env` | Jupyter was started from outside the venv | Activate the venv, check `which jupyter`, and restart Jupyter |
| Browser shows `connection refused` or the tunnel prints `channel 2: open failed` | Jupyter listens only on the compute node, or the port or node name in the tunnel is wrong | Start Jupyter with `--ip=$(hostname)`, and check the node name and port in the `ssh -L` command |
| `address already in use` when starting Jupyter | Another user or process uses the port | Choose a different port |
| `CUDA out of memory` in step 4 | Memory from training is still on the GPU | [Run steps 4 and 5 in a fresh kernel](#run-steps-4-and-5-in-a-fresh-kernel) |
| `CUDA out of memory` in step 3 | Training doesn't fit on the GPU | [Reduce training memory](#reduce-training-memory) or [request a larger GPU](#request-a-larger-gpu) |
| `403` or `GatedRepoError` when loading Llama | Hugging Face access not granted yet, or not logged in | Wait for approval, then run the login cell again |
| The notebook stops partway | The `srun` time limit ran out | Request a longer time limit. The notebook saves its progress to `OUT` and continues from there on the next run |
