# Running EAGLE on Google Cloud or the Insomnia HPC Cluster

This guide covers setting up and running EAGLE-3 with Llama-3.1-8B-Instruct in either environment:

Placeholders are written as `<LIKE_THIS>`. Replace them with your own values.

---

## Google Cloud

### 1. Create a GCP virtual machine with GPU

1. Go to `cloud.google.com` and click on Console
2. Click 'Create a VM'
3. Get the following machine: 1x NVIDIA L4, g2-standard-8, at least 50GB disk. For location, I've had luck with us-east4-a. Name the machine however you want.

Installing all dependencies takes up about 16 GB. The 8B model in fp16 needs about another 17–18 GB of GPU memory. I recommend starting with 50GB, which is enough for one 8B model. If you want to store more models at a time, increase the disk size in **Disk -> Click on your disk -> Edit**, or through the gcloud cli (See bottom of this file). Note that you can increase disk size but not decrease it.

### 2. Networking: Cloud NAT and SSH access

Columbia blocks external IP so we'll need to set up Cloud NAT Gateway on Google Console.

1. Search for "Cloud NAT" in Google Console search bar.
2. Click "Create Cloud NAT Gateway"
    Name: hpml-nat-gateway (or any preferred name)
    Network: Default settings (or your specific VPC)
    Region: Must match your VM's region (e.g., us-central1).
    Cloud Router: Create a new router (keep defaults).
        Name: hpml-cloud-router (or any preferred name)
    Click "Create".

3. Configure Firewall for IAP
    IAP tunnels traffic through a specific Google internal IP range. You must allow this traffic in your firewall.
    Search for "Firewall" in the GCP Console.
    Click "Create firewall rule".
        Name: allow-ingress-from-iap (or any preferred name)
        Targets: All instances in the network
        Source IPv4 ranges: 35.235.240.0/20 (copy this exactly)
        Protocols and ports: Check TCP and enter 22.
    Click "Create".
---

### 3. Local setup: install and configure `gcloud`

Technically you can use "SSH-in-browser" (click on "SSH" next to your VM name in Computer Engine -> VM Machines), and skip the gcloud set up. However, I find "SSH-in-browser" highly unreliable since it can disconnect at anytime without any reason.

Install the Google Cloud CLI on your local machine by following https://cloud.google.com/sdk/docs/install. Then:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable compute.googleapis.com
```

Optionally, set a default region and zone so you can leave out `--zone` later. Pick a zone that offers the GPU you want (see step 1):

```bash
gcloud config set compute/region us-east4
gcloud config set compute/zone us-east4-a
```


### 4. SSH into the VM from your local terminal

```bash
gcloud compute ssh <VM_NAME> --zone=<ZONE> --tunnel-through-iap
```

The first time you connect, `gcloud` creates an SSH key for you (`~/.ssh/google_compute_engine`) and adds it to the project.

To check the VM can reach the internet:

```bash
curl -sI https://pypi.org | head -1     # expect: HTTP/2 200
```

If this hangs, recheck step 3a. The NAT must be in the **same region and network** as the VM.

**Stop the VM when you're not using it.** GPU VMs are billed for every second they run:

```bash
gcloud compute instances stop <VM_NAME> --zone=<ZONE>
gcloud compute instances start <VM_NAME> --zone=<ZONE>
```
Or stop the VM on Google Cloud Console.

### 5. Install the NVIDIA driver

Run all of the following steps while **connected to the VM**.

Install the driver:

```bash
sudo apt update && sudo apt install -y python3 curl
curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver
```

Check to see if the driver is installed. It should show your GPU:

```bash
nvidia-smi
```

### 6. Install system packages

```bash
sudo apt update && sudo apt install -y git build-essential curl
```

---

## Insomnia HPC Cluster

For an Insomnia setup, first follow Columbia's getting-started guide:

https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145140/Insomnia+Getting+Started#Prepare-VPN-(if-necessary)

This guide covers VPN access, connecting to Insomnia, and the cluster's basic workflow. After logging in to a login node, request an interactive compute node before running EAGLE:

```bash
srun --pty -t 0-02:00 --gres=gpu:1 -c 8 --mem=64G -A edu /bin/bash
```

The options mean:

- `--pty`: allocate an interactive terminal.
- `-t 0-02:00`: request a two-hour time limit.
- `--gres=gpu:1`: request one GPU.
- `-c 8`: request eight CPU cores.
- `--mem=64G`: request 64 GB of memory.
- `-A edu`: charge the job to the `edu` account.
- `/bin/bash`: start a Bash shell on the compute node.

Request just **one GPU** for EAGLE so the model, draft model, and all generated tokens remain on the same GPU. Do not run GPU workloads directly on the login node.

Insomnia users do not have `sudo` privileges. Do not run the `sudo` commands in the Google Cloud driver and system-package steps above. Use the cluster's preinstalled software and install Python packages in your virtual environment instead; contact the cluster administrators if a required system package is unavailable.

---

## Shared EAGLE setup

Complete the following steps in either environment. On Insomnia, use the allocated compute node; on Google Cloud, use the VM.

### 1. Connect the machine to GitHub (optional; needed to push)

```bash
ssh-keygen -t ed25519 -C "<YOUR_EMAIL>"     # press Enter to accept the defaults
cat ~/.ssh/id_ed25519.pub
```

Copy the line that `cat` printed and add it to GitHub at **Settings → SSH and GPG keys → New SSH key**.

```bash
ssh -T git@github.com                       # expect: "Hi <user>! You've successfully authenticated..."
git config --global user.name  "<YOUR_NAME>"
git config --global user.email "<YOUR_EMAIL>"
```

---

### 2. Install Python 3.10 and create a virtual environment

The packages pinned in `requirements.txt` are old, and some of them (`protobuf==3.19.0`, `sentencepiece==0.1.99`) need **Python 3.10**. `uv` installs that Python version and creates the venv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

uv venv ~/venvs/ea_env --python 3.10
source ~/venvs/ea_env/bin/activate
```

You can activate the venv from any directory, for example with `source ~/venvs/ea_env/bin/activate`. It stays active for that shell session.

---

### 3. Clone EAGLE and install dependencies

Make a fork of the EAGLE repo. Then:

```bash
git clone git@github.com:<YOUR_GITHUB_USER>/eagle.git     # or https://github.com/SafeAILab/EAGLE.git
cd eagle

uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

> **Always use `--no-deps` with `pip install -e .`.** `setup.py` pins `torch==2.0.1` and `transformers==4.46.2`, which conflict with `requirements.txt`. Without `--no-deps`, these old versions get installed over the right ones, and you'll see the import errors listed under [Troubleshooting](#troubleshooting).

`pip install -e .` installs the repo in *editable* mode. `import eagle` then works from any directory, and your code changes take effect without reinstalling.

Check the install:

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
# expect: 2.6.0  4.5x.x  True
```

If `transformers` shows `5.x`, pin it: `uv pip install "transformers==4.53.1"`.

---

### 4. Hugging Face access

1. Create a token at **huggingface.co → Settings → Access Tokens**. **Read** access is enough. For a fine-grained token, tick only *"Read access to contents of all public gated repos you can access"*.
2. Request access to the gated base model at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. Approval can take minutes to hours. I got approved in half an hour.
3. Log in to Hugging Face on the machine:

```bash
huggingface-cli login       # paste the token; answer "n" to the git-credential question
```

To check your Llama access has been approved:

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct config.json
```

A `403` or `GatedRepoError` means the request is still pending.

**While you wait**, you can test the whole setup with `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` and `yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B`. DeepSeek-R1-Distill-Llama-8B isn't gated, uses the same Llama architecture and has an official EAGLE-3 draft model.

---

### 5. Run EAGLE-3 with Llama-3.1-8B-Instruct

Both models download automatically on first use and are stored in `~/.cache/huggingface`:

- Base model: `meta-llama/Llama-3.1-8B-Instruct` (about 16 GB)
- EAGLE-3 draft model: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`

On Insomnia, the models can exceed your home directory quota. To store them elsewhere, see [Move the Hugging Face cache to scratch space](#move-the-hugging-face-cache-to-scratch-space).

### Option A: Web UI

```bash
python -m eagle.application.webui \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --model-type llama-3-instruct \
  --total-token -1 \
  --no-eagle3
```

- `--model-type llama-3-instruct` is required. The default (`vicuna`) uses the wrong chat template.
- `--total-token -1` lets EAGLE choose how many tokens to draft.
- `--no-eagle3` lets you use EAGLE-3. I know, the logic is reversed. I think this is a bug in the source code.
- The UI prints a public `*.gradio.live` URL to open in your browser. Anyone who has the link can use your model while it's running.

### Option B: Python

```python
import torch
from eagle.model.ea_model import EaModel

model = EaModel.from_pretrained(
    base_model_path="meta-llama/Llama-3.1-8B-Instruct",
    ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1,
)
model.eval()

messages = [{"role": "user", "content": "Explain speculative decoding in two sentences."}]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = torch.as_tensor(model.tokenizer([prompt], add_special_tokens=False).input_ids).cuda()

output_ids = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=256, is_llama3=True)
print(model.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True))
```

- `is_llama3=True` makes generation stop at Llama 3's end-of-turn token (`<|eot_id|>`).
- For a baseline without EAGLE, call `model.naivegenerate(...)` with the same arguments. To compare speeds fairly, run one warm-up generation first and call `torch.cuda.synchronize()` before reading the timer.

### Option C: MT-Bench evaluation

```bash
# EAGLE-3 (the --use_eagle3 flag is required; this script defaults to EAGLE-1)
python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --use_eagle3 --bench-name mt_bench

# Baseline (plain decoding)
python -m eagle.evaluation.gen_baseline_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --bench-name mt_bench
```

The answers are written to `mt_bench/<model-id>.jsonl`. To compare speeds, edit the two filenames at the top of `eagle/evaluation/speed.py` to point at these files, then run it.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| VM can't reach the internet; `curl` hangs | No Cloud NAT in the VM's **region** or network, or the NAT only covers some subnets | Create a NAT in that region (step 3a), or set the existing one to cover all subnets |
| `No space left on device (os error 28)` during `pip install` | Boot disk too small (10 GB default) | Resize the disk (below), then `uv cache clean` and reinstall |
| `Request had insufficient authentication scopes` | Running `gcloud` *inside* the VM, which uses the VM's read-only service account | Run the command from your local machine, Cloud Shell or the console |
| `cannot import name 'use_kernel_forward_from_hub'` | `transformers` was downgraded to 4.46.2 by `pip install -e .` without `--no-deps` | Recreate the venv (below) |
| `libtorch_cuda.so: undefined symbol: ncclCommRegister` | Mixed versions of torch and NVIDIA packages left over from that downgrade | Recreate the venv (below) |
| `403` / `GatedRepoError` when loading Llama | Hugging Face access not granted yet, or not logged in | Wait for approval; run `huggingface-cli login` |
| CUDA out of memory | GPU too small (for example a T4 with 16 GB) | Use an L4 or A100, or a smaller model |
| `Disk quota exceeded` on Insomnia | Your home directory has a small quota, and the pip, uv and Hugging Face caches fill it | [Delete the pip and uv caches](#delete-the-pip-and-uv-caches) or [move the Hugging Face cache to scratch space](#move-the-hugging-face-cache-to-scratch-space) |

### Resize the boot disk

Run this from your **local machine or Cloud Shell**, not the VM:

```bash
gcloud compute disks resize <VM_NAME> --size=200GB --zone=<ZONE>
```

Then, on the VM:

```bash
lsblk                       # the disk should show the new size
df -h /                     # if this still shows the old size, grow the partition and filesystem:
sudo growpart /dev/sda 1    # NVMe disks: sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/sda1    # NVMe disks: sudo resize2fs /dev/nvme0n1p1
```

Rebooting (`sudo reboot`) also grows the filesystem on most GCP images. You don't need to reopen your SSH session.

### Recreate the virtual environment

```bash
deactivate
rm -rf ~/venvs/ea_env
uv venv ~/venvs/ea_env --python 3.10
source ~/venvs/ea_env/bin/activate
cd ~/eagle
uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

### Free disk space by deleting downloaded models

```bash
du -sh ~/.cache/huggingface/hub/*                     # see what's using space
rm -rf ~/.cache/huggingface/hub/models--<ORG>--<NAME>
# or choose interactively:
huggingface-cli delete-cache
```

### Delete the pip and uv caches

pip and uv keep copies of every package they download in `~/.cache/pip` and `~/.cache/uv` so that later installs skip the download. Installed packages run from the virtual environment, not from these caches, so deleting the caches doesn't affect your environment. Future installs download the packages again.

Check how much space the caches use:

```bash
du -sh ~/.cache/* ~/venvs/* 2>/dev/null | sort -h
```

Delete the pip cache:

```bash
pip cache purge             # or: rm -rf ~/.cache/pip
```

Before you delete the uv cache, check that your virtual environment doesn't link to it:

```bash
python -c "import torch, os; print(os.path.realpath(torch.__file__))"
```

If the path is inside `~/venvs/ea_env`, delete the uv cache:

```bash
uv cache clean
```

If the path is inside `~/.cache/uv`, keep the cache. Deleting it breaks the virtual environment.

**Note:** uv hard-links packages from its cache into the virtual environment, so `du` can report several GB for `~/.cache/uv` and almost nothing for the virtual environment. Deleting the uv cache then frees less space than `du` reports, because the virtual environment still uses those files.

### Move the Hugging Face cache to scratch space

The Hugging Face cache holds the downloaded models, about 16 GB for Llama-3.1-8B-Instruct and its EAGLE-3 draft model. Keep these models, because EAGLE downloads them again if you delete them. Instead, move the cache out of your home directory to scratch or group storage, which has a much larger quota.

The `HF_HOME` environment variable tells Hugging Face where its cache is. After you set it, your code and commands stay the same, and EAGLE loads the models from the new location without downloading them again.

1. Find your scratch or group storage. On Insomnia, it's usually under `/insomnia001/depts/`:

   ```bash
   ls /insomnia001/depts/
   ```

   Check the Insomnia documentation or ask your course staff if you can't find your group's folder.

2. Move the cache. The move copies about 16 GB between file systems, so it takes a few minutes:

   ```bash
   SCRATCH=/insomnia001/depts/<GROUP>/users/$USER    # replace with your actual path
   mkdir -p $SCRATCH/hf_cache
   mv ~/.cache/huggingface/* $SCRATCH/hf_cache/
   ```

3. Set `HF_HOME` for this session and future sessions:

   ```bash
   echo "export HF_HOME=$SCRATCH/hf_cache" >> ~/.bashrc
   source ~/.bashrc
   ```

   If you submit jobs with `sbatch`, also add the `export HF_HOME=...` line to your job script.

4. Verify that Hugging Face uses the new location:

   ```bash
   python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"
   ```

   The output is a path inside `$SCRATCH/hf_cache/hub`.

**Note:** Many clusters delete scratch files that nobody has used for a while, and they don't back up scratch storage. This is fine for a model cache, because the worst case is a new download. Keep your code and results in your home directory or group storage.
