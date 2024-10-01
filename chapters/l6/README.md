# Scalable training

# How to start

## Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version

## Set up
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

## Test PyTorch DDP on Mock data
```shell
# Test pure PyTorch DDP
poetry run torchrun --nproc_per_node=2 --nnodes=1 --master_addr="localhost" --master_port=12345 src/test_distributed.py --epochs 500 --device cpu
# Test PyTorch Lightning
poetry run python src/test_lightning.py --accelerator cpu --devices 2
```

## Rent GPUs on Vast.ai 
Filter template with PyTorch 2.4.0 -> Rent a 2X GPU instance in Asia -> Paste local machine SSH keys into the instance config -> Copy Proxy SSH connect -> Add to SSH VSCode > Connect to SSH and paste the proxy command to register the SSH config there -> Change the Host name -> Connect to the server by running SSH

Install Github CLI to pull private repo: https://github.com/cli/cli/blob/trunk/docs/install_linux.md
```shell
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y
sudo apt update
sudo apt install gh
gh auth login
# Choose Github.com
# Go to https://github.com/login/device and paste the code from the CLI
```

```shell
mkdir workstation
gh repo clone dvquy13/fsds-recsys
```

-> Open workspace fsds-recsys with VSCode

```shell
cd chapters/l6
# Install deps
conda create --prefix .venv python=3.11.9
pip install poetry==1.8.3
poetry env use .venv/bin/python
poetry install
```

-> Copy the necessary data over (item_sequence.jsonl, val_item_sequence.jsonl, idm.json) from local or rerun notebooks 000, 001, 010 to produce the data.

```shell
cd notebooks
poetry run python train_item2vec.py --accelerator cpu --devices 1
```

Open a new shell:
```shell
cd chapters/l6
poetry run tensorboard --logdir notebooks/data/<run_name>/logs/run --port 6009
```

-> Port forward to open the tensorboard on local