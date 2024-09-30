# poetry run torchrun --nproc_per_node=2 --nnodes=1 --master_addr="localhost" --master_port=12345 src/test_distributed.py --epochs 500 --device cpu
# poetry run torchrun --nproc_per_node=2 --nnodes=1 --master_addr="localhost" --master_port=12345 src/test_distributed.py --epochs 500 --device cuda

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# Custom Dataset (Example using random data)
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        # For simplicity, targets are also random
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len


# Simple model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


# Training function
def train(args):
    # Get rank and world size
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the process group based on device
    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend=backend)

    # Determine device type for logging
    device_type = "GPU" if args.device == "cuda" else "CPU"

    # Configure logger to display CPU or GPU instead of rank
    logger.remove()  # Remove default logger
    logger.add(
        sys.stderr,
        format=f"<green>{{time}}</green> | <level>{{level}}</level> | Device: {device_type} {rank} | <cyan>{{message}}</cyan>",
        level="INFO",
    )

    logger.info(
        f"Starting training on device: {device_type}. World size: {world_size}, local rank: {local_rank}"
    )

    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Set the device based on the passed argument
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Using GPU: cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU")

    # Create the model and move it to the correct device
    model = ToyModel().to(device)

    # Wrap the model with DDP, specifying the device if it's CUDA
    if args.device == "cuda":
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)  # No device_ids needed for CPU

    # Create dataset and distributed sampler
    dataset = RandomDataset(size=10, length=64)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Shuffle data for each epoch
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to the correct device
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log batch-level information
            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

        # Average loss across all batches
        avg_loss = epoch_loss / len(dataloader)

        # Log epoch-level information
        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}"
        )

    logger.info(f"Training completed on device: {device_type}.")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="PyTorch DDP Example with DataLoader and CLI Arguments"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size per process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use for training (cpu or cuda)",
    )
    args = parser.parse_args()

    train(args)
