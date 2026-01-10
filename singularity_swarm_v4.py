import torch
import torch.nn as nn
import torch.optim as optim
import os

# THE NEW IMPORTS (The Swarm Tools)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

def setup(rank, world_size):
    """
    Initializes the 'Phone Line' between GPUs.
    rank: Which GPU am I? (0 to 7)
    world_size: How many GPUs total? (8)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 'nccl' is the NVIDIA Collective Communications Library
    # It is the super-fast language Blackwell GPUs speak.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# --- THE SWARM TRAINER ---
def train(rank, world_size):
    print(f"GPU {rank} reporting for duty.")
    setup(rank, world_size)
    
    # 1. PREPARE DATA (The Slicing)
    # We use a custom Sampler that ensures GPU 0 gets unique data
    dataset = TextDataset(...) # Your text loading logic here
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # 2. PREPARE MODEL (The Cloning)
    # Move model to THIS specific GPU (rank)
    model = NanoWriter(...).to(rank)
    
    # Wrap it in DDP. This handles the "Telepathy" automatically.
    # When you call loss.backward(), DDP syncs with all other GPUs.
    model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. THE LOOP
    for epoch in range(1000):
        # Vital: Shuffle the data differently for each GPU every epoch
        sampler.set_epoch(epoch)
        
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward() # <--- DDP SYNC HAPPENS HERE (Magic!)
            optimizer.step()
        
        # Only the Master GPU (Rank 0) should print logs or save files
        if rank == 0 and epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()}")
            torch.save(model.module.state_dict(), "swarm_brain.pth")

    cleanup()

# This script is not run with 'python'. It is run with 'torchrun'.