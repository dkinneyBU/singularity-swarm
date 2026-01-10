import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import urllib.request
import os
import random

# --- 1. CONFIGURATION ---
# We use standard DDP setup
def setup():
    # Torchrun sets these variables automatically
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

# --- 2. THE BIGGER DATASET ---
class ShakespeareDataset(Dataset):
    def __init__(self, seq_length):
        self.seq_length = seq_length
        
        # Only Rank 0 downloads the file to avoid race conditions
        file_path = "shakespeare.txt"
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if not os.path.exists(file_path):
                print("--- DOWNLOADING SHAKESPEARE ---")
                url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                data = urllib.request.urlopen(url).read().decode('utf-8')
                with open(file_path, "w") as f:
                    f.write(data)
        
        # Wait for Rank 0 to finish downloading
        dist.barrier()
        
        # Everyone loads the file
        self.text = open(file_path, "r").read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}
        self.data = torch.tensor([self.char_to_int[c] for c in self.text], dtype=torch.long)
        
        print(f"Dataset Loaded. Length: {len(self.text)} chars.")

    def __len__(self):
        # We define epoch size arbitrarily large so GPUs chew for a while
        return len(self.text) // self.seq_length

    def __getitem__(self, idx):
        # Random sampling
        start = random.randint(0, len(self.data) - self.seq_length - 1)
        chunk = self.data[start : start + self.seq_length + 1]
        return chunk[:-1], chunk[1:]

# --- 3. THE MODEL (Standard NanoWriter) ---
class NanoWriter(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(512, 1, embed_dim))
        self.dropout = nn.Dropout(0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(0)
        # Slicing pos_encoder to match sequence length
        pos = self.pos_encoder[:seq_len, :, :]
        embedded = self.embedding(x) + pos
        embedded = self.dropout(embedded)
        output = self.transformer(embedded)
        return self.fc_out(output)

# --- 4. MAIN TRAINING LOOP ---
def main():
    setup()
    
    # Get local rank (GPU ID: 0, 1, 2...)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    # HYPERPARAMETERS
    batch_size = 64      # Per GPU (Total batch = 64 * 8 = 512)
    seq_length = 128
    
    # Prepare Data
    dataset = ShakespeareDataset(seq_length)
    sampler = DistributedSampler(dataset) # Handles the splitting across GPUs
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # Prepare Model
    model = NanoWriter(dataset.vocab_size).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for stability
    criterion = nn.CrossEntropyLoss()

    # Only Rank 0 prints logs
    if local_rank == 0:
        print("\n--- SINGULARITY SWARM ACTIVATED ---")
        print(f"GPUs Detected: {torch.cuda.device_count()}")
        print("Training on the Complete Works of Shakespeare...\n")

    for epoch in range(10): # 10 Epochs is plenty for Shakespeare on a DGX
        sampler.set_epoch(epoch) # Shuffle data for DDP
        
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.t().to(device), targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, dataset.vocab_size), targets)
            loss.backward()
            optimizer.step()
            
            if local_rank == 0 and step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
        
        # SAVE CHECKPOINT (Rank 0 only)
        if local_rank == 0:
            torch.save(model.module.state_dict(), f"swarm_shakespeare_epoch_{epoch}.pth")
            print(f"--- Epoch {epoch} Complete. Brain Saved. ---")

    cleanup()

if __name__ == "__main__":
    main()