import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
import random
import warnings

# Silence the "Beta Hardware" warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {device} (The Monolith)")

# --- 2. THE BIGGER DATASET (Shakespeare) ---
class ShakespeareDataset(Dataset):
    def __init__(self, seq_length):
        self.seq_length = seq_length
        file_path = "shakespeare.txt"
        
        if not os.path.exists(file_path):
            print("--- DOWNLOADING SHAKESPEARE ---")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            data = urllib.request.urlopen(url).read().decode('utf-8')
            with open(file_path, "w") as f:
                f.write(data)
        
        self.text = open(file_path, "r").read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}
        self.data = torch.tensor([self.char_to_int[c] for c in self.text], dtype=torch.long)
        print(f"Dataset Loaded. Length: {len(self.text)} chars.")

    def __len__(self):
        return len(self.text) // self.seq_length

    def __getitem__(self, idx):
        start = random.randint(0, len(self.data) - self.seq_length - 1)
        chunk = self.data[start : start + self.seq_length + 1]
        return chunk[:-1], chunk[1:]

# --- 3. THE MODEL (V4: Big Brain) ---
class NanoWriter(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, hidden_dim=768, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Position Encoder
        self.pos_encoder = nn.Parameter(torch.zeros(512, 1, embed_dim))
        self.dropout = nn.Dropout(0.2)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, dim_feedforward=hidden_dim, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(0)
        pos = self.pos_encoder[:seq_len, :, :]
        embedded = self.embedding(x) + pos
        embedded = self.dropout(embedded)
        output = self.transformer(embedded)
        return self.fc_out(output)

# --- 4. GENERATE FUNCTION ---
def generate(model, start_str="The ", predict_len=200, temperature=0.8):
    model.eval()
    input_indices = [dataset.char_to_int.get(c, 0) for c in start_str]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(1).to(device)
    generated = start_str
    
    with torch.no_grad():
        for _ in range(predict_len):
            if input_tensor.size(0) > 500: 
                input_tensor = input_tensor[-500:]
            output = model(input_tensor)
            logits = output[-1, 0, :]
            probs = torch.softmax(logits / temperature, dim=0)
            pred_id = torch.multinomial(probs, 1).item()
            generated += dataset.int_to_char[pred_id]
            next_input = torch.tensor([[pred_id]], device=device)
            input_tensor = torch.cat((input_tensor, next_input), dim=0)
    return generated

# --- 5. THE VIRTUAL SWARM LOOP ---
batch_size = 64
seq_length = 256
gradient_accumulation_steps = 8 # (64 * 8 = 512 effective batch)

dataset = ShakespeareDataset(seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = NanoWriter(dataset.vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003) 
criterion = nn.CrossEntropyLoss()

print("\n--- INITIALIZING VIRTUAL SWARM ---")
print(f"Physical Batch: {batch_size}")
print(f"Virtual  Batch: {batch_size * gradient_accumulation_steps} (The Swarm Effect)")

model.train()
optimizer.zero_grad()

# --- THE FIX IS HERE ---
for epoch in range(10):
    for step, (inputs, targets) in enumerate(dataloader):
        
        # 1. Transpose Inputs [Seq, Batch]
        inputs = inputs.t().to(device)
        
        # 2. Transpose AND Flatten Targets [Seq*Batch]
        # This aligns the targets with the flattened output
        targets = targets.t().reshape(-1).to(device)
        
        output = model(inputs)
        loss = criterion(output.view(-1, dataset.vocab_size), targets)
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch} | Step {step+1} | Loss: {loss.item() * gradient_accumulation_steps:.4f}")

    print(f"\n--- Epoch {epoch} Complete. Saving Brain... ---")
    torch.save(model.state_dict(), f"virtual_swarm_epoch_{epoch}.pth")
    print(f"Generated:\n{generate(model, temperature=0.8)}\n")