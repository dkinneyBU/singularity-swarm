import torch
import torch.nn as nn
import torch.optim as optim
import random
import urllib.request
import os

# --- 1. SETUP & DATA ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {device}")

# Download H.G. Wells
url = "https://www.gutenberg.org/files/35/35-0.txt"
if not os.path.exists("time_machine.txt"):
    print("Downloading book...")
    text = urllib.request.urlopen(url).read().decode('utf-8')
    # Clean header/footer
    start = text.find("*** START") + 30
    end = text.find("*** END")
    text = text[start:end]
    with open("time_machine.txt", "w") as f:
        f.write(text)
else:
    print("Loading book from disk...")
    text = open("time_machine.txt", "r").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
print(f"Vocab: {vocab_size} chars | Length: {len(text)}")

# --- 2. THE MODEL (V3: With Dropout) ---
class NanoWriter(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Position Encoder (Max 200 chars memory)
        self.pos_encoder = nn.Parameter(torch.zeros(200, 1, embed_dim))
        
        # DROPOUT: The Cure for Narcissism
        self.dropout = nn.Dropout(0.2) 
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(0)
        pos = self.pos_encoder[:seq_len, :, :]
        # Add position + apply dropout
        embedded = self.embedding(x) + pos
        embedded = self.dropout(embedded) 
        output = self.transformer(embedded)
        return self.fc_out(output)

# --- 3. GENERATE FUNCTION (GPU Optimized) ---
def generate(model, start_str="The ", predict_len=200, temperature=0.8):
    model.eval()
    input_indices = [char_to_int.get(c, 0) for c in start_str]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(1).to(device)
    generated = start_str
    
    with torch.no_grad():
        for _ in range(predict_len):
            # Sliding window (keep input smaller than pos_encoder limit)
            if input_tensor.size(0) > 190:
                input_tensor = input_tensor[-190:]
                
            output = model(input_tensor)
            logits = output[-1, 0, :]
            
            # Temperature Sampling
            probs = torch.softmax(logits / temperature, dim=0)
            pred_id = torch.multinomial(probs, 1).item()
            
            generated += int_to_char[pred_id]
            next_input = torch.tensor([[pred_id]], device=device)
            input_tensor = torch.cat((input_tensor, next_input), dim=0)
    return generated

# --- 4. TRAINING LOOP ---
embed_dim = 128
hidden_dim = 256
num_layers = 4
seq_length = 128
batch_size = 128

model = NanoWriter(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
data_tensor = torch.tensor([char_to_int[c] for c in text], dtype=torch.long).to(device)

def get_batch():
    starts = torch.randint(0, len(data_tensor) - seq_length - 1, (batch_size,))
    inputs = torch.stack([data_tensor[s : s + seq_length] for s in starts]).t()
    targets = torch.stack([data_tensor[s + 1 : s + seq_length + 1] for s in starts]).t().reshape(-1)
    return inputs, targets

print("\n--- STARTING V3 TRAINING ---")
for epoch in range(5001):
    model.train()
    input_batch, target_batch = get_batch()
    
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output.view(-1, vocab_size), target_batch)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        # Save Checkpoint
        torch.save(model.state_dict(), f"writer_v3_epoch_{epoch}.pth")
        # Generate Test
        print(f"Generated: {generate(model, temperature=0.8)}\n")