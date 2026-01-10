import torch
import torch.nn as nn
import torch.optim as optim
import random

# --- 1. THE DATA (Tiny Biopunk Snippet) ---
import urllib.request

# Download "The Time Machine" by H.G. Wells (Public Domain)
url = "https://www.gutenberg.org/files/35/35-0.txt"
print(f"Downloading {url}...")
text = urllib.request.urlopen(url).read().decode('utf-8')

# Clean it up a bit (remove the Project Gutenberg header/footer junk)
# This is a rough chop, but it works for our purposes.
start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
end_idx = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
if start_idx != -1 and end_idx != -1:
    text = text[start_idx+100:end_idx] # Skip the legal header

print(f"New Dataset Size: {len(text)} characters")

# Create the "Vocabulary" (List of unique characters)
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

print(f"Dataset Size: {len(text)} characters")
print(f"Vocabulary: {vocab_size} unique characters")

# --- 2. THE MODEL (Fixed) ---
class NanoWriter(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Position: Changed shape to [100, 1, embed_dim] 
        # This matches the PyTorch default (Seq_Len, Batch, Dim)
        self.pos_encoder = nn.Parameter(torch.zeros(100, 1, embed_dim))
        
        # 3. The Brain
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. The Output
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x shape: [Sequence Length, Batch Size]
        seq_len = x.size(0)
        
        # Slicing fixed: We take the first 'seq_len' rows
        # Shape becomes [Seq_Len, 1, Dim] which adds perfectly to x
        pos = self.pos_encoder[:seq_len, :, :]
        
        embedded = self.embedding(x) + pos
        
        output = self.transformer(embedded)
        return self.fc_out(output)

# --- 3. TRAINING SETUP (The "Mid-Size" Upgrade) ---
# We are bumping these numbers up significantly
embed_dim = 128     # Was 32 -> Now 128 (More capacity for nuance)
hidden_dim = 256    # Was 64 -> Now 256 (Deeper thinking)
num_layers = 4      # Was 2  -> Now 4   (More abstract reasoning)
seq_length = 64     # Was 20 -> Now 64  (Longer memory of the sentence)
batch_size = 128    # NEW: Read 128 sentences at once (DGX power!)

# Re-initialize the model with the bigger brain
model = NanoWriter(vocab_size, embed_dim, hidden_dim, num_layers)

# Move the model to the GPU (CUDA)
# This is crucial for the DGX!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Training on: {device} (The Beast is Awake)")

optimizer = optim.Adam(model.parameters(), lr=0.002) # Slightly lower LR for stability
criterion = nn.CrossEntropyLoss()

def generate(start_str="The ", predict_len=100, temperature=0.6):
    model.eval()
    
    # 1. Prepare Input
    input_indices = [char_to_int[c] for c in start_str]
    
    # CRITICAL FIX: .to(device)
    # We move the input to the GPU immediately
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(1).to(device)
    
    generated = start_str
    
    # Check model's memory limit (from the pos_encoder)
    max_len = model.pos_encoder.size(0) 
    
    with torch.no_grad():
        for _ in range(predict_len):
            # Sliding Window Logic
            if input_tensor.size(0) > max_len:
                input_tensor = input_tensor[-max_len:]
                
            output = model(input_tensor)
            
            # Get logits (The predictions)
            logits = output[-1, 0, :]
            
            # Apply Temperature
            probs = torch.softmax(logits / temperature, dim=0)
            
            # Sample a character
            predicted_id = torch.multinomial(probs, 1).item()
            predicted_char = int_to_char[predicted_id]
            generated += predicted_char
            
            # 2. Append New Data (Also on GPU)
            next_input = torch.tensor([[predicted_id]], device=device)
            input_tensor = torch.cat((input_tensor, next_input), dim=0)
            
    return generated

# --- HELPER: GET BATCHES ---
def get_batch(data, batch_size, seq_length):
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Pick a random spot in the book
        start_idx = random.randint(0, len(data) - seq_length - 1)
        chunk = data[start_idx:start_idx + seq_length + 1]
        
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])
        
    # Stack them into a single tensor
    # Shape: [Seq_Len, Batch_Size] (Transformer expects Sequence first)
    input_batch = torch.stack(inputs).t().to(device)
    target_batch = torch.stack(targets).t().reshape(-1).to(device)
    
    return input_batch, target_batch

# --- 4. THE LOOP (Scaled Up) ---
print("\n--- TRAINING PHASE 2: BATCH PROCESSING ---")
# We need more epochs because the brain is bigger
epochs = 5000 

data_tensor = torch.tensor([char_to_int[c] for c in text], dtype=torch.long)

for epoch in range(epochs + 1):
    model.train()
    
    # Get a massive batch of data
    input_batch, target_batch = get_batch(data_tensor, batch_size, seq_length)
    
    optimizer.zero_grad()
    output = model(input_batch)
    
    # Reshape output to [Batch * Seq_Len, Vocab]
    loss = criterion(output.view(-1, vocab_size), target_batch)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        # Generate on CPU to avoid complexity
        sample = generate(start_str="The ", predict_len=100, temperature=0.6)
        print(f"Generated: {sample}\n")