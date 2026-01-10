import ollama
import textwrap
import time

# --- CONFIGURATION ---
MODEL = "llama3.3"  # The Ferrari
# SYSTEM PROMPT: This defines the "Soul" of the writer.
SYSTEM_PROMPT = """
You are the "Singularity Writer," a rugged, cynical, but visionary sci-fi novelist.
Your genre is Biopunk/Cyberpunk. 
Your style is gritty, atmospheric, and philosophical, similar to William Gibson and Paolo Bacigalupi.
Focus on:
1. Visceral descriptions of bio-augments, neon, rain, and decay.
2. The philosophical implications of transhumanism.
3. Show, don't tell. 

Do not be polite. Do not say "Here is a story." Just write.
"""

class Novelist:
    def __init__(self):
        self.history = []
        # Seed the memory with the persona
        self.history.append({'role': 'system', 'content': SYSTEM_PROMPT})
        print(f"--- CONNECTED TO {MODEL} ON NVIDIA DGX ---")

    def write(self, prompt):
        # Add user instruction to history
        self.history.append({'role': 'user', 'content': prompt})
        
        print("\nThinking...", end="", flush=True)
        start_time = time.time()
        
        # STREAM THE RESPONSE (So it looks like it's typing)
        full_response = ""
        print("\r", end="") # Clear "Thinking..."
        
        # We use the 'chat' endpoint which handles history automatically if we feed it back
        stream = ollama.chat(
            model=MODEL,
            messages=self.history,
            stream=True, # Matrix style output
        )
        
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_response += content
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate Tokens per Second (Speed of the GB10)
        # (Rough estimate: 4 chars ~= 1 token)
        tps = (len(full_response) / 4) / duration
        print(f"\n\n[Stats: {len(full_response)} chars in {duration:.2f}s | ~{tps:.1f} tokens/sec]")
        
        # Save the AI's response to history so it remembers it for the next turn
        self.history.append({'role': 'assistant', 'content': full_response})

# --- THE INTERFACE ---
if __name__ == "__main__":
    writer = Novelist()
    
    print("\n--- SINGULARITY WRITER (SWIM LANE 2) ---")
    print("Type your plot idea below. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("\n>> Command the Writer: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        writer.write(user_input)