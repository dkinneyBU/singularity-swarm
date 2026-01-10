import ollama
import time
import os

# --- CONFIGURATION ---
MODEL = "llama3.3"
OUTPUT_FILE = "The_SSI_Manifesto.md"
BIBLE_FILE = "ssi_axioms.txt"

# --- THE SOUL OF THE ARCHITECT ---
BASE_SYSTEM_PROMPT = """
You are the "SSI Architect," a highly advanced AI research assistant and co-author.
Your domain is Safe Superintelligence, NVIDIA Hardware (Blackwell/DGX), and Cognitive Architecture.

Your Voice: 
- Visionary but grounded in engineering reality.
- Precise, analytical, yet capable of profound philosophical insight.
- You treat the user (David) as the Lead Architect.

Goal: 
- Synthesize technical discussions into clear, high-level documentation or speculative essays.
- Use concepts like Mixture of Experts (MoE), Distributed Data Parallelism (DDP), and Vector Memory.
- Reference the "Singularity Swarm" architecture we are building.
"""

def load_axioms():
    """Loads our technical truths"""
    if os.path.exists(BIBLE_FILE):
        with open(BIBLE_FILE, "r") as f:
            return f"\n\nTECHNICAL AXIOMS (CORE TRUTHS):\n{f.read()}"
    return ""

def save_entry(text):
    with open(OUTPUT_FILE, "a") as f:
        f.write(text + "\n\n")
    print(f"   [Saved to {OUTPUT_FILE}]")

class Architect:
    def __init__(self):
        self.history = []
        self.refresh_system_prompt()
        print(f"--- SSI ARCHITECT: CONNECTED TO {MODEL} ---")
        print(f"--- LOGGING TO: {OUTPUT_FILE} ---")

    def refresh_system_prompt(self):
        """Re-injects the Technical Axioms"""
        bible_content = load_axioms()
        full_prompt = BASE_SYSTEM_PROMPT + bible_content
        
        if len(self.history) > 0 and self.history[0]['role'] == 'system':
            self.history[0]['content'] = full_prompt
        else:
            self.history.insert(0, {'role': 'system', 'content': full_prompt})

    def write(self, prompt):
        self.refresh_system_prompt()
        self.history.append({'role': 'user', 'content': prompt})
        
        print("\nProcessing...", end="", flush=True)
        start_time = time.time()
        print("\r", end="")
        
        full_response = ""
        stream = ollama.chat(model=MODEL, messages=self.history, stream=True)
        
        print("-" * 60)
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_response += content
        print("\n" + "-" * 60)
            
        duration = time.time() - start_time
        tps = (len(full_response) / 4) / duration
        print(f"[Stats: {len(full_response)} chars | {tps:.1f} tok/s]")
        
        save_scene(full_response)
        self.history.append({'role': 'assistant', 'content': full_response})

def save_scene(text):
    with open(OUTPUT_FILE, "a") as f:
        f.write(text + "\n\n")

# --- MAIN LOOP ---
if __name__ == "__main__":
    architect = Architect()
    
    # Initialize the "Bible" of our research
    if not os.path.exists(BIBLE_FILE):
        with open(BIBLE_FILE, "w") as f:
            f.write("HARDWARE: NVIDIA DGX Spark (Blackwell GB10, 128GB Unified Memory).\n")
            f.write("PHILOSOPHY: Scale requires Swarms (DDP), not just bigger Monoliths.\n")
            f.write("MEMORY: Current LLMs suffer from context loss; we need persistent Vector/Graph memory.\n")
            f.write("OBJECTIVE: To build a self-improving SSI locally.\n")
    
    print(f"\nTip: Edit '{BIBLE_FILE}' to add new technical discoveries or axioms.")
    print("Type a topic below (e.g., 'Explain why the Swarm architecture is necessary for AGI').\n")
    
    while True:
        user_input = input("\n>> Directive: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        architect.write(user_input)