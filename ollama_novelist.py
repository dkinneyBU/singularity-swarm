import ollama
import time
import os

# --- CONFIGURATION ---
MODEL = "llama3.3"
NOVEL_FILE = "my_biopunk_novel.md"
BIBLE_FILE = "world_bible.txt"

# --- THE SOUL OF THE MACHINE ---
BASE_SYSTEM_PROMPT = """
You are a master Biopunk novelist. 
Style: Gritty, atmospheric, sensory (smell, sound, texture), Gibson-esque.
Goal: Write compelling scenes based on user commands.
Format: Output purely the story text. No "Here is the scene" intros.
"""

def load_bible():
    """Loads persistent world facts (Character names, rules, setting)"""
    if os.path.exists(BIBLE_FILE):
        with open(BIBLE_FILE, "r") as f:
            return f"\n\nWORLD BIBLE (ALWAYS REMEMBER):\n{f.read()}"
    return ""

def save_scene(text):
    """Saves the fresh ink to the manuscript"""
    with open(NOVEL_FILE, "a") as f:
        f.write(text + "\n\n")
    print(f"   [Saved to {NOVEL_FILE}]")

class Novelist:
    def __init__(self):
        self.history = []
        self.refresh_system_prompt()
        print(f"--- BIOPUNK STUDIO: CONNECTED TO {MODEL} ---")
        print(f"--- WRITING TO: {NOVEL_FILE} ---")

    def refresh_system_prompt(self):
        """Re-injects the World Bible into the system prompt"""
        bible_content = load_bible()
        full_prompt = BASE_SYSTEM_PROMPT + bible_content
        
        # Reset history or update the system message?
        # For simplicity, we just keep the system prompt at index 0
        if len(self.history) > 0 and self.history[0]['role'] == 'system':
            self.history[0]['content'] = full_prompt
        else:
            self.history.insert(0, {'role': 'system', 'content': full_prompt})

    def write(self, prompt):
        # 1. Refresh context (in case you edited the bible file while it was running)
        self.refresh_system_prompt()
        
        self.history.append({'role': 'user', 'content': prompt})
        
        print("\nWriting...", end="", flush=True)
        start_time = time.time()
        print("\r", end="")
        
        full_response = ""
        stream = ollama.chat(model=MODEL, messages=self.history, stream=True)
        
        # 2. Print output in a nice block
        print("-" * 60)
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_response += content
        print("\n" + "-" * 60)
            
        # 3. Stats & Saving
        duration = time.time() - start_time
        tps = (len(full_response) / 4) / duration
        print(f"[Stats: {len(full_response)} chars | {tps:.1f} tok/s]")
        
        save_scene(full_response)
        
        # 4. Update History
        self.history.append({'role': 'assistant', 'content': full_response})

# --- MAIN LOOP ---
if __name__ == "__main__":
    writer = Novelist()
    
    # Create a dummy bible if it doesn't exist
    if not os.path.exists(BIBLE_FILE):
        with open(BIBLE_FILE, "w") as f:
            f.write("Protagonist: Kael (Bio-hacker, Human eye: Brown, Implant: Blue).\n")
            f.write("Setting: Neo-Kyoto (Rains constantly, smells of ozone).\n")
    
    print(f"\nTip: Edit '{BIBLE_FILE}' anytime to add new characters or facts.")
    print("Type your plot instruction below (or 'exit').\n")
    
    while True:
        user_input = input("\n>> Scene Direction: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        writer.write(user_input)