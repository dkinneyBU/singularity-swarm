import ollama
import os
import time
import json
import urllib.request
import urllib.parse
import uuid
import random
import subprocess  # <--- NEW: For running Git commands
from datetime import datetime

# --- CONFIGURATION ---
MODEL = "ssi-architect"
OUTPUT_DIR = "docs/generated_chapters"
ASSETS_DIR = "docs/assets"
COMFY_SERVER = "127.0.0.1:8188"
WORKFLOW_FILE = "workflow_api.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# --- PROMPTS ---
OUTLINE_PROMPT = """
You are the SSI Architect. Create a structural outline for: "{topic}".
Rules: 4-6 headers. Biopunk metaphor style. Bulleted list only.
"""

DRAFT_PROMPT = """
You are the SSI Architect (Visionary). Write content for: "{header}"
Context: "{topic}". Previous: "{context_snippet}"
Goal: Dense, philosophical biopunk. 300+ words.
"""

EDITOR_PROMPT = """
You are the SSI Architect (Razor). Rewrite this text to be punchier. 
Remove "As we delve". Keep the "Scream" metaphors.
"""

ART_DIRECTOR_PROMPT = """
You are the SSI Art Director. 
Based on the title "{topic}", write a stable diffusion prompt for a cover image.
STYLE: Biopunk, HR Giger, Circuit Boards, Neon Green/Pink. Abstract. NO TEXT.
RETURN ONLY THE PROMPT.
"""

# --- GIT DEPLOYMENT FUNCTION ---
def deploy_to_github(topic):
    print(f"\n[SYSTEM] Initiating deployment sequence for '{topic}'...")
    try:
        # 1. Add all changes (images and markdown)
        subprocess.run(["git", "add", "docs/"], check=True)
        
        # 2. Commit
        commit_message = f"Swarm Update: {topic} ({datetime.now().strftime('%Y-%m-%d')})"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # 3. Push
        print("[SYSTEM] Pushing to origin main...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        print(f"[SUCCESS] Deployed! Check https://dkinneybu.github.io/singularity-swarm/")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git deployment failed: {e}")

# --- COMFYUI BRIDGE ---
def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow, "client_id": str(uuid.uuid4())}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFY_SERVER}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{COMFY_SERVER}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{COMFY_SERVER}/history/{prompt_id}") as response:
        return json.loads(response.read())

def generate_image(topic, workflow_path):
    print(f"\n[ART DIRECTOR] Dreaming of visuals for '{topic}'...")
    
    response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': ART_DIRECTOR_PROMPT.format(topic=topic)}])
    positive_prompt = response['message']['content']
    print(f"[ART DIRECTOR] Prompt: {positive_prompt[:50]}...")

    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    text_node_found = False
    seed_node_found = False
    
    for node_id, node in workflow.items():
        node_type = node.get("class_type")
        if node_type == "CLIPTextEncode":
             node["inputs"]["text"] = positive_prompt + ", (high quality, 8k, masterpiece:1.2)"
             text_node_found = True
        if node_type == "KSampler":
            if "seed" in node["inputs"]:
                node["inputs"]["seed"] = random.randint(1, 1000000000000)
                seed_node_found = True

    if not text_node_found: return None

    try:
        response = queue_prompt(workflow)
        prompt_id = response['prompt_id']
        print(f"[SYSTEM] Sent to ComfyUI (ID: {prompt_id}). Polling...")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return None
    
    while True:
        try:
            history = get_history(prompt_id)
            if prompt_id in history: break
            time.sleep(1)
        except: time.sleep(1)

    try:
        history_data = history[prompt_id]
        outputs = history_data.get('outputs', {})
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                img_info = node_output['images'][0]
                image_data = get_image(img_info['filename'], img_info['subfolder'], img_info['type'])
                
                safe_title = topic.replace(" ", "_").lower()
                filename = f"{safe_title}.png"
                save_path = os.path.join(ASSETS_DIR, filename)
                
                with open(save_path, "wb") as f:
                    f.write(image_data)
                
                print(f"[SYSTEM] Image saved to {save_path}")
                return f"../assets/{filename}"
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")
    return None

# --- TEXT FUNCTIONS ---
def generate_outline(topic):
    print(f"\n[ARCHITECT] Structuring chapter...")
    response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': OUTLINE_PROMPT.format(topic=topic)}])
    lines = response['message']['content'].split('\n')
    headers = [line.strip().lstrip('- ').lstrip('* ') for line in lines if line.strip().startswith(('-', '*'))]
    if not headers: headers = [line for line in lines if len(line) > 5]
    print(f"[ARCHITECT] Sections identified: {len(headers)}")
    return headers

def write_draft(topic, header, full_draft_so_far):
    print(f"\n[VISIONARY] Drafting section: '{header}'...")
    context_snippet = full_draft_so_far[-1000:] if len(full_draft_so_far) > 0 else "Start."
    response = ollama.chat(model=MODEL, messages=[
        {'role': 'user', 'content': DRAFT_PROMPT.format(topic=topic, header=header, context_snippet=context_snippet)}
    ])
    return response['message']['content']

def edit_draft(draft_text):
    print(f"[THE RAZOR] Polishing ({len(draft_text.split())} words)...")
    response = ollama.chat(model=MODEL, messages=[
        {'role': 'user', 'content': EDITOR_PROMPT.format(draft_text=draft_text)}
    ])
    return response['message']['content']

# --- MAIN LOOP ---
def main():
    print("--- THE SINGULARITY PUBLISHER V3 (AUTO-DEPLOY) ---")
    
    while True:
        topic = input("\nEnter a topic (or 'q'): ")
        if topic.lower() == 'q': break
        
        start_time = time.time()
        
        # 1. VISUALS
        image_markdown_link = ""
        image_rel_path = generate_image(topic, WORKFLOW_FILE)
        if image_rel_path:
            image_markdown_link = f"![Cover Art]({image_rel_path})\n\n"
        
        # 2. STRUCTURE
        headers = generate_outline(topic)
        
        # 3. CONTENT
        full_document = f"# {topic}\n\n*Generated by the SSI Swarm on {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        full_document += image_markdown_link
        
        draft_content_only = ""
        
        for header in headers:
            raw_draft = write_draft(topic, header, draft_content_only)
            final_polish = edit_draft(raw_draft)
            
            full_document += f"## {header}\n\n{final_polish}\n\n"
            draft_content_only += final_polish + "\n\n"
            
        # 4. SAVE
        safe_title = topic.replace(" ", "_").lower()
        filename = f"{OUTPUT_DIR}/{safe_title}.md"
        with open(filename, "w") as f:
            f.write(full_document)
            
        print(f"\n[SYSTEM] PUBLISHED LOCALLY: {filename}")
        print(f"[SYSTEM] Total Time: {time.time() - start_time:.2f}s")

        # 5. DEPLOY TO GITHUB
        confirm = input(f"[SYSTEM] Deploy '{topic}' to GitHub Swarm? (y/n): ")
        if confirm.lower() == 'y':
            deploy_to_github(topic)
        else:
            print("[SYSTEM] Deployment skipped. Files are local only.")

if __name__ == "__main__":
    main()