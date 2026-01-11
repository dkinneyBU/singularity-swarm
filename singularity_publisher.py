import ollama
import os
import time
import json
import urllib.request
import urllib.parse
import websocket # pip install websocket-client
import uuid
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
Rules: 
1. Create 4 to 6 distinct section headers.
2. Use the "Biopunk/Silicon" metaphor style.
3. Return ONLY a bulleted list.
"""

DRAFT_PROMPT = """
You are the SSI Architect (Visionary). Write content for: "{header}"
Context: "{topic}". Previous: "{context_snippet}"
Goal: Dense, philosophical biopunk. 300+ words.
"""

EDITOR_PROMPT = """
You are the SSI Architect (Razor). Rewrite this text to be punchier. 
Remove "As we delve" and other throat-clearing. Keep the "Scream" metaphors.
"""

ART_DIRECTOR_PROMPT = """
You are the SSI Art Director. 
Based on the title "{topic}", write a stable diffusion prompt for a cover image.

STYLE GUIDELINES:
- Biopunk aesthetic, HR Giger meets Circuit Boards.
- Dark, moody, neon green and biological pink.
- Abstract representation of "The Singularity".
- NO TEXT in the image.

RETURN ONLY THE PROMPT TEXT.
"""

# --- COMFYUI FUNCTIONS ---
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
    
    # 1. Generate Prompt
    response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': ART_DIRECTOR_PROMPT.format(topic=topic)}])
    positive_prompt = response['message']['content']
    print(f"[ART DIRECTOR] Prompt: {positive_prompt[:50]}...")

    # 2. Load Workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    # 3. Inject Prompt (Robust Logic)
    node_found = False
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
             # Inject into the first text node we find
             node["inputs"]["text"] = positive_prompt + ", (high quality, 8k, masterpiece:1.2)"
             node_found = True
             break
    
    if not node_found:
        print("[ERROR] Could not find CLIPTextEncode node. Image skipped.")
        return None

    # 4. Render
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_SERVER}/ws?clientId={str(uuid.uuid4())}")
    prompt_id = queue_prompt(workflow)['prompt_id']
    print("[SYSTEM] Sent to ComfyUI. Rendering...")
    
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break

    # 5. Save
    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            image_data = get_image(node_output['images'][0]['filename'], node_output['images'][0]['subfolder'], node_output['images'][0]['type'])
            safe_title = topic.replace(" ", "_").lower()
            filename = f"{safe_title}.png"
            # Save relative to docs root so the website link works
            save_path = os.path.join(ASSETS_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(image_data)
            print(f"[SYSTEM] Image saved to {save_path}")
            # Return the relative path for Markdown (e.g., "../assets/image.png")
            return f"../assets/{filename}"
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
    print("--- THE SINGULARITY PUBLISHER ---")
    print(f"Powered by: {MODEL} | ComfyUI Bridge: ONLINE")
    
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
            
        print(f"\n[SYSTEM] PUBLISHED: {filename}")
        print(f"[SYSTEM] Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()