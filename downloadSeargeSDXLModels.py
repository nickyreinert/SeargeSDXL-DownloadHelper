import curses
from huggingface_hub import snapshot_download
import os

def draw_menu(stdscr, models):
    # Message and instructions
    message = (
        "Select the models you wish to download using the arrow keys.\n"
        "Press SPACE to select/unselect, ENTER to confirm.\n"
        "Required models are pre-selected and cannot be changed.\n"
    )
    
    # Split the message into lines
    message_lines = message.splitlines()
    num_message_lines = len(message_lines)
    
    # Calculate the minimum screen size needed
    min_rows = num_message_lines + len(models) + 1
    min_cols = max(len(line) for line in message_lines) + 5  # Some padding for model lines

    # Clear screen and check if the terminal size is sufficient
    stdscr.clear()
    stdscr.refresh()
    
    # Get the screen size
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    
    max_rows, max_cols = stdscr.getmaxyx()

    # Check if the terminal size is sufficient
    if max_rows < min_rows or max_cols < min_cols:
        stdscr.addstr(0, 0, "The terminal window is too small to display the content. Please resize and try again.")
        stdscr.refresh()
        stdscr.getch()  # Wait for user input before exiting
        return []

    # Display the message at the top
    for i, line in enumerate(message_lines):
        stdscr.addstr(i, 0, line)
    stdscr.addstr(num_message_lines, 0, "-" * 80)

    current_row = 0

    # Preselect recommended models and mark required models
    for model in models:
        if model["importance"] == "required":
            model["selected"] = True
        elif model["importance"] == "recommended":
            model["selected"] = True
        else:
            model["selected"] = False

    while True:
        try:
            # Draw the model list starting from the row after the message and separator
            for idx, model in enumerate(models):
                filename = model["filename"]
                importance = model["importance"]
                if importance == "required":
                    mark = "[*]"
                else:
                    mark = "[x]" if model.get("selected", False) else "[ ]"
                
                line = f"{mark} {filename} (importance: {importance})"
                row_position = idx + num_message_lines + 1
                if row_position < max_rows:
                    if idx == current_row:
                        stdscr.addstr(row_position, 0, line, curses.A_REVERSE)
                    else:
                        stdscr.addstr(row_position, 0, line)
                else:
                    break

            key = stdscr.getch()
            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(models) - 1:
                current_row += 1
            elif key == ord(' '):
                if models[current_row]["importance"] != "required":
                    models[current_row]["selected"] = not models[current_row].get("selected", False)
            elif key == ord('\n'):
                break

        except curses.error:
            pass

    # Collect selected models
    selected_models = [model for model in models if model.get("selected", False)]
    return selected_models

def download_selected_models(models, token=None, local_dir='./'):
    # Check if the local directory exists; if not, quit with a warning
    if not os.path.exists(local_dir):
        print(f"Error: Directory '{local_dir}' does not exist.")
        print("This script must be run from the ComfyUI base folder.")
        return  # Exit the function and quit the process
    
    print(f"Directory exists: {local_dir}")

    # Call curses.wrapper with the draw_menu function and models as additional argument
    selected_models = curses.wrapper(draw_menu, models)

    # Download selected models
    for model in selected_models:
        filename = model["filename"]
        url = model["url"]
        folder = model["folder"]
        file_path = os.path.join(local_dir, folder, filename)
        
        # Create the folder if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Extract repo_id from the URL
        repo_id = "/".join(url.split("/")[3:5])  # Extracts the 'stabilityai/stable-diffusion-xl-base-1.0' part
        
        # Download the file using snapshot_download
        print(f"Downloading {filename} from {repo_id}...")
        snapshot_download(
            token=token,
            repo_id=repo_id,
            allow_patterns=filename,
            local_dir=os.path.dirname(file_path)
        )
        print(f"Downloaded {filename} to {file_path}")

# Define the list of models
MODELS = [
    {
        "filename": "sd_xl_base_1.0_0.9vae.safetensors",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/",
        "folder": "checkpoints",
        "importance": "required",
    },
    {
        "filename": "sd_xl_refiner_1.0_0.9vae.safetensors",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/",
        "folder": "checkpoints",
        "importance": "recommended",
    },
    {
        "filename": "sdxl_vae.safetensors",
        "url": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/",
        "folder": "vae",
        "importance": "optional",
    },
    {
        "filename": "sd_xl_offset_example-lora_1.0.safetensors",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/",
        "folder": "loras",
        "importance": "optional",
    },
    {
        "filename": "4x-UltraSharp.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "recommended",
    },
    {
        "filename": "4x_NMKD-Siax_200k.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "recommended",
    },
    {
        "filename": "4x_Nickelback_70000G.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "recommended",
    },
    {
        "filename": "1x-ITF-SkinDiffDetail-Lite-v1.pth",
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/",
        "folder": "upscale_models",
        "importance": "optional",
    },
    {
        "filename": "ControlNetHED.pth",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/",
        "folder": "annotators",
        "importance": "required",
    },
    {
        "filename": "res101.pth",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/",
        "folder": "annotators",
        "importance": "required",
    },
    {
        "filename": "clip_vision_g.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/revision/",
        "folder": "clip_vision",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-canny-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-depth-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-recolor-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    },
    {
        "filename": "control-lora-sketch-rank256.safetensors",
        "url": "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/",
        "folder": "controlnet",
        "importance": "recommended",
    }
]

# Use the function
download_selected_models(MODELS, token="ABC", local_dir="./models")
