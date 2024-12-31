from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif, export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import datetime
import os
import threading
import glob
from PIL import Image
import io
import base64
import psutil
import torch
import numpy as np
import psutil
import subprocess
import json
import argparse

parser = argparse.ArgumentParser(
                    prog='AnimDiffWeb',
                    description='Web interface for AnimateDiff Lightning')

parser.add_argument('-c', '--config', default="config.json")

args = parser.parse_args()

config = None

with open(args.config) as f:
    config = json.load(f)
    print("Config loaded")


device = config["device"]#"cuda"
dtype = torch.float16

output_dir = config["outputDir"]#"//Desktop-4090/D/AnimDiff/outputs"

# Setup AnimateDiff Pipeline
step = config["animateDiffSteps"]#8  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = config["baseModel"]#"Lykon/AbsoluteReality" #"digiplay/majicMIX_realistic_v7" #"frankjoshua/realisticVisionV51_v51VAE"

def load_pipeline():
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    return pipe

def unload_pipeline(pipe):
    del pipe
    torch.cuda.empty_cache()



pipe = None
last_prompt = config["defaults"]["prompt"]#
last_negative_prompt = config["defaults"]["negativePrompt"]#
last_height = config["defaults"]["height"]#
last_width = config["defaults"]["width"]#
last_guidance = config["defaults"]["guidance"]#
last_steps = config["defaults"]["steps"]#
last_noise_factor = config["defaults"]["noiseFactor"]#
last_num_frames = config["defaults"]["numFrames"]#
last_num_videos = config["defaults"]["numVideos"]#




def add_noise_to_image(image, noise_factor=0.05):
    image = np.array(image)
    
    noise = np.random.normal(scale=noise_factor * 255, size=image.shape).astype(np.int32)
    
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_image)

def get_system_usage():
    system_ram = psutil.virtual_memory()
    system_ram_usage = system_ram.percent
    system_total_ram = system_ram.total / (1024 ** 3)
    
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024 ** 2)
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        gpu_usage = (gpu_memory_allocated / total_gpu_memory) * 100
        vram_usage = gpu_memory_allocated
    else:
        gpu_usage = 0
        vram_usage = 0
        total_gpu_memory = 0

    return {
        "gpu_usage": gpu_usage,
        "vram_usage": vram_usage,
        "total_vram": total_gpu_memory,
        "system_ram_usage": system_ram_usage,
        "system_total_ram": system_total_ram
    }


def send_system_usage():
    while True:
        usage = get_system_usage()
        socketio.emit('system_usage', usage)
        socketio.sleep(2)

def progress_callback(local_pipe, step_index, timestep, callback_kwargs):
    progress = step_index / pipe.num_timesteps
    percent_complete = int(progress * 100 if progress is not None else 0)

    latents = callback_kwargs["latents"]

    image = latents_to_rgb(latents[0][0][0])
    img_str = ""
    if image != None:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    socketio.emit('progress', {'percent': percent_complete, 'image': img_str})

    return callback_kwargs

def export_to_gif_with_noise(frames, gif_path, noise_factor=0.05):
    noisy_frames = []
    
    for frame in frames:
        noisy_frame = add_noise_to_image(frame, noise_factor=noise_factor)
        noisy_frames.append(noisy_frame)
    
    export_to_gif(noisy_frames, gif_path)

app = Flask(__name__)
socketio = SocketIO(app)

def latents_to_rgb(latents):

    print(latents.shape)

    return None

    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)




def get_gif_paths():
    gif_paths = []
    gif_dir = output_dir  
    for folder in os.listdir(gif_dir):
        folder_path = os.path.join(gif_dir, folder)
        if os.path.isdir(folder_path) and "liked" not in folder_path:
            for file in os.listdir(folder_path):
                if file.endswith(".gif"):
                    gif_paths.append(os.path.join(folder_path, file).replace(output_dir, ""))
    return list(reversed(gif_paths))

@app.route('/')
def index():
    global last_prompt
    global last_negative_prompt
    gifs = [ x.replace("\\", "/") for x in get_gif_paths()][:20]
    print(gifs)
    return render_template('index.html', folders=gifs, 
                           last_prompt=last_prompt, 
                           last_negative_prompt=last_negative_prompt,
                           last_height=last_height,
                           last_width=last_width,
                           last_guidance=last_guidance,
                           last_steps=last_steps,
                           last_noise_factor=last_noise_factor,
                           last_num_frames=last_num_frames,
                           last_num_videos=last_num_videos,
                           )

@app.route('/create', methods=['POST'])
def create():
    global pipe
    global last_prompt
    global last_negative_prompt
    global last_height
    global last_width
    global last_guidance
    global last_steps
    global last_noise_factor
    global last_num_frames
    global last_num_videos

    socketio.emit('progress', {'percent': 0, 'image': ""})

    if pipe is None:
        pipe = load_pipeline()

    data = request.json
    prompt = data['prompt']
    negative_prompt = data['negative_prompt']
    height = int(data['height'])
    width = int(data['width'])
    guidance_scale = float(data['guidance_scale'])
    steps = int(data['steps'])
    noise_factor = float(data['noise_factor'])
    num_frames = int(data['num_frames'])
    num_videos = int(data['num_videos'])

    last_prompt = prompt
    last_negative_prompt = negative_prompt
    last_height = height
    last_width = width
    last_guidance = guidance_scale
    last_steps = steps
    last_noise_factor = noise_factor
    last_num_frames = num_frames
    last_num_videos = num_videos

    folder_name = f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"

    def generate():
        global pipe
        try:
            for i in range(num_videos):
                output = pipe(prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            num_inference_steps=steps,
                            num_frames=num_frames,
                            callback_on_step_end=progress_callback)
                
                os.makedirs(f"{output_dir}/{folder_name}", exist_ok=True)

                gif_path = f"{output_dir}/{folder_name}/{i}.gif"
                video_path = f"{output_dir}/{folder_name}/{i}.mp4"

                export_to_gif_with_noise(output.frames[0], gif_path, noise_factor=noise_factor)

                export_to_video(output.frames[0], video_path)

                socketio.emit('progress', {'percent': 100, 'folder': gif_path.replace(output_dir, "")})
        finally:
            unload_pipeline(pipe)
            pipe = None

    threading.Thread(target=generate).start()

    return jsonify({"status": "started", "folder": folder_name})

@app.route('/gif/<path:path>')
def get_gif(path):
    print(path)
    path = f"{output_dir}/{path}"
    if ".gif" in path:
        return send_file(f"{path}", mimetype='image/gif')
    elif ".mp4" in path:
        return send_file(f"{path}", mimetype='video/mp4')

@app.route('/all')
def all_gifs():
    gifs = [x.replace("\\", "/") for x in get_gif_paths()]
    return render_template('all_gifs.html', gifs=gifs[:15], start_index=15)

@app.route('/more_gifs')
def more_gifs():
    start = int(request.args.get('start'))
    gifs = [x.replace("\\", "/") for x in get_gif_paths()]
    more_gifs = gifs[start:start+5]
    return jsonify({"gifs": more_gifs})


@app.route('/createlooped')
def create_looped():
    raw_path = str(request.args.get('path'))
    path = f"{output_dir}/{raw_path.replace('.gif', '.mp4')}"
    print(f"Creating looped video for {raw_path}")
    create_reversed_loop(path)
    return jsonify({"success": True})


def get_fps(input_path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", input_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fps = result.stdout.strip()
    if '/' in fps:
        num, denom = map(int, fps.split('/'))
        fps = num / denom
    return fps

def create_reversed_loop(input_path):
    try:
        base_dir = os.path.dirname(input_path)
        forward_path = os.path.join(base_dir, "forward.mp4")
        reversed_path = os.path.join(base_dir, "reversed.mp4")
        concat_list_path = os.path.join(base_dir, "concat_list.txt")
        output_path = os.path.join(base_dir, "looped.mp4")
        frames_dir = os.path.join(base_dir, "frames")

        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        fps = get_fps(input_path)
        print(f"Detected FPS: {fps}")

        encode_forward_command = [
            "ffmpeg", "-y", "-i", input_path, "-r", str(fps), "-g", "50",
            "-vf", "eq=brightness=0.0:contrast=1.0:saturation=1.0",
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", forward_path
        ]
        subprocess.run(encode_forward_command, check=True)

        reverse_command = [
            "ffmpeg", "-y", "-i", input_path, "-vf", "reverse,eq=brightness=0.0:contrast=1.0:saturation=1.0", 
            "-af", "areverse", "-r", str(fps), "-g", "50",
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", reversed_path
        ]
        subprocess.run(reverse_command, check=True)

        extract_frames_command = [
            "ffmpeg", "-y", "-i", input_path, os.path.join(frames_dir, "frame_%04d.png")
        ]
        subprocess.run(extract_frames_command, check=True)

        with open(concat_list_path, "w") as f:
            for _ in range(5):
                f.write(f"file '{forward_path}'\n")
                f.write(f"file '{reversed_path}'\n")

        concat_command = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-strict", "experimental", output_path
        ]
        subprocess.run(concat_command, check=True)

        os.remove(forward_path)
        os.remove(reversed_path)
        os.remove(concat_list_path)

        print(f"Saved reversed loop video to {output_path}")
        print(input_path)

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")




if __name__ == '__main__':
    socketio.start_background_task(send_system_usage)
    socketio.run(app, host='0.0.0.0', port=config["port"], debug=True)
