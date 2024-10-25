import os, json, requests, random, time, shutil, runpod

import torch
from PIL import Image
import numpy as np

import nodes
from nodes import NODE_CLASS_MAPPINGS
from nodes import load_custom_node

import asyncio
import execution
import server
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server)

load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-MochiWrapper")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite")

CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
DownloadAndLoadMochiModel = NODE_CLASS_MAPPINGS["DownloadAndLoadMochiModel"]()
MochiTextEncode = NODE_CLASS_MAPPINGS["MochiTextEncode"]()
MochiSampler = NODE_CLASS_MAPPINGS["MochiSampler"]()
MochiDecode = NODE_CLASS_MAPPINGS["MochiDecode"]()
VHS_VideoCombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

with torch.inference_mode():
    clip = CLIPLoader.load_clip('google_t5-v1_1-xxl_encoderonly-fp16.safetensors', type="sd3")[0]
    model, vae = DownloadAndLoadMochiModel.loadmodel('mochi_preview_dit_bf16.safetensors', 'mochi_preview_vae_bf16.safetensors', 'bf16', 'flash_attn')

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(input):
    values = input["input"]

    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    width = values['width']
    height = values['height']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    num_frames = values['num_frames']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    positive = MochiTextEncode.process(clip, positive_prompt, strength=1.0, force_offload=True)[0]
    negative = MochiTextEncode.process(clip, negative_prompt, strength=1.0, force_offload=True)[0]
    samples = MochiSampler.process(model, positive, negative, steps, cfg, seed, height, width, num_frames)[0]
    enable_vae_tiling = True
    tile_sample_min_height = 160
    tile_sample_min_width = 312
    tile_overlap_factor_height = 0.25
    tile_overlap_factor_width = 0.25
    auto_tile_size = False
    frame_batch_size = 10
    frames = MochiDecode.decode(vae, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, tile_overlap_factor_width, auto_tile_size, frame_batch_size)[0]
    out_video = VHS_VideoCombine.combine_video(images=frames, frame_rate=24, loop_count=0, filename_prefix="Mochi", format="video/h264-mp4", save_output=True, prompt=None, unique_id=None)
    source = out_video["result"][0][1][1]
    destination = '/content/ComfyUI/output/mochi-1-preview-tost.mp4'
    shutil.move(source, destination)

    result = '/content/ComfyUI/output/mochi-1-preview-tost.mp4'
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})