from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dataclasses import dataclass, fields
import gradio as gr
import threading
import time
import os
import re
import requests
import json
import subprocess
import sys
import ast
import time
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import huggingface_hub
from huggingface_hub import snapshot_download
import logging
import psutil
import git
from git import Repo
# import redis
import redis.asyncio as redis





print(f'** connecting to redis on port: {os.getenv("REDIS_PORT")} ... ')
# r = redis.Redis(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0)
pool = redis.ConnectionPool(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0, decode_responses=True, max_connections=10)
r = redis.Redis(connection_pool=pool)
pipe = r.pipeline()


async def update_timer():
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} bla"

VLLM_URL = f'http://container_vllm_xoo:{os.getenv("VLLM_PORT")}/status'
BACKEND_URL = f'http://container_backend:{os.getenv("BACKEND_PORT")}/docker'
IMAGE_URL = f'http://container_video:{os.getenv("VIDEO_PORT")}/generateimage'
VIDEO_URL = f'http://container_video:{os.getenv("VIDEO_PORT")}/generatevideo'
VIP_URL = f'http://container_video:{os.getenv("VIDEO_PORT")}/generatevip'
TR_URL = f'http://container_tr:{os.getenv("TR_PORT")}'

DEFAULT_PROMPT = "You are a merchant on the Silk Road in the 13th century. Write a journal entry describing a typical day, including the sights, sounds, people you meet, and the risks you face"
DEFAULT_IMAGE = f'/usr/src/app/video/dragon.png'
DEFAULT_VIDEO = f'/usr/src/app/video/napoli.mp4'

REQUEST_TIMEOUT = 300

def wait_for_backend():
    start_time = time.time()
    while time.time() - start_time < REQUEST_TIMEOUT:
        try:
            response = requests.post(BACKEND_URL, json={"method": "list"}, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print("Backend container is online.")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Backend is not yet reachable
        time.sleep(5)  # Wait for 5 seconds before retrying
    print(f"Timeout: Backend container did not come online within {REQUEST_TIMEOUT} seconds.")
    return False


VLLM_REQUEST_TIMEOUT = 30

def wait_for_vllm():
    start_time = time.time()
    while time.time() - start_time < VLLM_REQUEST_TIMEOUT:
        try:
            print("genau")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        print("zzz")
        time.sleep(5)
    print(f'zzz finished! trying to prompt again ...')
    return True



GLOBAL_VLLMS = []
GLOBAL_FISH = []

test_vllms = []
test_vllms_list_running = []

docker_container_list = []
current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''
SELECTED_MODEL_ID = ''
SELECTED_XOO_ID = ''
SELECTED_XOO_OBJ = ''
MEM_TOTAL = 0
MEM_USED = 0
MEM_FREE = 0

SEARCH_INPUT_TS = 0
SEARCH_INPUT_THRESHOLD = 10
SEARCH_REQUEST_TIMEOUT = 3
SEARCH_INITIAL_DELAY = 10


print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ FRONTEND WAITING FOR BACKEND BOOT TO GET VLLMS')
print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ querying ...')
res_backend = wait_for_backend()
print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ true or no?  res_backend: {res_backend}')
if res_backend:
    print(f' ~~~~~~~~~~~~~~~~~~~~~~~~ ok is true... trying to get vllms ...')
else:    
    print(f' ~~~~~~~~~~~ ERRROR ~~~~~~~~~~~~~ 4 responded False')













LOG_PATH= './logs'
LOGFILE_CONTAINER = './logs/logfile_container_frontend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f' [START] started logging in {LOGFILE_CONTAINER}')

def load_log_file(req_container_name):
    print(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    logging.info(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_name}')
    try:
        with open(f'{LOG_PATH}/logfile_{req_container_name}.log', "r", encoding="utf-8") as file:
            lines = file.readlines()
            last_20_lines = lines[-20:]
            reversed_lines = last_20_lines[::-1]
            return ''.join(reversed_lines)
    except Exception as e:
        return f'{e}'



DEFAULTS_PATH = "/usr/src/app/utils/defaults.json"
if not os.path.exists(DEFAULTS_PATH):
    logging.info(f' [START] File missing: {DEFAULTS_PATH}')

with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
    defaults_frontend = json.load(f)["frontend"]
    logging.info(f' [START] SUCCESS! Loaded: {DEFAULTS_PATH}')
    logging.info(f' [START] {len(defaults_frontend['vllm_supported_architectures'])} supported vLLM architectures found!')



























        
        
def dropdown_load_tested_models():

    global current_models_data
    response_models = defaults_frontend['tested_models']
    print(f'response_models: {response_models}')
    current_models_data = response_models.copy()
    model_ids = [m["id"] for m in response_models]
    print(f'model_ids: {model_ids}')
    return [gr.update(choices=model_ids, value=response_models[0]["id"], visible=True),gr.update(value=response_models[0]["id"],show_label=True, label=f'Loaded {len(model_ids)} models!')]



async def get_network_data():
    try:
        res_network_data_all = json.loads(await r.get('db_network'))
        return res_network_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

async def get_gpu_data():
    try:
        res_gpu_data_all = json.loads(await r.get('db_gpu'))
        return res_gpu_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

async def get_disk_data():
    try:
        res_disk_data_all = json.loads(await r.get('db_disk'))
        return res_disk_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e





def docker_api(req_method,req_var):
    try:
        global BACKEND_URL
        global docker_container_list
        
        if req_method == "list":
            response = requests.post(BACKEND_URL, json={"method":req_method})
            res_json = response.json()
            docker_container_list = res_json.copy()
            if response.status_code == 200:
                return res_json
            else:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'Error: {response.status_code}'

        if req_method == "logs":
            response = requests.post(BACKEND_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return ''.join(res_json["result_data"])
        
        if req_method == "start":
            response = requests.post(BACKEND_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return res_json
        
        if req_method == "stop":
            response = requests.post(BACKEND_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return res_json
        
        if req_method == "delete":
            response = requests.post(BACKEND_URL, json={"method":req_method,"model":req_var})
            res_json = response.json()
            return res_json
        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker_api] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker_api] {e}'




async def get_dataframes():
    
    df_gpu = []
    df_disk = []
    df_vllm = []
    df_fish = []
    
    # gpu_to_pd2
    
    global MEM_TOTAL
    global MEM_USED
    global MEM_FREE
    rows = []

    try:

        gpu_data2 = await r.get('gpu_key')

        current_data2 = json.loads(gpu_data2) if gpu_data2 else None

        for entry in current_data2:

            gpu_info = entry.copy()

            current_gpu_mem_total = gpu_info.get("mem_total", "0")
            current_gpu_mem_used = gpu_info.get("mem_used", "0")
            current_gpu_mem_free = gpu_info.get("mem_free", "0")
            MEM_TOTAL = float(MEM_TOTAL) + float(current_gpu_mem_total.split()[0])
            MEM_USED = float(MEM_USED) + float(current_gpu_mem_used.split()[0])
            MEM_FREE = float(MEM_FREE) + float(current_gpu_mem_free.split()[0])
        
            
            
            rows.append({        
                "mem_util": gpu_info.get("mem_util", "0"),            
                "ts": gpu_info.get("ts", "0"),
                "name": gpu_info.get("name", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "fan_speed": gpu_info.get("fan_speed", "0"),
                "temperature": gpu_info.get("temperature", "0"),
                "gpu_util": gpu_info.get("gpu_util", "0"),
                "power_usage": gpu_info.get("power_usage", "0"),
                "clock_info_graphics": gpu_info.get("clock_info_graphics", "0"),
                "clock_info_mem": gpu_info.get("clock_info_mem", "0"),                
                "cuda_cores": gpu_info.get("cuda_cores", "0"),
                "compute_capability": gpu_info.get("compute_capability", "0"),
                "current_uuid": gpu_info.get("current_uuid", "0"),
                "gpu_i": entry.get("gpu_i", "0"),
                "supported": gpu_info.get("supported", "0"),
                "not_supported": gpu_info.get("not_supported", "0"),
                "status": "ok"
            })

        df_gpu = pd.DataFrame(rows)
    

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_dataframes -> df_gpu] {e}')

    # disk_to_pd2
    rows = []
    try:
        disk_data = await r.get('disk_key')

        disk_data_json = json.loads(disk_data) if disk_data else None

        for entry in disk_data_json:

            rows.append({
                "ts": entry.get("ts", "0"),
                "disk_i": entry.get("disk_i", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "device": entry.get("device", "0"),
                "usage_percent": entry.get("usage_percent", "0"),
                "mountpoint": entry.get("mountpoint", "0"),
                "fstype": entry.get("fstype", "0"),
                "opts": entry.get("opts", "0"),
                "usage_total": entry.get("usage_total", "0"),
                "usage_used": entry.get("usage_used", "0"),
                "usage_free": entry.get("usage_free", "0"),                
                "io_read_count": entry.get("io_read_count", "0"),
                "io_write_count": entry.get("io_write_count", "0")                
            })
        df_disk = pd.DataFrame(rows)

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_dataframes -> df_disk] {e}')



    # vllm_to_pd2
    global GLOBAL_VLLMS
    rows = []

    try:

        vllm_data = await r.get('vllm_key')
        vllm_data_json = json.loads(vllm_data) if vllm_data else None
        for entry in vllm_data_json:

            rows.append({
                "ts": entry.get("ts", "0"),
                "name": entry.get("name", "0"),
                "container_name": entry.get("container_name", "0"),
                "uid": entry.get("uid", "0"),
                "status": entry.get("status", "0"),
                "gpu_list": entry.get("gpu_list", "0"),
                "mem": entry.get("mem", "0"),
                "gpu": entry.get("gpu", "0"),
                "temp": entry.get("temp", "0")
            })
        df_vllm = pd.DataFrame(rows)
        rows_copy = df_vllm.copy()
        GLOBAL_VLLMS = rows_copy
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_dataframes -> df_vllm] {e}')


    # fish_to_pd2
    
    global GLOBAL_FISH
    rows = []

    try:

        fish_data = await r.get('fish_key')
        fish_data_json = json.loads(fish_data) if fish_data else None
        GLOBAL_FISH = fish_data_json if fish_data_json else []
        
        for entry in fish_data_json:

            rows.append({
                "ts": entry.get("ts", "0"),
                "name": entry.get("name", "0"),
                "container_name": entry.get("container_name", "0"),
                "uid": entry.get("uid", "0"),
                "status": entry.get("status", "0"),
                "mem": entry.get("mem", "0")
            })
        df_fish = pd.DataFrame(rows)



    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_dataframes -> df_fish] {e}')

    return df_gpu, df_disk, df_vllm, df_fish




def search_change(input_text):
    global SEARCH_INPUT_TS
    global current_models_data
    current_ts = int(datetime.now().timestamp())
    if SEARCH_INPUT_TS + SEARCH_INPUT_THRESHOLD > current_ts:
        wait_time = SEARCH_INPUT_TS + SEARCH_INPUT_THRESHOLD - current_ts
        return [gr.update(show_label=False),gr.update(show_label=True, label=f'Found {len(current_models_data)} models! Please wait {wait_time} sec or click on search')]
    if len(input_text) < 3: 
        # return [gr.update(show_label=False),gr.update(show_label=True, label=" < 3")]
        return [gr.update(show_label=False),gr.update(show_label=True)]
    if SEARCH_INPUT_TS == 0 and len(input_text) > 5:
        SEARCH_INPUT_TS = int(datetime.now().timestamp())
        res_huggingface_hub_search_model_ids,  res_huggingface_hub_search_current_value = search_models(input_text)
        if len(res_huggingface_hub_search_model_ids) >= 1000:
            return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found >1000 models!')]
        return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found {len(res_huggingface_hub_search_model_ids)} models!')]
        
    if SEARCH_INPUT_TS == 0:
        SEARCH_INPUT_TS = int(datetime.now().timestamp()) + SEARCH_INITIAL_DELAY
        return [gr.update(show_label=False),gr.update(show_label=True, label=f'Waiting auto search {SEARCH_INITIAL_DELAY} sec')]
    if SEARCH_INPUT_TS + SEARCH_INPUT_THRESHOLD <= current_ts:
        SEARCH_INPUT_TS = int(datetime.now().timestamp())
        res_huggingface_hub_search_model_ids,  res_huggingface_hub_search_current_value = search_models(input_text)
        if len(res_huggingface_hub_search_model_ids) >= 1000:
            return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found >1000 models!')]
        return [gr.update(choices=res_huggingface_hub_search_model_ids, value=res_huggingface_hub_search_current_value, visible=True),gr.update(show_label=True, label=f'Found {len(res_huggingface_hub_search_model_ids)} models!')]















def search_models(query):
    try:
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}')
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        if len(model_ids) < 1:
            model_ids = ["No models found!"]
        return gr.update(choices=model_ids, value=response_models[0]["id"], show_label=True, label=f'found {len(response_models)} models!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')























def format_bytes(req_format, req_size):
    if req_format == "human":
        for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
            if abs(req_size) < 1024.0:
                return f'{req_size:3.1f}{unit}B'
            req_size /= 1024.0
        return f'{req_size:.1f}YiB'
    elif req_format == "bytes":
        req_size = req_size.upper()
        if 'KB' in req_size:
            return int(float(req_size.replace('KB', '').strip()) * 1024)
        elif 'MB' in req_size:
            return int(float(req_size.replace('MB', '').strip()) * 1024 * 1024)
        elif 'GB' in req_size:
            return int(float(req_size.replace('GB', '').strip()) * 1024 * 1024 * 1024)
        elif 'B' in req_size:
            return int(float(req_size.replace('B', '').strip()))
        return 0
    else:
        raise ValueError("Invalid format specified. Use 'human' or 'bytes'.")




def convert_to_bytes(size_str):
    """Convert human-readable file size to bytes"""
    size_str = size_str.upper()
    if 'KB' in size_str:
        return int(float(size_str.replace('KB', '').strip()) * 1024)
    elif 'MB' in size_str:
        return int(float(size_str.replace('MB', '').strip()) * 1024 * 1024)
    elif 'GB' in size_str:
        return int(float(size_str.replace('GB', '').strip()) * 1024 * 1024 * 1024)
    elif 'B' in size_str:
        return int(float(size_str.replace('B', '').strip()))
    return 0








def get_git_model_size(selected_id):    
    try:
        repo = Repo.clone_from(f'https://huggingface.co/{selected_id}', selected_id, no_checkout=True)
    except git.exc.GitCommandError as e:
        if "already exists and is not an empty directory" in str(e):
            repo = Repo(selected_id)
        else:
            raise
    
    lfs_files = repo.git.lfs("ls-files", "-s").splitlines()
    files_list = []
    for line in lfs_files:
        parts = line.split(" - ")
        if len(parts) == 2:
            file_hash, file_info = parts
            file_parts = file_info.rsplit(" (", 1)
            if len(file_parts) == 2:
                file_name = file_parts[0]
                size_str = file_parts[1].replace(")", "")
                size_bytes = format_bytes("bytes",size_str)
                
                files_list.append({
                    "id": file_hash.strip(),
                    "file": file_name.strip(),
                    "size": size_bytes,
                    "size_human": size_str
                })
            
        
    return sum([file["size"] for file in files_list]), format_bytes("human",sum([file["size"] for file in files_list]))
    
















def calculate_model_size(json_info): # to fix    
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0
        
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)

            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes * 2
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0





def get_info(selected_id):
    
    print(f' @@@ [get_info] 0')
    print(f' @@@ [get_info] 0')   
    container_name = ""
    res_model_data = {
        "search_data" : "",
        "model_id" : "",
        "pipeline_tag" : "",
        "architectures" : "",
        "transformers" : "",
        "private" : "",
        "downloads" : ""
    }
    
    if selected_id == None:
        print(f' @@@ [get_info] selected_id NOT FOUND!! RETURN ')
        print(f' @@@ [get_info] selected_id NOT FOUND!! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    
    global CURRENT_MODELS_DATA
    global SELECTED_MODEL_ID
    SELECTED_MODEL_ID = selected_id
    print(f' @@@ [get_info] {selected_id} 2')
    print(f' @@@ [get_info] {selected_id} 2')  
    
    print(f' @@@ [get_info] {selected_id} 3')
    print(f' @@@ [get_info] {selected_id} 3')  
    container_name = str(res_model_data["model_id"]).replace('/', '_')
    print(f' @@@ [get_info] {selected_id} 4')
    print(f' @@@ [get_info] {selected_id} 4')  
    if len(CURRENT_MODELS_DATA) < 1:
        print(f' @@@ [get_info] len(CURRENT_MODELS_DATA) < 1! RETURN ')
        print(f' @@@ [get_info] len(CURRENT_MODELS_DATA) < 1! RETURN ') 
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    try:
        print(f' @@@ [get_info] {selected_id} 5')
        print(f' @@@ [get_info] {selected_id} 5') 
        for item in CURRENT_MODELS_DATA:
            print(f' @@@ [get_info] {selected_id} 6')
            print(f' @@@ [get_info] {selected_id} 6') 
            if item['id'] == selected_id:
                print(f' @@@ [get_info] {selected_id} 7')
                print(f' @@@ [get_info] {selected_id} 7') 
                res_model_data["search_data"] = item
                
                if "pipeline_tag" in item:
                    res_model_data["pipeline_tag"] = item["pipeline_tag"]
  
                if "tags" in item:
                    if "transformers" in item["tags"]:
                        res_model_data["transformers"] = True
                    else:
                        res_model_data["transformers"] = False
                                    
                if "private" in item:
                    res_model_data["private"] = item["private"]
                                  
                if "architectures" in item:
                    res_model_data["architectures"] = item["architectures"][0]
                                                    
                if "downloads" in item:
                    res_model_data["downloads"] = item["downloads"]
                  
                container_name = str(res_model_data["model_id"]).replace('/', '_')
                
                print(f' @@@ [get_info] {selected_id} 8')
                print(f' @@@ [get_info] {selected_id} 8') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
            else:
                
                print(f' @@@ [get_info] {selected_id} 9')
                print(f' @@@ [get_info] {selected_id} 9') 
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
    except Exception as e:
        print(f' @@@ [get_info] {selected_id} 10')
        print(f' @@@ [get_info] {selected_id} 10') 
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["architectures"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name








def get_additional_info(selected_id):    
        res_model_data = {
            "hf_data" : "",
            "hf_data_config" : "",
            "config_data" : "",
            "architectures" : "",
            "model_type" : "",
            "quantization" : "",
            "tokenizer_config" : "",
            "model_id" : selected_id,
            "size" : 0,
            "size_human" : 0,
            "gated" : "",
            "torch_dtype" : "",
            "hidden_size" : "",
            "cuda_support" : "",
            "compute_capability" : ""
        }                
        try:
            try:
                model_info = huggingface_hub.model_info(selected_id)
                model_info_json = vars(model_info)
                res_model_data["hf_data"] = model_info_json
                
                if "config" in model_info.__dict__:
                    res_model_data['hf_data_config'] = model_info_json["config"]
                    if "architectures" in model_info_json["config"]:
                        res_model_data['architectures'] = model_info_json["config"]["architectures"][0]
                    if "model_type" in model_info_json["config"]:
                        res_model_data['model_type'] = model_info_json["config"]["model_type"]
                    if "tokenizer_config" in model_info_json["config"]:
                        res_model_data['tokenizer_config'] = model_info_json["config"]["tokenizer_config"]
                               
                if "gated" in model_info.__dict__:
                    res_model_data['gated'] = model_info_json["gated"]
                
                if "safetensors" in model_info.__dict__:
                    print(f'  FOUND safetensors')
                    print(f'  GFOUND safetensors')   
                    
                    safetensors_json = vars(model_info.safetensors)
                    
                    
                    print(f'  FOUND safetensors:::::::: {safetensors_json}')
                    print(f'  GFOUND safetensors:::::::: {safetensors_json}') 
                    try:
                        quantization_key = next(iter(safetensors_json['parameters'].keys()))
                        print(f'  FOUND first key in parameters:::::::: {quantization_key}')
                        res_model_data['quantization'] = quantization_key
                        
                    except Exception as get_model_info_err:
                        print(f'  first key NOT FOUND in parameters:::::::: {quantization_key}')
                        pass
                    
                    print(f'  FOUND safetensors TOTAL :::::::: {safetensors_json["total"]}')
                    print(f'  GFOUND safetensors:::::::: {safetensors_json["total"]}')
                                        
                    print(f'  ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    print(f'ooOOOOOOOOoooooo res_model_data["quantization"] {res_model_data["quantization"]}')
                    if res_model_data["quantization"] == "F32":
                        print(f'  ooOOOOOOOOoooooo found F32 -> x4')
                        print(f'ooOOOOOOOOoooooo found F32 -> x4')
                    else:
                        print(f'  ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        print(f'ooOOOOOOOOoooooo NUUUH FIND F32 -> x2')
                        res_model_data['size'] = int(safetensors_json["total"]) * 2
                else:
                    print(f' !!!!DIDNT FIND safetensors !!!! :::::::: ')
                    print(f' !!!!!! DIDNT FIND safetensors !!:::::::: ') 
            
            
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                response = requests.get(f'https://huggingface.co/{selected_id}/resolve/main/config.json', timeout=SEARCH_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json
                    
                    if "architectures" in res_model_data["config_data"]:
                        res_model_data["architectures"] = res_model_data["config_data"]["architectures"][0]
                        
                    if "torch_dtype" in res_model_data["config_data"]:
                        res_model_data["torch_dtype"] = res_model_data["config_data"]["torch_dtype"]
                        print(f'  ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                        print(f'ooOOOOOOOOoooooo torch_dtype: {res_model_data["torch_dtype"]}')
                    if "hidden_size" in res_model_data["config_data"]:
                        res_model_data["hidden_size"] = res_model_data["config_data"]["hidden_size"]
                        print(f'  ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                        print(f'ooOOOOOOOOoooooo hidden_size: {res_model_data["hidden_size"]}')
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            

            res_model_data["size"], res_model_data["size_human"] = get_git_model_size(selected_id)
            
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["architectures"], res_model_data["model_id"], gr.update(value=res_model_data["size"], label=f'size ({res_model_data["size_human"]})'), res_model_data["gated"], res_model_data["model_type"], res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]
        
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], gr.update(value=res_model_data["size"], label=f'size ({res_model_data["size_human"]})'), res_model_data["gated"], res_model_data["model_type"],  res_model_data["quantization"], res_model_data["torch_dtype"], res_model_data["hidden_size"]


def gr_load_check(selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization):
    

    
    # check CUDA support mit backend call
    
    # if "gguf" in selected_model_id.lower():
    #     return f'Selected a GGUF model!', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    req_model_storage = "/models"
    req_model_path = f'{req_model_storage}/{selected_model_id}'
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path}) ...')
    print(f' **************** [gr_load_check] searching {selected_model_id} in {req_model_storage} (req_model_path: {req_model_path})...')
    


    models_found = []
    # try:                   
    #     if os.path.isdir(req_model_storage):
    #         print(f' **************** found model storage path! {req_model_storage}')
    #         print(f' **************** getting folder elements ...')       
    #         print(f' **************** found model storage path! {req_model_storage}')
    #         print(f' **************** getting folder elements ...')                        
    #         for m_entry in os.listdir(req_model_storage):
    #             m_path = os.path.join(req_model_storage, m_entry)
    #             if os.path.isdir(m_path):
    #                 for item_sub in os.listdir(m_path):
    #                     sub_item_path = os.path.join(m_path, item_sub)
    #                     models_found.append(sub_item_path)        
    #         print(f' **************** found models ({len(models_found)}): {models_found}')
    #         print(f' **************** found models ({len(models_found)}): {models_found}')
    #     else:
    #         print(f' **************** found models ({len(models_found)}): {models_found}')

    # except Exception as e:
    #     print(f' **************** ERR getting models in {req_model_storage}: {e}')


    model_path = selected_model_id
    if req_model_path in models_found:
        print(f' **************** FOUND MODELS ALREADY!!! {selected_model_id} ist in {models_found}')
        model_path = req_model_path
        return f'Model already downloaded!', gr.update(visible=True), gr.update(visible=True)
    else:
        print(f' **************** NUH UH DIDNT FIND MODEL YET!! {selected_model_id} ist NAWT in {models_found}')
    
    
        
    if selected_model_architectures == '':
        return f'Selected model has no architecture', gr.update(visible=False), gr.update(visible=False)


    # if selected_model_architectures.lower() not in defaults_frontend['vllm_supported_architectures']:
    #     if selected_model_transformers != 'True':   
    #         return f'Selected model architecture is not supported by vLLM but transformers are available (you may try to load the model in gradio Interface)', gr.update(visible=True), gr.update(visible=True)
    #     else:
    #         return f'Selected model architecture is not supported by vLLM and has no transformers', gr.update(visible=False), gr.update(visible=False)     
    
    if selected_model_pipeline_tag == '':
        return f'Selected model has no pipeline tag', gr.update(visible=True), gr.update(visible=True)
            
    if selected_model_pipeline_tag not in ["text-generation","automatic-speech-recognition"]:
        return f'Only "text-generation" and "automatic-speech-recognition" models supported', gr.update(visible=False), gr.update(visible=False)
    
    if selected_model_private != 'False':        
        return f'Selected model is private', gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_gated != 'False':        
        return f'Selected model is gated', gr.update(visible=False), gr.update(visible=False)
        
    if selected_model_transformers != 'True':        
        return f'Selected model has no transformers', gr.update(visible=True), gr.update(visible=True)
        
    if selected_model_size == '0':        
        return f'Selected model has no size', gr.update(visible=False), gr.update(visible=False)


    return f'Selected model is supported by vLLM!'




def toggle_load_create(choice):
    
    if choice == 'load':
        return [gr.update(visible=True), gr.update(visible=False)]
    
    return [gr.update(visible=False), gr.update(visible=True)]



                
def toggle_audio_device(device):
    
    if device == 'cpu':
        return gr.update(choices=["int8"], value="int8")
    
    return gr.update(choices=["int8_float16", "float16"], value="float16")

                
def toggle_image_device(device):
    
    if device == 'cpu':
        return gr.update(choices=["int8"], value="int8")
    
    return gr.update(choices=["torch.float16", "torch.float32"], value="torch.float16")


                

def toggle_video(choice):
    
    if choice == 'prompt':
        return [gr.update(visible=True), gr.update(visible=False)]
    
    return [gr.update(visible=False), gr.update(visible=True)]




def get_audio_path(audio_file):
    req_file = audio_file
    return [f'req_file: {req_file}', f'{req_file}']


def get_video_image_path(image_file):
    req_file = image_file
    # return [f'req_file: {image_file}', f'{req_file}']
    return f'req_file: {image_file}'

def get_trellis_image_path(image_file):
    req_file = image_file
    return f'{image_file}'


def get_vip_image_path(image_file):
    req_file = image_file
    # return [f'req_file: {image_file}', f'{req_file}']
    return f'{image_file}'




def vip_settings_change(req_settings,req_input_path,req_vip_cv2_oil_pinsel,req_vip_cv2_oil_radius,req_vip_cv2_aquarell_pinsel,req_vip_cv2_aquarell_radius,req_vip_cv2_bleistift_buntstift,req_vip_cv2_bleistift_pinsel,req_vip_cv2_bleistift_radius,req_vip_cv2_bleistift_schatten,req_vip_cv2_point_color):
    if req_settings == "Öl":
        res_output,res_text = vip_generate_oil(req_input_path,req_vip_cv2_oil_pinsel,req_vip_cv2_oil_radius)
        return ['oil', gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f'{res_output}', f'{res_text}']
    
    if req_settings == "Aquarell":
        res_output,res_text = vip_generate_aquarell(req_input_path,req_vip_cv2_aquarell_pinsel,req_vip_cv2_aquarell_radius)
        return ['aquarell', gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), f'{res_output}', f'{res_text}']
    
    if req_settings == "Bleistift":
        res_output,res_text = vip_generate_bleistift(req_input_path,req_vip_cv2_bleistift_buntstift,req_vip_cv2_bleistift_pinsel,req_vip_cv2_bleistift_radius,req_vip_cv2_bleistift_schatten)
        return ['bleistift', gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), f'{res_output}', f'{res_text}']
    
    if req_settings == "Pointillismus":
        res_output,res_text = vip_generate_point(req_input_path,req_vip_cv2_point_color)
        return ['point', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), f'{res_output}', f'{res_text}']
    
    
    return ['oil', gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f'{res_output}', f'{res_text}']


def vip_generate(input_path,req_model,req_vers):  
    
    try:

        global BACKEND_URL
        global VIP_URL

        # print(f'[vip_generate] starting ...')
        # logging.info(f'[vip_generate] starting ...')
        # vllm_stop_response = requests.post(BACKEND_URL, json={
        #     "method":"stop",
        #     "model":"container_vllm_xoo"
        # }, timeout=60)


        # print(f'[vip_generate] vllm_stop_response: {vllm_stop_response} ...')
        # logging.info(f'[vip_generate] vllm_stop_response: {vllm_stop_response} ...')

        print(f'[vip_generate] getting status ... ')
        logging.info(f'[vip_generate] getting status ... ')
        
        response = requests.post(f'{VIP_URL}', json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[vip_generate] >> got response == 200 ... building json ... {response}')
            logging.info(f'[vip_generate] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[vip_generate] >> got res_json ... {res_json}')
            logging.info(f'[vip_generate] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[vip_generate] >> status: "ok" ... starting to generate image .... ')
                logging.info(f'[vip_generate] >> status: "ok" ... starting to generate image .... ')
      
                response = requests.post(f'{VIP_URL}', json={
                    "method": "go",
                    "req_path": input_path,
                    "image_prompt": input_path,
                    "req_ver": req_vers
                })

                print(f'[vip_generate] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[vip_generate] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
                
                if res_json["result_status"] == 200:
                    return f'/usr/src/app/video/{res_json["result_data"]}',f'/usr/src/app/video/{res_json["result_data"]}'
                else: 
                    return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'
            else:
                print('[vip_generate] ERROR IMAGE SERVER DOWN!?')
                logging.info('[vip_generate] ERROR IMAGE SERVER DOWN!?')
                return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'

    except Exception as e:
        return f'Error: {e}'



def vip_generate_oil(input_path,req_pinsel,req_radius):  
    
    try:

        global BACKEND_URL
        global VIP_URL
        global DEFAULT_IMAGE

        print(f'[vip_generate_oil] getting status ... ')
        logging.info(f'[vip_generate_oil] getting status ... ')
        
        response = requests.post(f'{VIP_URL}', json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[vip_generate_oil] >> got response == 200 ... building json ... {response}')
            logging.info(f'[vip_generate_oil] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[vip_generate_oil] >> got res_json ... {res_json}')
            logging.info(f'[vip_generate_oil] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[vip_generate_oil] >> status: "ok" ... starting to generate image .... ')
                logging.info(f'[vip_generate_oil] >> status: "ok" ... starting to generate image .... ')
      
                response = requests.post(f'{VIP_URL}', json={
                    "method": "oil",
                    "req_path": input_path,
                    "req_pinsel": req_pinsel,
                    "req_radius": req_radius
                })

                print(f'[vip_generate_oil] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[vip_generate_oil] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
                
                if res_json["result_status"] == 200:
                    return f'/usr/src/app/video/{res_json["result_data"]}',f'/usr/src/app/video/{res_json["result_data"]}'
                else: 
                    return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'
            else:
                print('[vip_generate_oil] ERROR IMAGE SERVER DOWN!?')
                logging.info('[vip_generate_oil] ERROR IMAGE SERVER DOWN!?')
                return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'

    except Exception as e:
        return f'Error: {e}'


def vip_generate_aquarell(input_path,req_pinsel,req_radius):  
    
    try:

        global BACKEND_URL
        global VIP_URL
        global DEFAULT_IMAGE

        print(f'[vip_generate_aquarell] getting status ... ')
        logging.info(f'[vip_generate_aquarell] getting status ... ')
        
        response = requests.post(f'{VIP_URL}', json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[vip_generate_aquarell] >> got response == 200 ... building json ... {response}')
            logging.info(f'[vip_generate_aquarell] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[vip_generate_aquarell] >> got res_json ... {res_json}')
            logging.info(f'[vip_generate_aquarell] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[vip_generate_aquarell] >> status: "ok" ... starting to generate image .... ')
                logging.info(f'[vip_generate_aquarell] >> status: "ok" ... starting to generate image .... ')
      
                response = requests.post(f'{VIP_URL}', json={
                    "method": "aquarell",
                    "req_path": input_path,
                    "req_pinsel": req_pinsel,
                    "req_radius": req_radius
                })

                print(f'[vip_generate_aquarell] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[vip_generate_aquarell] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
                
                if res_json["result_status"] == 200:
                    return f'/usr/src/app/video/{res_json["result_data"]}',f'/usr/src/app/video/{res_json["result_data"]}'
                else: 
                    return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'
            else:
                print('[vip_generate_aquarell] ERROR IMAGE SERVER DOWN!?')
                logging.info('[vip_generate_aquarell] ERROR IMAGE SERVER DOWN!?')
                return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'

    except Exception as e:
        return f'Error: {e}'


def vip_generate_bleistift(input_path,req_vip_cv2_bleistift_buntstift,req_vip_cv2_bleistift_pinsel,req_vip_cv2_bleistift_radius,req_vip_cv2_bleistift_schatten):  
    
    try:

        global BACKEND_URL
        global VIP_URL
        global DEFAULT_IMAGE

        print(f'[vip_generate_bleistift] getting status ... ')
        logging.info(f'[vip_generate_bleistift] getting status ... ')
        
        response = requests.post(f'{VIP_URL}', json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[vip_generate_bleistift] >> got response == 200 ... building json ... {response}')
            logging.info(f'[vip_generate_bleistift] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[vip_generate_bleistift] >> got res_json ... {res_json}')
            logging.info(f'[vip_generate_bleistift] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[vip_generate_bleistift] >> status: "ok" ... starting to generate image .... ')
                logging.info(f'[vip_generate_bleistift] >> status: "ok" ... starting to generate image .... ')
      
                response = requests.post(f'{VIP_URL}', json={
                    "method": "bleistift",
                    "req_path": input_path,
                    "req_vip_cv2_bleistift_buntstift": req_vip_cv2_bleistift_buntstift,
                    "req_vip_cv2_bleistift_pinsel": req_vip_cv2_bleistift_pinsel,
                    "req_vip_cv2_bleistift_radius": req_vip_cv2_bleistift_radius,
                    "req_vip_cv2_bleistift_schatten": req_vip_cv2_bleistift_schatten
                })

                print(f'[vip_generate_bleistift] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[vip_generate_bleistift] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
                
                if res_json["result_status"] == 200:
                    return f'/usr/src/app/video/{res_json["result_data"]}',f'/usr/src/app/video/{res_json["result_data"]}'
                else: 
                    return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'
            else:
                print('[vip_generate_bleistift] ERROR IMAGE SERVER DOWN!?')
                logging.info('[vip_generate_bleistift] ERROR IMAGE SERVER DOWN!?')
                return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'

    except Exception as e:
        return f'Error: {e}'


def vip_generate_point(input_path,req_color):  
    
    try:

        global BACKEND_URL
        global VIP_URL
        global DEFAULT_IMAGE

        print(f'[vip_generate_point] getting status ... ')
        logging.info(f'[vip_generate_point] getting status ... ')
        
        response = requests.post(f'{VIP_URL}', json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[vip_generate_point] >> got response == 200 ... building json ... {response}')
            logging.info(f'[vip_generate_point] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[vip_generate_point] >> got res_json ... {res_json}')
            logging.info(f'[vip_generate_point] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[vip_generate_point] >> status: "ok" ... starting to generate image .... ')
                logging.info(f'[vip_generate_point] >> status: "ok" ... starting to generate image .... ')
      
                response = requests.post(f'{VIP_URL}', json={
                    "method": "point",
                    "req_path": input_path,
                    "req_color": req_color
                })

                print(f'[vip_generate_point] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[vip_generate_point] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
                
                if res_json["result_status"] == 200:
                    return f'/usr/src/app/video/{res_json["result_data"]}',f'/usr/src/app/video/{res_json["result_data"]}'
                else: 
                    return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'
            else:
                print('[vip_generate_point] ERROR IMAGE SERVER DOWN!?')
                logging.info('[vip_generate_point] ERROR IMAGE SERVER DOWN!?')
                return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'

    except Exception as e:
        return f'Error: {e}'



def audio_transcribe(audio_model,audio_path,audio_device,audio_compute_type):  
    try:
        print(f'[audio_transcribe] audio_path ... {audio_path}')
        logging.info(f'[audio_transcribe] audio_path ... {audio_path}')
      
        AUDIO_URL = f'http://container_audio:{os.getenv("AUDIO_PORT")}/t'

        print(f'[audio_transcribe] AUDIO_URL ... {AUDIO_URL}')
        logging.info(f'[audio_transcribe] AUDIO_URL ... {AUDIO_URL}')

        print(f'[audio_transcribe] getting status ... ')
        logging.info(f'[audio_transcribe] getting status ... ')
        
        response = requests.post(AUDIO_URL, json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[audio_transcribe] >> got response == 200 ... building json ... {response}')
            logging.info(f'[audio_transcribe] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[audio_transcribe] >> got res_json ... {res_json}')
            logging.info(f'[audio_transcribe] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[audio_transcribe] >> status: "ok" ... starting transcribe .... ')
                logging.info(f'[audio_transcribe] >> status: "ok" ... starting transcribe .... ')
      
                response = requests.post(AUDIO_URL, json={
                    "method": "transcribe",
                    "audio_model": audio_model,
                    "audio_path": audio_path,
                    "audio_device": audio_device,
                    "audio_compute_type": audio_compute_type
                })

                print(f'[audio_transcribe] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[audio_transcribe] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
   
                print(f'[audio_transcribe] >> #22222 got res_json ... {res_json}')
                logging.info(f'[audio_transcribe] >> #22222 got res_json ... {res_json}')
                
                if res_json["result_status"] == 200:
                    return f'{res_json["result_data"]}'
                else: 
                    return 'Error :/'
            else:
                print('[audio_transcribe] ERROR AUDIO SERVER DOWN!?')
                logging.info('[audio_transcribe] ERROR AUDIO SERVER DOWN!?')
                return 'Error :/'

    except Exception as e:
        return f'Error: {e}'





def image_generate(image_model,image_prompt,image_device,image_compute_type):  
    
    try:

        global BACKEND_URL
        global IMAGE_URL

        print(f'[image_generate] starting ...')
        logging.info(f'[image_generate] starting ...')
        tr_stop_response = requests.post(BACKEND_URL, json={
            "method":"restart",
            "model":"container_tr"
        }, timeout=60)

        vllm_stop_response = requests.post(BACKEND_URL, json={
            "method":"stop",
            "model":"container_vllm_xoo"
        }, timeout=60)


        print(f'[image_generate] vllm_stop_response: {vllm_stop_response} ...')
        logging.info(f'[image_generate] vllm_stop_response: {vllm_stop_response} ...')


        print(f'[image_generate] IMAGE_URL ... {IMAGE_URL}')
        logging.info(f'[image_generate] IMAGE_URL ... {IMAGE_URL}')

        print(f'[image_generate] getting status ... ')
        logging.info(f'[image_generate] getting status ... ')
        
        response = requests.post(IMAGE_URL, json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[image_generate] >> got response == 200 ... building json ... {response}')
            logging.info(f'[image_generate] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[image_generate] >> got res_json ... {res_json}')
            logging.info(f'[image_generate] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[image_generate] >> status: "ok" ... starting to generate image .... ')
                logging.info(f'[image_generate] >> status: "ok" ... starting to generate image .... ')
      
                response = requests.post(IMAGE_URL, json={
                    "method": "generate_image",
                    "image_model": image_model,
                    "image_prompt": image_prompt,
                    "image_device": image_device,
                    "image_compute_type": image_compute_type
                })

                print(f'[image_generate] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[image_generate] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()
   
                print(f'[image_generate] >> #22222 got res_json ... {res_json}')
                logging.info(f'[image_generate] >> #22222 got res_json ... {res_json}')   
                print(f'[image_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                logging.info(f'[image_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                
                if res_json["result_status"] == 200:
                    return f'{res_json["result_data"]["output_path"]}'
                else: 
                    return f'{DEFAULT_IMAGE}'
            else:
                print('[image_generate] ERROR IMAGE SERVER DOWN!?')
                logging.info('[image_generate] ERROR IMAGE SERVER DOWN!?')
                return f'{DEFAULT_IMAGE}'

    except Exception as e:
        return f'Error: {e}'



def video_input_generate(video_input_toggle,video_input_prompt,video_input_upload,video_image_model,video_image_device,video_image_compute_type):  
    
    try:

        
        global BACKEND_URL
        global IMAGE_URL


        print(f'[video_input_generate] starting ...')
        logging.info(f'[video_input_generate] starting ...')



        if video_input_toggle == 'prompt':
            print(f"[video_input_generate] video_input_toggle == 'prompt'")

            print(f'[video_input_generate] getting status ... ')
            logging.info(f'[video_input_generate] getting status ... ')



            tr_stop_response = requests.post(BACKEND_URL, json={
                "method":"restart",
                "model":"container_tr"
            }, timeout=60)
            vllm_stop_response = requests.post(BACKEND_URL, json={
                "method":"stop",
                "model":"container_vllm_xoo"
            }, timeout=60)


            print(f'[video_input_generate] vllm_stop_response: {vllm_stop_response} ...')
            logging.info(f'[video_input_generate] vllm_stop_response: {vllm_stop_response} ...')

            print(f'[video_input_generate] IMAGE_URL ... {IMAGE_URL}')
            logging.info(f'[video_input_generate] IMAGE_URL ... {IMAGE_URL}')

            print(f'[video_input_generate] getting status ... ')
            logging.info(f'[video_input_generate] getting status ... ')
            
            response = requests.post(IMAGE_URL, json={
                "method": "status"
            }, timeout=600)

            if response.status_code == 200:          
                print(f'[video_input_generate] >> got response == 200 ... building json ... {response}')
                logging.info(f'[video_input_generate] >> got response == 200 ... building json ... {response}')
                res_json = response.json()    
                print(f'[video_input_generate] >> got res_json ... {res_json}')
                logging.info(f'[video_input_generate] >> got res_json ... {res_json}')

                if res_json["result_data"] == "ok":
                    print(f'[video_input_generate] >> status: "ok" ... starting to generate image .... ')
                    logging.info(f'[video_input_generate] >> status: "ok" ... starting to generate image .... ')
        
                    response = requests.post(IMAGE_URL, json={
                        "method": "generate_image",
                        "image_model": video_image_model,
                        "image_prompt": video_input_prompt,
                        "image_device": video_image_device,
                        "image_compute_type": video_image_compute_type
                    })

                    print(f'[video_input_generate] >> got response #22222 == 200 ... building json ... {response}')
                    logging.info(f'[video_input_generate] >> got response #22222 == 200 ... building json ... {response}')
                    
                    res_json = response.json()
    
                    print(f'[video_input_generate] >> #22222 got res_json ... {res_json}')
                    logging.info(f'[video_input_generate] >> #22222 got res_json ... {res_json}')   
                    print(f'[video_input_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                    logging.info(f'[video_input_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                    
                    if res_json["result_status"] == 200:
                        print(f'[video_input_generate] == == 200')
                        logging.info(f'[video_input_generate] == == 200')
                        return f'{res_json["result_data"]["output_path"]}',f'{res_json["result_data"]["output_path"]}'
                    else: 
                        print(f'[video_input_generate] == == 333')
                        logging.info(f'[video_input_generate] == == 333')
                        return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'
                else:
                    print(f'[video_input_generate] == == 444')
                    logging.info(f'[video_input_generate] == == 444')
                    print('[video_input_generate] ERROR IMAGE SERVER DOWN!?')
                    logging.info('[video_input_generate] ERROR IMAGE SERVER DOWN!?')
                    return f'{DEFAULT_IMAGE}',f'{DEFAULT_IMAGE}'




        if video_input_toggle == 'upload':
            print(f"[video_generate] video_input_toggle == 'upload'")
            return f'{video_input_upload}',f'{video_input_upload}'



    except Exception as e:
        return f'Error: {e}',f'Error: {e}'




def video_generate(video_image,video_input_prompt,video_model,video_device,video_compute_type,video_variant,video_decode_chunk_size,video_motion_bucket_id,video_noise_aug_strength,video_fps):  
    
    global BACKEND_URL
    global VIDEO_URL
    try:
        print(f'[video_generate] starting ...')
        logging.info(f'[video_generate] starting ...')


        print(f'[video_generate] video_image: {video_image} ...')
        logging.info(f'[video_generate] video_image: {video_image} ...')


        
        print(f'[video_generate] video_input_prompt: {video_input_prompt} ...')
        logging.info(f'[video_generate] video_input_prompt: {video_input_prompt} ...')

        video_input_prompt_clean_to_filename = re.sub(r'[^\w\s-]', '', video_input_prompt)
        video_input_prompt_clean_to_filename = re.sub(r'[-\s]+', '_', video_input_prompt_clean_to_filename)
        video_input_prompt_clean_to_filename = video_input_prompt_clean_to_filename[:50]
        video_input_prompt_clean_to_filename = video_input_prompt_clean_to_filename.strip('_-')
        video_input_prompt_clean_to_filename = f"{video_input_prompt_clean_to_filename}.mp4"

        print(f'[video_generate] video_input_prompt_clean_to_filename: {video_input_prompt_clean_to_filename} ...')
        logging.info(f'[video_generate] video_input_prompt_clean_to_filename: {video_input_prompt_clean_to_filename} ...')

        tr_stop_response = requests.post(BACKEND_URL, json={
            "method":"restart",
            "model":"container_tr"
        }, timeout=60)
        vllm_stop_response = requests.post(BACKEND_URL, json={
            "method":"stop",
            "model":"container_vllm_xoo"
        }, timeout=60)


        print(f'[video_generate] vllm_stop_response: {vllm_stop_response} ...')
        logging.info(f'[video_generate] vllm_stop_response: {vllm_stop_response} ...')
    
        

        print(f'[video_generate] restarting container_tr ...')
        logging.info(f'[video_generate] restarting container_tr...')
        container_tr_restart_response = requests.post(BACKEND_URL, json={
            "method":"restart",
            "model":"container_tr"
        }, timeout=60)
        print(f'[video_generate] >> container_tr_restart_response {container_tr_restart_response}')
        logging.info(f'[video_generate] >> container_tr_restart_response {container_tr_restart_response}')





        print(f'[video_generate] VIDEO_URL ... {VIDEO_URL}')
        logging.info(f'[video_generate] VIDEO_URL ... {VIDEO_URL}')

        print(f'[video_generate] getting status ... ')
        logging.info(f'[video_generate] getting status ... ')
        
        response = requests.post(VIDEO_URL, json={
            "method": "status"
        }, timeout=600)

        if response.status_code == 200:          
            print(f'[video_generate] >> got response == 200 ... building json ... {response}')
            logging.info(f'[video_generate] >> got response == 200 ... building json ... {response}')
            res_json = response.json()    
            print(f'[video_generate] >> got res_json ... {res_json}')
            logging.info(f'[video_generate] >> got res_json ... {res_json}')

            if res_json["result_data"] == "ok":
                print(f'[video_generate] >> status: "ok" ... starting to generate video .... ')
                logging.info(f'[video_generate] >> status: "ok" ... starting to generate video .... ')     

                response = requests.post(VIDEO_URL, json={
                    "method": "generate_video",
                    "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
                    "input_image_path": video_image,  # Path to your input image
                    "device": "cuda",  # or "cpu"
                    "torch_dtype": "torch.float16",  # or "torch.float32"
                    "variant": "fp16",
                    "decode_chunk_size": 8,  # Reduce for lower VRAM usage
                    "motion_bucket_id": 180,  # Higher = more motion
                    "noise_aug_strength": 0.1,
                    "output_path": video_input_prompt_clean_to_filename,  # Optional
                    "fps": 10  # Optional, frames per second
                })





                print(f'[video_generate] >> got response #22222 == 200 ... building json ... {response}')
                logging.info(f'[video_generate] >> got response #22222 == 200 ... building json ... {response}')
                
                res_json = response.json()

                print(f'[video_generate] >> #22222 got res_json ... {res_json}')
                logging.info(f'[video_generate] >> #22222 got res_json ... {res_json}')   
                print(f'[video_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                logging.info(f'[video_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                
                if res_json["result_status"] == 200:
                    print(f'[video_generate] >> result_status == 200 ... {res_json["result_status"]}')
                    logging.info(f'[video_generate] >> result_status == 200 ... {res_json["result_status"]}') 
                    return f'{res_json["result_data"]["output_path"]}',f'{res_json["result_data"]["output_path"]}'
                else: 
                    print(f'[video_generate] >> result_status !=!= 200 ... {res_json["result_status"]}')
                    logging.info(f'[video_generate] >> result_status !=!= 200 ... {res_json["result_status"]}') 
                    return f'{DEFAULT_VIDEO}',f'{DEFAULT_VIDEO}'
            else:
                print('[video_generate] ERROR IMAGE SERVER DOWN no result_status !?')
                logging.info('[video_generate] ERROR IMAGE SERVER DOWN!? no result_status')
                return f'{DEFAULT_VIDEO}',f'{DEFAULT_VIDEO}'

    except Exception as e:
        return f'Error: {e}',f'Error: {e}'






def trellis_generate(image_prompt):  
    
    try:

        global BACKEND_URL
        global TR_URL

        print(f'[trellis_generate] starting ... image_prompt: {image_prompt}')
        logging.info(f'[trellis_generate] starting ... image_prompt: {image_prompt}')
        vllm_stop_response = requests.post(BACKEND_URL, json={
            "method":"restart",
            "model":"container_vllm_xoo"
        }, timeout=60)


        print(f'[trellis_generate] vllm_stop_response: {vllm_stop_response} ...')
        logging.info(f'[trellis_generate] vllm_stop_response: {vllm_stop_response} ...')

        print(f'[trellis_generate] >> got response != 200 ... stopping video ...')
        logging.info(f'[trellis_generate] >> got response != 200 ... stopping video ...')
        container_video_stop_response = requests.post(BACKEND_URL, json={
            "method":"restart",
            "model":"container_video"
        }, timeout=60)
        print(f'[trellis_generate] >> container_video_stop_response: {container_video_stop_response}')
        logging.info(f'[trellis_generate] >> container_video_stop_response: {container_video_stop_response}')


        print(f'[trellis_generate] IMAGE_URL ... {IMAGE_URL}')
        logging.info(f'[trellis_generate] IMAGE_URL ... {IMAGE_URL}')

        # print(f'[trellis_generate] getting status ... ')
        # logging.info(f'[trellis_generate] getting status ... ')
        

        print(f'[trellis_generate] >> status: "ok" ... starting to generate 3D .... ')
        logging.info(f'[trellis_generate] >> status: "ok" ... starting to generate 3D .... ')

        response = requests.post(f'{TR_URL}/t2', json={
            "input_image_path": f'{image_prompt}'
        })

        print(f'[trellis_generate] >> got response #22222 == 200 ... building json ... {response}')
        logging.info(f'[trellis_generate] >> got response #22222 == 200 ... building json ... {response}')
        
        res_json = response.json()

        print(f'[trellis_generate] >> #22222 got res_json ... {res_json}')
        logging.info(f'[trellis_generate] >> #22222 got res_json ... {res_json}')   
        print(f'[trellis_generate] >> #22222 got res_json["result_data"] ... {res_json["result_data"]}')
        logging.info(f'[trellis_generate] >> #22222 got res_json["result_data"] ... {res_json["result_data"]}')
        print(f'[trellis_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
        logging.info(f'[trellis_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
        
        return f'{res_json["result_data"]["output_path"]}',f'{res_json["result_data"]["output_path"]}'

    except Exception as e:
        return f'Error: {e}',f'Error: {e}'


























def network_to_pd():       
    rows = []
    try:
        network_list = get_network_data()
        # logging.info(f'[network_to_pd] network_list: {network_list}')  # Use logging.info instead of logging.exception
        for entry in network_list:

            rows.append({
                "container": entry["container"],
                "current_dl": entry["current_dl"]
            })
            
            
        df = pd.DataFrame(rows)
        return df,rows[0]["current_dl"]
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "container": "0",
                "current_dl": f'0',
                "timestamp": f'0',
                "info": f'0'
        })
        df = pd.DataFrame(rows)
        return df












async def nur_update(**kwargs):
    try:
        # print(f' **nur_update: kwargs["db_name"] {kwargs["db_name"]}')
        res_db_list = await r.lrange(kwargs["db_name"], 0, -1)

        # print(f' **nur_update: found {len(res_db_list)} entries!')
        res_db_list = [json.loads(entry) for entry in res_db_list]
        # print(f' **nur_update: res_db_list {res_db_list}')
        
        if kwargs["method"] == "update":
            if len(res_db_list) > 0:
                update_i = 0
                for entry in [json.dumps(entry) for entry in res_db_list]:
                    await r.lrem(kwargs["db_name"], 0, entry)
                    entry = json.loads(entry)
                    entry["gpu"]["mem"] = f'blablabla + {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    await r.rpush(kwargs["db_name"], json.dumps(entry))
                    update_i = update_i + 1
                return res_db_list
            else:
                print(f' **REDIS: Error: no entry to update for db_name: {kwargs["db_name"]}')
                return False
        
        return False
    
    except Exception as e:
        print(f' **REDIS: Error: {e}')
        return False

REDIS_DB_VLLM = "db_test28"
test_call_update = {
    "db_name": REDIS_DB_VLLM,
    "method": "update",
    "select": "all",
    "filter_key": "id",
    "filter_val": "3",
}

async def update_vllms_list():
    res_redis = await nur_update(**test_call_update)
    return res_redis

req_db = "db_test28"

test_call_get = {
    "db_name": req_db,
    "method": "get",
    "select": "all"
}



async def selected_xoo_fish_info(selected_radio):

    global GLOBAL_FISH
    global SELECTED_XOO_ID
    global SELECTED_XOO_OBJ

    print(f'~~~~~~ [selected_fish_info] GLOBAL_FISH: {GLOBAL_FISH}')
    logging.info(f'~~~~~~ [selected_fish_info] GLOBAL_FISH: {GLOBAL_FISH}')
    print(f'~~~~~~ [selected_fish_info] got selected_radio: {selected_radio}')
    logging.info(f'~~~~~~ [selected_fish_info] got selected_radio: {selected_radio}')
    print(f'~~~~~~ searching for fish ...')
    logging.info(f'~~~~~~ searching for fish ...')
    
    found_fish = [fish for fish in GLOBAL_FISH if fish["container_name"] == selected_radio]
    print(f'~~~~~~ found_fish: {found_fish}')
    logging.info(f'~~~~~~ found_fish: {found_fish}')
    
    found_fish_name = found_fish[0]["name"]
    found_fish_name_split = found_fish_name.split('/')[1]
    print(f'~~~~~~ found_fish_name_split: {found_fish_name_split}')
    logging.info(f'~~~~~~ found_fish_name_split: {found_fish_name_split}')
    
        
    found_fish_port = found_fish[0]["port"]
    print(f'~~~~~~ found_fish_port: {found_fish_port}')
    logging.info(f'~~~~~~ found_fish_port: {found_fish_port}')
    
    
    found_fish_image = found_fish[0]["image_vllm"]
    print(f'~~~~~~ found_fish_image: {found_fish_image}')
    logging.info(f'~~~~~~ found_fish_image: {found_fish_image}')
    
    found_fish_max_model_len = found_fish[0]["max_model_len"]
    print(f'~~~~~~ found_fish_max_model_len: {found_fish_max_model_len}')
    logging.info(f'~~~~~~ found_fish_max_model_len: {found_fish_max_model_len}')
    
    found_fish_tensor_parallel_size = len(found_fish[0]["gpu_list"])
    print(f'~~~~~~ found_fish_tensor_parallel_size: {found_fish_tensor_parallel_size}')
    logging.info(f'~~~~~~ found_fish_tensor_parallel_size: {found_fish_tensor_parallel_size}')

    SELECTED_XOO_ID = selected_radio
    SELECTED_XOO_OBJ = found_fish
    return f'{found_fish_name_split}', int(found_fish_port), f'{found_fish_image}', f'{found_fish_max_model_len}', f'{found_fish_tensor_parallel_size}', f'{found_fish}', f'{selected_radio}'


async def selected_fish_info(selected_radio):

    global GLOBAL_FISH
    global SELECTED_XOO_ID
    global SELECTED_XOO_OBJ

    print(f'~~~~~~ [selected_fish_info] GLOBAL_FISH: {GLOBAL_FISH}')
    logging.info(f'~~~~~~ [selected_fish_info] GLOBAL_FISH: {GLOBAL_FISH}')
    print(f'~~~~~~ [selected_fish_info] got selected_radio: {selected_radio}')
    logging.info(f'~~~~~~ [selected_fish_info] got selected_radio: {selected_radio}')
    print(f'~~~~~~ searching for fish ...')
    logging.info(f'~~~~~~ searching for fish ...')
    found_fish = []
    found_fish += [fish for fish in GLOBAL_FISH if fish["container_name"] == selected_radio]
    print(f'~~~~~~ found_fish: {found_fish}')
    logging.info(f'~~~~~~ found_fish: {found_fish}')

    print(f'~~~~~~ len(found_fish): {len(found_fish)}')
    logging.info(f'~~~~~~ len(found_fish): {len(found_fish)}')

    SELECTED_XOO_ID = selected_radio
    SELECTED_XOO_OBJ = found_fish
    return f'{found_fish}', f'{selected_radio}'

async def selected_vllm_info(selected_radio):

    global GLOBAL_VLLMS
    # print(f'~~~~~~ [selected_vllm_info] REDIS_DB_VLLM: {REDIS_DB_VLLM}')

    # print(f'~~~~~~ [selected_vllm_info] GLOBAL_VLLMS: {GLOBAL_VLLMS}')
    # print(f'~~~~~~ [selected_vllm_info] got selected_radio: {selected_radio}')
    
    # print(f'~~~~~~ searching for vllm ...')

    found_vllm = [vllm for vllm in GLOBAL_VLLMS if vllm["container_name"] == selected_radio]
    print(f'~~~~~~ found_vllm: {found_vllm}')
    return f'{found_vllm}', f'{selected_radio}'
    # req_vllm = {
    #     "db_name": REDIS_DB_VLLM,
    #     "method": "get",
    #     "select": "filter",
    #     "filter_key": "container_name",
    #     "filter_val": selected_radio,
    # }
    
    # res_vllm = await redis_connection(**req_vllm)
    # print(f'~~~~~~ [selected_vllm_info] got res_vllm: {res_vllm}')
    # print(f'~~~~~~ [selected_vllm_info] got res_vllm[0]: {res_vllm[0]}')
    
    # return f'{res_vllm}', f'{selected_radio}'



default_vllm = {
    "container_name": "vllm1",
    "uid": "123123",
    "created_hr": "123123",
    "created_ts": "123123",
    "created_by": "userasdf",
    "deleted_by": "userasdf",
    "used_by": ["userpublicasduzg","userasdf"],
    "expires": "created_ts+100000sec",
    "access": "public",
    
    "status": "running",

    "State": {
                "Status": "running"
            },
    "usage": {
                "tokens": ["111111","222222"],
                "prompts": ["NVIDIA RTX 3060","NVIDIA H100"],
                "prompts_response_time": [0.01,0.2,0.4,1.2],
                "prompts_response_time_per_token": [0.01,0.2,0.4,1.2],
                "gpu_util_per_sec_running": "gpu_util_per_sec_running",
                "mem_util_per_sec_running": "mem_util_per_sec_running"
            },
    "gpu": {
                "gpu_uuids": ["111111","222222"],
                "gpu_names": ["NVIDIA RTX 3060","NVIDIA H100"],
                "mem_util": "bissplitslash",
                "gpu_util": "bissplitslash",
                "temperature": "bissplitslash",
                "power_usage": "bissplitslash",
                "cuda_cores": "bissplitslash",
                "compute_capability": "bissplitslash",
                "status": "bissplitslash"
            },
    "model": {
                "id": "creater/name",
                "creater": "bissplitslash",
                "name": "nachsplitslash",
                "size": "nachsplitslash",
                "downloaded_hr": "bissplitslash",
                "downloaded_ts": "bissplitslash",
                "model_info": "bissplitslash",
                "model_configs": "bissplitslash"
            },
    "load_settings": {
                "load_component": "load_component"
            },
    "create_settings": {
                "create_component": "create_component"
            },
    "prompt_settings": {
                "prompt_component": "prompt_component"
            },
    "docker": {
                "container_id": "container_id",
                "container_name": "container_name",
                "container_status": "container_status",
                "container_info": "bissplitslash"
            },
    "ts": "0"
}

                # "name": gpu_info.get("name", "0"),
                # "mem_util": gpu_info.get("mem_util", "0"),
                # "timestamp": entry.get("timestamp", "0"),
                # "fan_speed": gpu_info.get("fan_speed", "0"),
                # "temperature": gpu_info.get("temperature", "0"),
                # "gpu_util": gpu_info.get("gpu_util", "0"),
                # "power_usage": gpu_info.get("power_usage", "0"),
                # "clock_info_graphics": gpu_info.get("clock_info_graphics", "0"),
                # "clock_info_mem": gpu_info.get("clock_info_mem", "0"),                
                # "cuda_cores": gpu_info.get("cuda_cores", "0"),
                # "compute_capability": gpu_info.get("compute_capability", "0"),
                # "current_uuid": gpu_info.get("current_uuid", "0"),
                # "gpu_i": entry.get("gpu_i", "0"),
                # "supported": gpu_info.get("supported", "0"),
                # "not_supported": gpu_info.get("not_supported", "0"),
                # "status": "ok"

                        # container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                        
                        # container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
            
                        # container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")






def toggle_test_vllms_create(vllm_list):
    # print(f'got vllm_list: {vllm_list}')
    if "Create New" in vllm_list:
        return (
            gr.Accordion(open=False,visible=False),
            gr.Button(visible=False),
            gr.Accordion(open=True,visible=True),
            gr.Button(visible=True)
        )

    return (
        gr.Accordion(open=True,visible=True),
        gr.Button(visible=True),    
        gr.Accordion(open=False,visible=False),
        gr.Button(visible=False)
    )














def get_time():
    try:
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllms] {e}')
        return f'err {str(e)}'






def refresh_container():
    try:
        global docker_container_list
        response = requests.post(BACKEND_URL, json={"method": "list"})
        docker_container_list = response.json()
        return docker_container_list
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'err {str(e)}'

            
@dataclass
class VllmCreateComponents:
    method: gr.Textbox
    container_name: gr.Textbox
    image: gr.Textbox
    runtime: gr.Textbox
    shm_size: gr.Slider
    port: gr.Slider
    max_model_len: gr.Slider
    tensor_parallel_size: gr.Number
    gpu_memory_utilization: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class VllmCreateValues:
    method: str
    container_name: str
    image: str
    runtime: str
    shm_size: int
    port: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: int



@dataclass
class VllmLoadComponents:
    method: gr.Textbox
    vllmcontainer: gr.Textbox
    port: gr.Slider
    image: gr.Textbox
    max_model_len: gr.Slider
    tensor_parallel_size: gr.Number
    gpu_memory_utilization: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class VllmLoadValues:
    method: str
    vllmcontainer: str
    port: int
    image: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: int



@dataclass
class PromptComponents:
    vllmcontainer: gr.Radio
    port: gr.Slider
    prompt: gr.Textbox
    top_p: gr.Slider
    temperature: gr.Slider
    max_tokens: gr.Slider
    
    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]

@dataclass
class PromptValues:
    vllmcontainer: str
    port: int
    prompt: str
    top_p: int
    temperature: int
    max_tokens: int



def toggle_vllm_load_create(vllm_list):
    
    if "Create New" in vllm_list:
        return (
            gr.Accordion(open=False,visible=False),
            gr.Button(visible=False),
            gr.Accordion(open=True,visible=True),
            gr.Button(visible=True)
        )

    return (
        gr.Accordion(open=True,visible=True),
        gr.Button(visible=True),    
        gr.Accordion(open=False,visible=False),
        gr.Button(visible=False)
    )

# def toggle_vllm_prompt(vllm_list_prompt):
#     global DEFAULT_PROMPT
#     PROMPT = 'asdasdasdasdasdasdasd'
#     return gr.Textbox(value="9999")   
    # if "Create New" in vllm_list_prompt:
    #     return (
    #         gr.Accordion(open=False,visible=False),
    #         gr.Button(visible=False),
    #         gr.Accordion(open=True,visible=True),
    #         gr.Button(visible=True)
    #     )

    # return (
    #     gr.Accordion(open=True,visible=True),
    #     gr.Button(visible=True),    
    #     gr.Accordion(open=False,visible=False),
    #     gr.Button(visible=False)
    # )

   
# asdasd
def llm_load(*params):
    
    try:
        global SELECTED_MODEL_ID
        global SELECTED_XOO_ID
        global SELECTED_XOO_OBJ
        print(f' >>> llm_load SELECTED_XOO_ID: {SELECTED_XOO_ID} ')
        print(f' >>> llm_load SELECTED_XOO_OBJ: {SELECTED_XOO_OBJ} ')
        print(f' >>> llm_load SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')    
        print(f' >>> llm_load got params: {params} ')
        
        
        
        logging.info(f'[llm_load] >> SELECTED_XOO_ID: {SELECTED_XOO_ID} ')
        logging.info(f'[llm_load] >> SELECTED_XOO_OBJ: {SELECTED_XOO_OBJ} ')
        logging.info(f'[llm_load] >> SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        logging.info(f'[llm_load] >> got params: {params} ')        
        
        
        req_params = VllmLoadComponents(*params)

        response = requests.post(BACKEND_URL, json={
            "method":req_params.method,
            "vllmcontainer":req_params.vllmcontainer,
            "image":req_params.image,
            "port":req_params.port,
            "model":SELECTED_MODEL_ID,
            "tensor_parallel_size":req_params.tensor_parallel_size,
            "gpu_memory_utilization":req_params.gpu_memory_utilization,
            "max_model_len":req_params.max_model_len
        }, timeout=REQUEST_TIMEOUT)


        if response.status_code == 200:
            print(f' [llm_load] >> got response == 200 building json ... {response} ')
            logging.exception(f'[llm_load] >> got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' [llm_load] >> GOT RES_JSON: SELECTED_MODEL_ID: {res_json} ')         
            return f'{res_json}'
        else:
            logging.exception(f'[llm_load] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
        
    
def llm_create(*params):
    
    try:
        global SELECTED_MODEL_ID
        print(f' >>> llm_create SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        print(f' >>> llm_create got params: {params} ')
        logging.exception(f'[llm_create] >> SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        logging.exception(f'[llm_create] >> got params: {params} ')
                
        req_params = VllmCreateComponents(*params)

        response = requests.post(BACKEND_URL, json={
            "method":req_params.method,
            "container_name":req_params.container_name,
            "image":req_params.image,
            "runtime":req_params.runtime,
            "shm_size":f'{str(req_params.shm_size)}gb',
            "port":req_params.port,
            "model":SELECTED_MODEL_ID,
            "tensor_parallel_size":req_params.tensor_parallel_size,
            "gpu_memory_utilization":req_params.gpu_memory_utilization,
            "max_model_len":req_params.max_model_len
        }, timeout=REQUEST_TIMEOUT)
        print(f'[llm_create] >> 2222 ')
        logging.exception(f'[llm_create] >> 2222 ')

        if response.status_code == 200:
            print(f' [llm_create] >> got response == 200 building json ... {response} ')
            logging.exception(f'[llm_create] >> got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' [llm_create] >> GOT RES_JSON: SELECTED_MODEL_ID: {res_json} ')         
            return f'{res_json}'
        else:
            logging.exception(f'[llm_create] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    
        
def llm_prompt(*params):
    
    try:


        global BACKEND_URL
        global SELECTED_MODEL_ID
        global VLLM_URL
        print(f' >>> llm_prompt SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        print(f' >>> llm_prompt got params: {params} ')
        logging.info(f'[llm_prompt] >> SELECTED_MODEL_ID: {SELECTED_MODEL_ID} ')
        logging.info(f'[llm_prompt] >> got params: {params} ')

        response = None
        try:
            response = requests.post(VLLM_URL, json={
                "method": "status"
            }, timeout=600)
        except requests.RequestException as e:
            print(f'[llm_prompt] >> request to VLLM_URL failed: {e} ')
            logging.info(f'[llm_prompt] >> request to VLLM_URL failed: {e}')

            container_vllm_xoo_restart_response = requests.post(BACKEND_URL, json={
                "method":"restart",
                "model":"container_vllm_xoo"
            }, timeout=600)
            print(f'[llm_prompt] >> container_vllm_xoo_restart_response: {container_vllm_xoo_restart_response}')
            logging.info(f'[llm_prompt] >> container_vllm_xoo_restart_response: {container_vllm_xoo_restart_response}')

            time.sleep(20)



        if response and response.status_code == 200:       
            print(f'[llm_prompt] >> got response == 200 ... {response}')
            logging.info(f'[llm_prompt] >> got response == 200 ... {response}')
        else:
            tr_stop_response = requests.post(BACKEND_URL, json={
                "method":"restart",
                "model":"container_tr"
            }, timeout=60)
            print(f'[llm_prompt] >> got response != 200 ... stopping video ...')
            logging.info(f'[llm_prompt] >> got response != 200 ... stopping video ...')
            container_video_stop_response = requests.post(BACKEND_URL, json={
                "method":"restart",
                "model":"container_video"
            }, timeout=600)
            print(f'[llm_prompt] >> container_video_stop_response: {container_video_stop_response}')
            logging.info(f'[llm_prompt] >> container_video_stop_response: {container_video_stop_response}')


            response = requests.post(VLLM_URL, json={
                "method": "status"
            }, timeout=600)

            print(f'[llm_prompt] vllm status response: {response} ...')
            logging.info(f'[llm_prompt] vllm status response: {response} ...')


            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ LOAD MODEL VLLM XOO ...')
            vllm_start_response = requests.post(BACKEND_URL, json={
                "method":"load",
                "vllmcontainer":"container_vllm_xoo",
                "image":"xoo4foo/zzvllm52:latest",
                "port":1370,
                "model":"Qwen/Qwen2.5-1.5B-Instruct",
                "tensor_parallel_size":1,
                "gpu_memory_utilization":0.87,
                "max_model_len":4096
            }, timeout=60)

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ...')


        req_params = PromptComponents(*params)

        DEFAULTS_PROMPT = {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "vllmcontainer": "container_vllm_xoo",
            "port": 1370,
            "prompt": "Tell a joke",
            "top_p": 0.95,
            "temperature": 0.8,
            "max_tokens": 150
        }

        response = requests.post(BACKEND_URL, json={
            "method":"generate",
            "model":SELECTED_MODEL_ID,
            "vllmcontainer":getattr(req_params, "vllmcontainer", DEFAULTS_PROMPT["vllmcontainer"]),
            "port":getattr(req_params, "port", DEFAULTS_PROMPT["port"]),
            "prompt": getattr(req_params, "prompt", DEFAULTS_PROMPT["prompt"]),
            "top_p":getattr(req_params, "top_p", DEFAULTS_PROMPT["top_p"]),
            "temperature":getattr(req_params, "temperature", DEFAULTS_PROMPT["temperature"]),
            "max_tokens":getattr(req_params, "max_tokens", DEFAULTS_PROMPT["max_tokens"])
        }, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            print(f' !?!?!?!? [llm_prompt] got response == 200 building json ... {response} ')
            logging.info(f'!?!?!?!? [llm_prompt] got response == 200 building json ...  {response} ')
            res_json = response.json()        
            print(f' !?!?!?!? [llm_prompt] GOT RES_JSON: llm_prompt SELECTED_MODEL_ID: {res_json} ')
            logging.info(f'!?!?!?!? [llm_prompt] GOT RES_JSON: {res_json} ')
            # to fix 200 to 400 in xoo image
            if res_json["result_status"] != 200:
                print(f' !?!?!?!? [llm_prompt] res_json["result_status"] != 200: {res_json} ')
                logging.exception(f'[llm_prompt] Response Error: {res_json["result_data"]}')
                return f'{res_json}'
            if 'No model loaded' in res_json["result_data"]:
                print(f' !?!?!?!? [llm_prompt] no model found! reloading ... ')
                logging.info(f'!?!?!?!? no model found! reloading ... ')
                vllm_start_response = requests.post(BACKEND_URL, json={
                    "method":"load",
                    "vllmcontainer":"container_vllm_xoo",
                    "image":"xoo4foo/zzvllm52:latest",
                    "port":1370,
                    "model":"Qwen/Qwen2.5-1.5B-Instruct",
                    "tensor_parallel_size":1,
                    "gpu_memory_utilization":0.7,
                    "max_model_len":4096
                }, timeout=60)

                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ... trying to prompt again in 30 sec ..')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ... trying to prompt again in 30 sec ..')
                # to fix with VRAM usage von ubuntu cmds..
                if wait_for_vllm():
                    response = requests.post(BACKEND_URL, json={
                        "method":"generate",
                        "model":SELECTED_MODEL_ID,
                        "vllmcontainer":getattr(req_params, "vllmcontainer", DEFAULTS_PROMPT["vllmcontainer"]),
                        "port":getattr(req_params, "port", DEFAULTS_PROMPT["port"]),
                        "prompt": getattr(req_params, "prompt", DEFAULTS_PROMPT["prompt"]),
                        "top_p":getattr(req_params, "top_p", DEFAULTS_PROMPT["top_p"]),
                        "temperature":getattr(req_params, "temperature", DEFAULTS_PROMPT["temperature"]),
                        "max_tokens":getattr(req_params, "max_tokens", DEFAULTS_PROMPT["max_tokens"])
                    }, timeout=REQUEST_TIMEOUT)
                
            return f'{res_json["result_data"]}'
        else:
            logging.exception(f'[llm_prompt] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'
    



def download_from_hf_hub(selected_model_id):
    try:
        selected_model_id_arr = str(selected_model_id).split('/')
        print(f'selected_model_id_arr {selected_model_id_arr}...')       
        model_path = snapshot_download(
            repo_id=selected_model_id,
            local_dir=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'
        )
        return f'Saved to {model_path}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'


download_info_prev_bytes_recv = 0   
download_info_current_model_bytes_recv = 0    
 
def download_info(req_model_size, progress=gr.Progress()):
    global download_info_prev_bytes_recv
    global download_info_current_model_bytes_recv
    download_info_prev_bytes_recv = 0
    download_info_current_model_bytes_recv = 0
    progress(0, desc="Initializing ...")
    progress(0.01, desc="Calculating Download Time ...")
    
    avg_dl_speed_val = 0
    avg_dl_speed = []
    for i in range(0,5):
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2) 
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv
        avg_dl_speed.append(download_speed)
        avg_dl_speed_val = sum(avg_dl_speed)/len(avg_dl_speed)
        logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
        print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  
        time.sleep(1)
    
    logging.info(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] append: {download_speed} append avg_dl_speed: {avg_dl_speed} len(avg_dl_speed): {len(avg_dl_speed)} avg_dl_speed_val: {avg_dl_speed_val}')  



    calc_mean = lambda data: np.mean([x for x in data if (np.percentile(data, 25) - 1.5 * (np.percentile(data, 75) - np.percentile(data, 25))) <= x <= (np.percentile(data, 75) + 1.5 * (np.percentile(data, 75) - np.percentile(data, 25)))]) if data else 0


    avg_dl_speed_val = calc_mean(avg_dl_speed)
        
    
    logging.info(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')
    print(f' **************** [download_info] avg_dl_speed_val: {avg_dl_speed_val}')    

    est_download_time_sec = int(req_model_size)/int(avg_dl_speed_val)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')

    est_download_time_sec = int(est_download_time_sec)
    logging.info(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')
    print(f' **************** [download_info] calculating seconds ... {req_model_size}/{avg_dl_speed_val} -> {est_download_time_sec}')

    logging.info(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec}')
    print(f' **************** [download_info] zzz waiting for download_complete_event zzz waiting {est_download_time_sec}')
    current_dl_arr = []
    for i in range(0,est_download_time_sec):
        if len(current_dl_arr) > 5:
            current_dl_arr = []
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - download_info_prev_bytes_recv
        current_dl_arr.append(download_speed)
        print(f' &&&&&&&&&&&&&& current_dl_arr: {current_dl_arr}')
        if all(value < 10000 for value in current_dl_arr[-4:]):
            print(f' &&&&&&&&&&&&&& DOWNLOAD FINISH EHH??: {current_dl_arr}')
            yield f'Progress: 100%\nFiniiiiiiiish!'
            return
            
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)
        
        download_info_prev_bytes_recv = int(bytes_recv)
        download_info_current_model_bytes_recv = download_info_current_model_bytes_recv + download_info_prev_bytes_recv

        progress_percent = (i + 1) / est_download_time_sec
        progress(progress_percent, desc=f"Downloading ... {download_speed_mbit_s:.2f} MBit/s")

        time.sleep(1)
    logging.info(f' **************** [download_info] LOOP DONE!')
    print(f' **************** [download_info] LOOP DONE!')
    yield f'Progress: 100%\nFiniiiiiiiish!'


def parallel_download(selected_model_size, model_dropdown):
    # Create threads for both functions
    thread_info = threading.Thread(target=download_info, args=(selected_model_size,))
    thread_hub = threading.Thread(target=download_from_hf_hub, args=(model_dropdown,))

    # Start both threads
    thread_info.start()
    thread_hub.start()

    # Wait for both threads to finish
    thread_info.join()
    thread_hub.join()

    return "Download finished!"




def change_tab(n):
    return gr.Tabs(selected=n)












custom_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <title>NAI9 - UNLIMITED</title>		
    <meta name="description" content="UNLIMITED AI! No Sign-Up. No subscription.">
    <meta name="keywords" content="ai video, vllm, ai 3d, diffuser, unlimited, ai chat, no sign-up, free ai">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <!-- Add multiple icon formats for better browser support -->
    <link rel="shortcut icon" type="image/x-icon" href="/favicon.ico">
    <link rel="icon" type="image/png" href="/favicon.png">

    <style>
        .gradio-container {
            background: #2B6CB0;
            padding-bottom: 0px !important;
        }
        footer {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
        }
        .mobile-warning { 
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
        }
        .alert, 
        .gradio-alert, 
        [class*="alert"], 
        [data-testid="alert"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
        }
        .svelte-1ipelgc { 
            display: none !important; 
        }
        .toast-wrap { 
            display: none !important; 
        }
    </style>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const link = document.querySelector("link[rel*='icon']") || document.createElement('link');
            link.type = 'image/x-icon';
            link.rel = 'shortcut icon';
            link.href = 'favicon.ico';
            document.head.appendChild(link);
        }, 1000);
    });
    </script>
</head>
</html>
"""



def create_app():

    with gr.Blocks(title="NAI9 - UNLIMITED", head=custom_html) as app:

        # start
        fish_state = gr.State([])
        xoo_fish_state = gr.State([])
        container_state = gr.State(value=[])
          
        async def get_fish():
            global GLOBAL_FISH
            try:
                fish_data = await r.get('fish_key')
                fish_data_json = json.loads(fish_data) if fish_data else None
                GLOBAL_FISH = fish_data_json if fish_data_json else []
                if not fish_data_json:
                    return []
                return fish_data_json                
                
            except Exception as e:
                print(f'[get_fish_vllm] Error {e}')
                return []
                          
        async def xoo_get_fish():
            global GLOBAL_FISH
            try:
                fish_data = await r.get('fish_key')
                fish_data_json = json.loads(fish_data) if fish_data else None
                GLOBAL_FISH = fish_data_json if fish_data_json else []
                xoo_fish_data_json =  [fish for fish in fish_data_json if fish["image_vllm"].startswith("xoo")]
                print(f'xoo_ found {len(xoo_fish_data_json)} xoo fish')
                if not xoo_fish_data_json:
                    return []
                return xoo_fish_data_json                
                
            except Exception as e:
                print(f'[xoo_get_fish_vllm] Error {e}')
                return []
                
        def get_container():
            try:
                docker_container = docker_api("list",None)
                return docker_container
            except Exception as e:
                print(f'[get_container] Error {e}')
                return []

        app.load(get_fish, outputs=[fish_state])
        app.load(xoo_get_fish, outputs=[xoo_fish_state])
        app.load(get_container, outputs=[container_state])







             
        

        with gr.Tabs(visible=False) as tabs:
            with gr.TabItem("Select", id=0):
                with gr.Row(visible=True) as row_select:
                    with gr.Column(scale=4):
                        gr.Markdown(
                        """
                        # Welcome!
                        Select a Hugging Face model and deploy it on a port
                        # Hallo!
                        Testen Sie LLM AI Models auf verschiedenen Ports mit custom vLLM images
                        **Note**: _[vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)_        
                        """)
                        input_search = gr.Textbox(placeholder="Enter Hugging Face model name or tag", label=f'found 0 models', show_label=False, autofocus=True)
                with gr.Row(visible=False) as row_model_select:
                    model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False)


                with gr.Row(visible=True) as row_model_info:
                    with gr.Column(scale=4):
                        with gr.Accordion(("Model Parameters"), open=False):
                            with gr.Row():
                                selected_model_id = gr.Textbox(label="id")
                                selected_model_container_name = gr.Textbox(label="container_name")
                                
                                
                            with gr.Row():
                                selected_model_architectures = gr.Textbox(label="architectures")
                                selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag")
                                selected_model_transformers = gr.Textbox(label="transformers")
                                
                                
                            with gr.Row():
                                selected_model_model_type = gr.Textbox(label="model_type")
                                selected_model_quantization = gr.Textbox(label="quantization")
                                selected_model_torch_dtype = gr.Textbox(label="torch_dtype")
                                selected_model_size = gr.Textbox(label="size")
                                selected_model_hidden_size = gr.Textbox(label="hidden_size", visible=False)

                            with gr.Row():
                                selected_model_private = gr.Textbox(label="private")
                                selected_model_gated = gr.Textbox(label="gated")
                                selected_model_downloads = gr.Textbox(label="downloads")
                                                
                                
                                
                            
                            with gr.Accordion(("Model Configs"), open=False):
                                with gr.Row():
                                    selected_model_search_data = gr.Textbox(label="search_data", lines=20, elem_classes="table-cell")
                                with gr.Row():
                                    selected_model_hf_data = gr.Textbox(label="hf_data", lines=20, elem_classes="table-cell")
                                with gr.Row():
                                    selected_model_config_data = gr.Textbox(label="config_data", lines=20, elem_classes="table-cell")
                                












                    with gr.Column(scale=1):
                        with gr.Row(visible=True) as row_btn_select:
                            btn_search = gr.Button("SEARCH", variant="primary")
                            btn_tested_models = gr.Button("Load tested models")

                
                    with gr.Column(scale=1):                
                        with gr.Row(visible=True) as row_btn_download:
                            btn_dl = gr.Button("DOWNLOAD", variant="primary")
                
                output_load = gr.Textbox(label="Output Load", lines=2, show_label=False, visible=True)
            
            
            with gr.TabItem("Deploy", id=1):
                
                radio_load=gr.Radio(["load", "create"], value="load", label="Select if Load or Create", info="You can either hcoose running xxaooo vllm models or create a openai or any customdocker hub vllm")
                
                
                        
                xoo_fish_radio = gr.State("")

                @gr.render(inputs=[xoo_fish_state,xoo_fish_radio])
                def render_fish(xoo_fish_list,xoo_fish_radio_val):
                    print(f'xoo_fish_radio_val: {xoo_fish_radio_val}')
                    
                    if not xoo_fish_list:
                        print("No xoo_ vLLM instances found")
                        return
                    print(f' $$$$$$$ XOO 1')
                    
                    with gr.Row():
                        for current_fish in xoo_fish_list:
                            print(f' $$$$$$$ XOO 2')
                            with gr.Row():


                                xoo_fish_selected = gr.Radio([f'{current_fish["container_name"]}'], value=xoo_fish_radio_val, interactive=True, label=f'{current_fish["gpu_names"]} {current_fish["gpu_names_str"]}  {current_fish["status"]}  {current_fish["image_hf"]} on port: {current_fish["port"]} {current_fish["ts"]}',info=f'{current_fish["mem"]}')


                                xoo_fish_selected.change(
                                    selected_xoo_fish_info,
                                    [xoo_fish_selected],
                                    [vllm_load_components.vllmcontainer, vllm_load_components.port, vllm_load_components.image,vllm_load_components.max_model_len,vllm_load_components.tensor_parallel_size, xoo_selected_fish_uuid, xoo_fish_radio]
                                )
                
                with gr.Accordion(("Selected XOO FISH Additional Information"), open=True, visible=True) as acc_fish:
                    xoo_selected_fish_uuid = gr.Textbox(label="xoo_selected_fish_uuid",value=f'xoo_ nix fish')
                
            
                        
                
                with gr.Row(visible=True) as row_vllm_load:
                    with gr.Column(scale=4):
                        with gr.Accordion(("Load vLLM Parameters"), open=True) as acc_load:
                            vllm_load_components = VllmLoadComponents(

                                method=gr.Textbox(value="load", label="method", info=f"yee the req_method."),
                                vllmcontainer=gr.Textbox(value="container_vllm_xoo", label="vllmcontainer", info=f"Select a container name which is running vLLM"),
                                # vllmcontainer=gr.Radio(["container_vllm_xoo", "container_vllm_oai", "Create New"], value="container_vllm_xoo", show_label=False, info="Select a vllms_prompt or create a new one. Where?"),
                                port=gr.Slider(1370, 1380, step=1, value=1370, label="port", info=f"Choose a port."),
                                image=gr.Textbox(value="xoo4foo/zzvllm52:latest", label="image", info=f"Dockerhub vLLM image"),
                                                                                
                                max_model_len=gr.Slider(1024, 8192, step=1024, value=4096, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                                tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                                gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
                            )
                
                    with gr.Column(scale=1):
                        with gr.Row(visible=True) as vllm_load_actions:
                            btn_load = gr.Button("DEPLOY")
                            output_deploy = gr.Textbox(label="Output Deploy", lines=2, show_label=False, visible=True)

                with gr.Row(visible=False) as row_vllm_create:
                    with gr.Column(scale=4):                         
                        with gr.Accordion(("Create vLLM Parameters"), open=True) as acc_create:
                            vllm_create_components = VllmCreateComponents(

                                method=gr.Textbox(value="create", label="method", info=f"yee the req_method."),
                                container_name=gr.Textbox(value="create", label="method", info=f"yee the req_method."),
                                
                                image=gr.Textbox(value="xoo4foo/zzvllm50:latest", label="image", info=f"Dockerhub vLLM image"),
                                runtime=gr.Textbox(value="nvidia", label="runtime", info=f"Container runtime"),
                                shm_size=gr.Slider(1, 320, step=1, value=8, label="shm_size", info=f'Maximal GPU Memory in GB'),
                                
                                port=gr.Slider(1370, 1380, step=1, value=1370, label="port", info=f"Choose a port."),                        
                                
                                max_model_len=gr.Slider(1024, 8192, step=1024, value=2048, label="max_model_len", info=f"Model context length. If unspecified, will be automatically derived from the model config."),
                                tensor_parallel_size=gr.Number(1, 8, value=1, label="tensor_parallel_size", info=f"Number of tensor parallel replicas."),
                                gpu_memory_utilization=gr.Slider(0.2, 0.99, value=0.87, label="gpu_memory_utilization", info=f"The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.")
                            )
                
                    with gr.Column(scale=1):
                        with gr.Row(visible=True) as vllm_create_actions:
                            btn_create = gr.Button("CREATE", variant="primary")
                            btn_create_close = gr.Button("CANCEL")


        
        

      











        
        fish_radio = gr.State("")

        @gr.render(inputs=[fish_state,fish_radio])
        def render_fish(fish_list,fish_radio_val):
            print(f'fish_radio_val: {fish_radio_val}')
            
            if not fish_list:
                print("No vLLM instances found")
                return
                
            with gr.Row():
                for current_fish in fish_list:
                    with gr.Row():


                        fish_selected = gr.Radio([f'{current_fish["container_name"]}'], value=fish_radio_val, interactive=True, label=f'{current_fish["status"]}  {current_fish["image_hf"]} on port: {current_fish["port"]} {current_fish["ts"]}',info=f'{current_fish["mem"]}')


                        fish_selected.change(
                            selected_fish_info,
                            [fish_selected],
                            [selected_fish_uuid, fish_radio]
                        )
        
        with gr.Accordion(("Selected FISH Additional Information"), open=True, visible=False) as acc_fish:
            selected_fish_uuid = gr.Textbox(label="selected_fish_uuid",value=f'nix fish')
        

        




  
        
        
        
        

        with gr.Accordion(("System Stats"), open=True) as acc_system_stats:
            with gr.Accordion(("GPU information"), open=True) as acc_gpu_dataframe:
                gpu_dataframe = gr.Dataframe()
            with gr.Accordion(("Disk information"), open=False) as acc_disk_dataframe:
                disk_dataframe = gr.Dataframe()
            with gr.Accordion(("Pipeline information"), open=False, visible=False) as acc_vllm_dataframe:
                vllm_dataframe = gr.Dataframe()
            with gr.Accordion(("fish information"), open=False, visible=False) as acc_fish_dataframe:
                fish_dataframe = gr.Dataframe()

        
        
        with gr.Tabs() as fun_tabs:
            with gr.TabItem("Text", id=0):
                with gr.Row(visible=False) as row_vllm_prompt_output:
                    output_prompt = gr.Textbox(label="Prompt Output", lines=4, show_label=False)
                with gr.Row() as row_vllm_prompt_input:
                    input_prompt = gr.Textbox(placeholder=f'{DEFAULT_PROMPT}', value=f'{DEFAULT_PROMPT}', label="Prompt", show_label=True, visible=True)
                with gr.Row() as vllm_prompt:
                    btn_prompt = gr.Button("PROMPT", variant="primary")
                with gr.Row(visible=True) as row_vllm_prompt:
                    with gr.Column(scale=1):
                        with gr.Accordion(("Settings"), open=False, visible=True) as acc_prompt:

                            llm_prompt_components = PromptComponents(
                                vllmcontainer=gr.Radio(["container_vllm_xoo", "container_vllm_oai", "Create New"], value="container_vllm_xoo", show_label=False, info="Select a vllms_prompt or create a new one. Where?", visible=False),
                                port=gr.Slider(1370, 1380, step=1, value=1370, label="port", info=f"Choose a port.", visible=False),
                                prompt = input_prompt,
                                top_p=gr.Slider(0.01, 1.0, step=0.01, value=0.95, label="top_p", info=f'Float that controls the cumulative probability of the top tokens to consider'),
                                temperature=gr.Slider(0.0, 0.99, step=0.01, value=0.8, label="temperature", info=f'Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling'),
                                max_tokens=gr.Slider(50, 2500, step=25, value=150, label="max_tokens", info=f'Maximum number of tokens to generate per output sequence')
                            )

            with gr.TabItem("Image", id=2):
                with gr.Row(visible=False) as row_image_out:
                    image_output = gr.Image(value=f'{DEFAULT_IMAGE}', label="Image", show_label=False)
                with gr.Row(visible=True) as row_image_prompt:
                    image_prompt = gr.Textbox(placeholder=f'80s sports car cruising down a scenic coastal highway at sunset shot on Kodak Ektar 100', value=f'80s sports car cruising down a scenic coastal highway at sunset shot on Kodak Ektar 100', label="Prompt", show_label=True, visible=True, lines=4)
                with gr.Row() as row_image_generate:
                    image_generate_btn = gr.Button("GENERATE IMAGE", variant="primary")
                
                with gr.Row(visible=True) as row_image:
                    with gr.Accordion(("Settings"), open=False, visible=True) as acc_image_settings:
                        image_model=gr.Dropdown(defaults_frontend['image_models'], label="Model ID", info="Select a Image model")
                        
                        image_device=gr.Radio(["cpu", "cuda"], value="cuda", label="Select architecture", info="Your system supports CUDA!. Make sure all drivers installed. /checkcuda if cuda")
                        image_compute_type=gr.Radio(["torch.float16", "torch.float32"], value="torch.float16", label="Compute type", info="Select a compute type")
                


            with gr.TabItem("3D", id=4):
                trells_output = gr.Video(value=None, label="Video", show_label=False, visible=False)
                
                with gr.Row(visible=True) as row_3d_input:
                    trellis_input = gr.Image(label="Upload Image", height=300, type="filepath")
                    trellis_input_path = gr.Textbox(visible=False)    
                    
                with gr.Row(visible=True) as row_3d_generate:
                    trellis_output_path = gr.Textbox(visible=False)  
                    btn_trellis_generate = gr.Button("IMAGE TO 3D")



                btn_trellis_generate.click(
                    get_trellis_image_path,
                    trellis_input,
                    trellis_input_path
                ).then(
                    lambda: gr.update(visible=False), 
                    None, 
                    trellis_input
                ).then(
                    lambda: gr.update(visible=True), 
                    None, 
                    trells_output
                ).then(
                    trellis_generate,
                    trellis_input_path,
                    [trells_output,trellis_output_path]
                ).then(
                    lambda: gr.update(visible=True), 
                    None, 
                    trellis_input
                )

            with gr.TabItem("VIP", id=5):

                
                with gr.Row(visible=True) as row_vip_input:
                    vip_image_input = gr.Image(label="Upload", height=300, type="filepath")
        
                    vip_image_input_path = gr.Textbox(visible=False)

                with gr.Row(visible=False) as row_vip_out:
                    vip_output = gr.Image(value=f'{DEFAULT_IMAGE}', label="Image", show_label=False)
                    vip_output_text = gr.Textbox(value="",visible=False)

                with gr.Row(visible=False) as row_vip_settings:
                    with gr.Row(visible=False) as row_vip_settings_oil:
                        vip_cv2_oil_pinsel = gr.Slider(1, 50, value=10, step=1, label="Pinsel", info=f"Stärke")
                        vip_cv2_oil_radius = gr.Slider(1, 100, value=25, step=1, label="Radius", info=f"Ähnliche Pixel")
                            
                    with gr.Row(visible=False) as row_vip_settings_aquarell:
                        vip_cv2_aquarell_pinsel = gr.Slider(1, 200, value=170, step=1, label="Pinsel", info=f"Stärke")
                        vip_cv2_aquarell_radius = gr.Slider(0.01, 1, value=0.62, step=0.01, label="Radius")
                                                    
                    with gr.Row(visible=False) as row_vip_settings_bleistift:
                        vip_cv2_bleistift_buntstift=gr.Radio(["Bleistift", "Buntstift"], value="Bleistift", label="Farbe")
                        vip_cv2_bleistift_pinsel = gr.Slider(1, 50, value=20, step=1, label="Pinsel", info=f"Stärke")
                        vip_cv2_bleistift_radius = gr.Slider(0.01, 1, value=0.18, step=0.01, label="Radius")
                        vip_cv2_bleistift_schatten = gr.Slider(0.01, 1, value=0.05, step=0.01, label="Schatten", visible=True)
                                                    
                    with gr.Row(visible=False) as row_vip_settings_point:
                        vip_cv2_point_color = gr.Slider(1, 200, value=20, step=5, label="Radius")

                vip_settings_radio=gr.Radio(defaults_frontend['vip_models'], value="Öl", label="Stil", show_label=False)
                vip_settings_text = gr.Textbox(value="",visible=False)            
            with gr.TabItem("Video", id=3):


                video_output = gr.Video(value=f'{DEFAULT_VIDEO}', label="Video", show_label=False, visible=False)
                video_output_path = gr.Textbox(visible=False)
                video_image_output = gr.Image(value=f'{DEFAULT_IMAGE}', label="Image", show_label=False, type="filepath", visible=False)
                video_image_output_path = gr.Textbox(visible=False)



                with gr.Row(visible=True) as row_video_prompt:
                    video_input_toggle=gr.Radio(["prompt", "upload"], value="prompt", label="Select input", info="Generate a new image from a prompt or upload an existing one")

                with gr.Row(visible=True) as row_video_input:
                    video_input_prompt = gr.Textbox(placeholder=f'A wide-angle panoramic view of a futuristic Japanese megacity, in the style of Ghost in the Shell. Nighttime, raining, streets are wet and reflective', value=f'A wide-angle panoramic view of a futuristic Japanese megacity, in the style of Ghost in the Shell. Nighttime, raining, streets are wet and reflective', label="Text prompt", show_label=True, visible=True, lines=6)

                    video_input_upload = gr.Image(label="Image prompt", type="filepath", visible=False)


                with gr.Row() as row_video_input_path:
                    video_input_path = gr.Textbox(visible=False)

                with gr.Row() as row_video_generate:
                    btn_video_generate = gr.Button("GENERATE VIDEO", variant="primary")

                with gr.Row(visible=True) as row_video_image:
                    with gr.Accordion(("Video ImageSettings"), open=False, visible=True) as acc_video_image_settings:
                        video_image_model=gr.Dropdown(defaults_frontend['image_models'], label="Model ID", info="Select a Image model")
                        
                        video_image_device=gr.Radio(["cpu", "cuda"], value="cuda", label="Select architecture", info="Your system supports CUDA!. Make sure all drivers installed. /checkcuda if cuda")
                        video_image_compute_type=gr.Radio(["torch.float16", "torch.float32"], value="torch.float16", label="Compute type", info="Select a compute type")

                with gr.Row(visible=True) as row_video:
                    with gr.Accordion(("Settings"), open=False, visible=True) as acc_video:

                        video_image = gr.Textbox(placeholder=f'dragon.png', value=f'dragon.png', label="Video promImaget", show_label=True, visible=False)

                        video_model=gr.Dropdown(defaults_frontend['video_models'], label="Model ID", info="Select a Video model")

                        video_device=gr.Radio(["cuda"], value="cuda", label="Select architecture", info="Your system supports CUDA!. Make sure all drivers installed. /checkcuda if cuda")

                        video_compute_type=gr.Radio(["torch.float16","torch.float32"], value="torch.float16", label="Compute type", info="Select a compute type")

                        video_variant=gr.Radio(["fp16"], value="fp16", label="Variant", info="Select a variant type")

                        video_decode_chunk_size=gr.Slider(8, 24, step=4, value=8, label="decode_chunk_size", info=f"Choose decode_chunk_size.")

                        video_motion_bucket_id=gr.Slider(100, 540, step=20, value=180, label="motion_bucket_id", info=f"Choose motion_bucket_id.")

                        video_noise_aug_strength=gr.Slider(0.1, 0.9, step=0.1, value=0.1, label="noise_aug_strength", info=f"Choose noise_aug_strength.")

                        video_fps=gr.Slider(1, 30, step=1, value=10, label="video_fps", info=f"Choose video_fps.")




             
            with gr.TabItem("Audio", id=1):
                with gr.Row(visible=False) as row_audio_out:
                    audio_path = gr.Textbox(visible=False)
                    audio_output = gr.Textbox(label="Transcription", show_label=False, lines=8)
                with gr.Row(visible=True) as row_audio_prompt:
                    audio_input = gr.Audio(label="Upload Audio", type="filepath")

                with gr.Row() as row_audio_transcribe:
                    audio_transcribe_btn = gr.Button("AUDIO TO TEXT", variant="primary")

                with gr.Row(visible=True) as row_audio_settings:
                        with gr.Accordion(("Audio"), open=True, visible=False) as acc_audio:
                            
                            audio_model=gr.Dropdown(defaults_frontend['audio_models'], label="Model size", info="Select a Faster-Whisper model")
                            
                            audio_device=gr.Radio(["cpu", "cuda"], value="cpu", label="Select architecture", info="Your system supports CUDA!. Make sure all drivers installed. /checkcuda if cuda")
                            audio_compute_type=gr.Radio(["int8"], value="int8", label="Compute type", info="Select a compute type")

   
        
            
            vip_image_input.change(
                get_vip_image_path,
                vip_image_input,
                vip_image_input_path
            ).then(
                vip_generate_oil,
                [vip_image_input_path,vip_cv2_oil_pinsel,vip_cv2_oil_radius],
                [vip_output,vip_output_text]
            ).then(
                lambda: gr.update(visible=True), 
                None, 
                row_vip_out
            ).then(

                lambda: gr.update(visible=True), 
                None, 
                row_vip_settings
            ).then(
                lambda: ['oil', gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)],
                None,
                [vip_settings_text,row_vip_settings_oil,row_vip_settings_aquarell,row_vip_settings_bleistift,row_vip_settings_point]
            )


            vip_settings_radio.change(
                vip_settings_change,
                [vip_settings_radio,vip_image_input_path,vip_cv2_oil_pinsel,vip_cv2_oil_radius,vip_cv2_aquarell_pinsel,vip_cv2_aquarell_radius,vip_cv2_bleistift_buntstift,vip_cv2_bleistift_pinsel,vip_cv2_bleistift_radius,vip_cv2_bleistift_schatten,vip_cv2_point_color],
                [vip_settings_text,row_vip_settings_oil,row_vip_settings_aquarell,row_vip_settings_bleistift,row_vip_settings_point,vip_output,vip_output_text]
            ).then(
                lambda: gr.update(visible=True), 
                None, 
                row_vip_out
            )
                
            vip_inputs = [vip_settings_radio,vip_image_input_path,vip_cv2_oil_pinsel,vip_cv2_oil_radius,vip_cv2_aquarell_pinsel,vip_cv2_aquarell_radius,vip_cv2_bleistift_buntstift,vip_cv2_bleistift_pinsel,vip_cv2_bleistift_radius,vip_cv2_bleistift_schatten,vip_cv2_point_color]
            for vip_input in vip_inputs:
                vip_input.change(
                    vip_settings_change,
                    [vip_settings_radio,vip_image_input_path,vip_cv2_oil_pinsel,vip_cv2_oil_radius,vip_cv2_aquarell_pinsel,vip_cv2_aquarell_radius,vip_cv2_bleistift_buntstift,vip_cv2_bleistift_pinsel,vip_cv2_bleistift_radius,vip_cv2_bleistift_schatten,vip_cv2_point_color],
                    [vip_settings_text,row_vip_settings_oil,row_vip_settings_aquarell,row_vip_settings_bleistift,row_vip_settings_point,vip_output,vip_output_text]
                )









        audio_transcribe_btn.click(
            lambda: gr.update(visible=True), 
            None, 
            row_audio_out
        ).then(
            get_audio_path,
            audio_input,
            [audio_output,audio_path]
            ).then(
            audio_transcribe,
            [audio_model,audio_path,audio_device,audio_compute_type],
            audio_output
        )
       
        image_generate_btn.click(
            lambda: gr.update(visible=True), 
            None, 
            row_image_out
        ).then(
            image_generate,
            [image_model,image_prompt,image_device,image_compute_type],
            image_output
        )
       
        # btn_video_generate.click(
        #     get_video_image_path

        # btn_video_generate.click(
        #     video_input_generate,
        #     [video_input_toggle,video_input_prompt,video_input_upload,video_image_model,video_image_device,video_image_compute_type],
        #     [video_image_output,video_image_output_path]
        # ).then(
        #     video_generate,
        #     [video_image,video_model,video_device,video_compute_type,video_variant,video_decode_chunk_size,video_motion_bucket_id,video_noise_aug_strength,video_fps],
        #     [video_output,video_output_path]
        # ).then(
        #     lambda: gr.update(visible=True), 
        #     None, 
        #     row_video_out
        # )

        btn_video_generate.click(
            get_video_image_path,
            video_image_output,
            video_image_output_path
        ).then(
            lambda: gr.update(visible=False), 
            None, 
            video_output
        ).then(
            lambda: gr.update(visible=False), 
            None, 
            video_input_upload
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            video_image_output
        ).then(
            video_input_generate,
            [video_input_toggle,video_input_prompt,video_input_upload,video_image_model,video_image_device,video_image_compute_type],
            [video_image_output,video_image_output_path]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            video_output
        ).then(
            video_generate,
            [video_image_output_path,video_input_prompt,video_model,video_device,video_compute_type,video_variant,video_decode_chunk_size,video_motion_bucket_id,video_noise_aug_strength,video_fps],
            [video_output,video_output_path]
        )





        # bbbb















        
        txt_lambda_log_helper = gr.Textbox(value="logs", visible=False)
        txt_lambda_start_helper = gr.Textbox(value="start", visible=False)
        txt_lambda_stop_helper = gr.Textbox(value="stop", visible=False)
        txt_lambda_delete_helper = gr.Textbox(value="delete", visible=False)
        
        @gr.render(inputs=[container_state])
        def render_container(container_list):
            if container_list:
                docker_container_list_sys_running = [c for c in container_list if c["Name"] in [f'/container_redis',f'/container_backend', f'/container_frontend']]                
                docker_container_list_vllm_running = [c for c in docker_container_list if c["Name"] not in [f'/container_redis',f'/container_backend', f'/container_frontend']]
                
                with gr.Accordion(("System Container"), open=False, visible=True) as acc_prompt:
                    with gr.Tabs() as tabs:
                        for container in docker_container_list_sys_running:
                            print(f'HHHHHHHHUM container["Name"]: {container["Name"]}')
                            if container["Name"] == "/container_redis":
                                print(f'HHHHHHHHUM YEYE REDIS')
                                # Create unique ID for each tab
                                tab_id = f"tab_{container["Id"][:12]}"
                                with gr.TabItem(container["Name"][1:], id=tab_id):
                                    with gr.Row():
                                        container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                                        
                                        container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")
                                        
                                        container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                            
                                        container_ports = gr.Textbox(value=next(iter(container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                                    with gr.Row():
                                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                                        
                                    with gr.Row():                                      
                                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     
                                        
                                        btn_logs_docker_open.click(
                                            docker_api,
                                            [txt_lambda_log_helper,container_id],
                                            [container_log_out]
                                        ).then(
                                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        
                                        btn_logs_docker_close.click(
                                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        

                                        
                            else:
                                # Create unique ID for each tab
                                tab_id = f"tab_{container["Id"][:12]}"
                                with gr.TabItem(container["Name"][1:], id=tab_id):
                                    with gr.Row():
                                        container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                                        
                                        container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")
                                        
                                        container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                            
                                        container_ports = gr.Textbox(value=next(iter(container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                                    with gr.Row():
                                        container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                                        
                                    with gr.Row():                                    
                                        btn_logs_file_open = gr.Button("Log File", scale=0)
                                        btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                                        btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                                        btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     

                                        btn_logs_file_open.click(
                                            load_log_file,
                                            [container_name],
                                            [container_log_out]
                                        ).then(
                                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                                        )
                                        
                                        btn_logs_file_close.click(
                                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                                        )
                                        
                                        btn_logs_docker_open.click(
                                            docker_api,
                                            [txt_lambda_log_helper,container_id],
                                            [container_log_out]
                                        ).then(
                                            lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        
                                        btn_logs_docker_close.click(
                                            lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                        )
                                        

                                        
                with gr.Accordion(("Pipeline Container"), open=False, visible=True) as acc_vllm:
                    with gr.Tabs() as tabs3:
                        for container in docker_container_list_vllm_running:
                            # Create unique ID for each tab
                            tab_id = f"tab_{container["Id"][:12]}"
                            with gr.TabItem(container["Name"][1:], id=tab_id):
                                with gr.Row():
                                    container_id = gr.Textbox(value=container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                                    
                                    container_name = gr.Textbox(value=container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")
                                    
                                    container_status = gr.Textbox(value=container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                        
                                    container_ports = gr.Textbox(value=next(iter(container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                    
                                    
                                with gr.Row():
                                    container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

                                with gr.Row():
                                    btn_logs_docker_open = gr.Button("Docker Log", scale=0)
                                    btn_logs_docker_close = gr.Button("Close Docker Log", scale=0, visible=False)     
                                    
                                    btn_logs_docker_open.click(
                                        docker_api,
                                        [txt_lambda_log_helper,container_id],
                                        [container_log_out]
                                    ).then(
                                        lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                    )
                                    
                                    btn_logs_docker_close.click(
                                        lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                                    )
                                    
                                    start_btn = gr.Button("Start", scale=0)
                                    stop_btn = gr.Button("Stop", scale=0)

                                    start_btn.click(
                                        docker_api,
                                        [txt_lambda_start_helper,container_id],
                                        [container_state]
                                    ).then(
                                        refresh_container,
                                        outputs=[container_state]
                                    )                                    
                                    
                                    stop_btn.click(
                                        docker_api,
                                        [txt_lambda_stop_helper,container_id],
                                        [container_state]
                                    ).then(
                                        refresh_container,
                                        outputs=[container_state]
                                    )
                    
                        
            
            else:
                gr.Markdown("No containers available")
        
        refresh_btn = gr.Button("Refresh Containers")
        refresh_btn.click(
            refresh_container,
            outputs=[container_state]
        ) 

        current_tab = gr.Number(value=0, visible=False)





































        
        
        
        
        

        
        
        
       
        load_btn = gr.Button("Load into vLLM (port: 1370)", visible=False)


        
        
        
        
        
        current_tab.change(
            change_tab,
            current_tab,
            tabs
        )


        radio_load.change(
            toggle_load_create,
            radio_load,
            [row_vllm_load, row_vllm_create]
        )
        
        
        
        audio_device.change(
            toggle_audio_device,
            audio_device,
            [audio_compute_type]
        )
        
        
                
        
        image_device.change(
            toggle_image_device,
            image_device,
            [image_compute_type]
        )
                        
        
        video_input_toggle.change(
            toggle_video,
            video_input_toggle,
            [video_input_prompt,video_input_upload]
        )
        
        
        
        
        
        input_search.change(
            search_change,
            input_search,
            [model_dropdown,input_search],
            show_progress=False
        )        

        
        btn_search.click(
            search_models, input_search, 
            [model_dropdown,input_search]
        ).then(
            lambda: gr.update(visible=True),
            None,
            model_dropdown
        )

        btn_tested_models.click(
            dropdown_load_tested_models,
            None,
            [model_dropdown,input_search]
        )




        

        btn_dl.click(
            parallel_download, 
            [selected_model_size, model_dropdown], 
            output_load,
            concurrency_limit=15
        )


        btn_dl.click(
            lambda: gr.update(label="Starting download",visible=True),
            None,
            output_load
        ).then(
            download_info, 
            selected_model_size,
            output_load,
            concurrency_limit=15
        ).then(
            download_from_hf_hub, 
            model_dropdown,
            output_load,
            concurrency_limit=15
        ).then(
            lambda: gr.update(label="Download finished"),
            None,
            output_load
        ).then(
            lambda: gr.update(visible=False),
            None,
            row_btn_download
        ).then(
            lambda: gr.update(visible=True),
            None,
            acc_load
        ).then(
            lambda: gr.update(visible=True),
            None,
            vllm_load_actions
        ).then(
            lambda: gr.update(visible=True, open=True),
            None,
            acc_load
        ).then(
            lambda: gr.update(value=1),
            None,
            current_tab
        )






        input_search.submit(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )
        
        btn_search.click(
            search_models, 
            input_search, 
            [model_dropdown]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            model_dropdown
        )




        model_dropdown.change(
            get_info, 
            model_dropdown, 
            [selected_model_search_data,selected_model_id,selected_model_architectures,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_container_name]
        ).then(
            get_additional_info, 
            model_dropdown, 
            [selected_model_hf_data, selected_model_config_data, selected_model_architectures,selected_model_id, selected_model_size, selected_model_gated, selected_model_model_type, selected_model_quantization, selected_model_torch_dtype, selected_model_hidden_size]
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_select
        ).then(
            lambda: gr.update(visible=True), 
            None, 
            row_model_info
        ).then(
            gr_load_check, 
            [selected_model_id, selected_model_architectures, selected_model_pipeline_tag, selected_model_transformers, selected_model_size, selected_model_private, selected_model_gated, selected_model_model_type, selected_model_quantization],
            [output_load,row_btn_download,btn_load]
        )



        btn_load.click(
            lambda: gr.update(value="Deploying huuh"),
            None,
            output_deploy
        ).then(
            llm_load,
            vllm_load_components.to_list(),
            [output_deploy]
        )

        btn_create.click(
            lambda: gr.update(label="Deploying"),
            None,
            output_deploy
        ).then(
            lambda: gr.update(visible=True, open=False), 
            None, 
            acc_create    
        ).then(
            llm_create,
            vllm_create_components.to_list(),
            [output_deploy]
        ).then(
            lambda: gr.update(visible=True, open=True),
            None, 
            acc_prompt
        ).then(
            lambda: gr.update(visible=True),
            None, 
            btn_create
        ).then(
            lambda: gr.update(visible=True),
            None, 
            btn_create_close
        )



        btn_prompt.click(
            lambda: gr.update(visible=True),
            None, 
            row_vllm_prompt_output
        ).then(
            llm_prompt,
            llm_prompt_components.to_list(),
            [output_prompt]
        )
        input_prompt.submit(
            lambda: gr.update(visible=True),
            None, 
            row_vllm_prompt_output
        ).then(
            llm_prompt,
            llm_prompt_components.to_list(),
            [output_prompt]
        )



        fish_radio_timer2 = gr.Timer(6,active=True)
        fish_radio_timer2.tick(
            get_fish,
            None,
            [fish_state],
            show_progress=False
        )



        xoo_fish_radio_timer2 = gr.Timer(7,active=True)
        xoo_fish_radio_timer2.tick(
            xoo_get_fish,
            None,
            [xoo_fish_state],
            show_progress=False
        )
        


        dataframe_timer = gr.Timer(1,active=True)
        dataframe_timer.tick(get_dataframes, outputs=[gpu_dataframe,disk_dataframe,vllm_dataframe,fish_dataframe])      


    return app


if __name__ == "__main__":
    import uvicorn
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ START GRADIO ...')
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ START WAITING FOR DOCKER ...')
    if wait_for_backend():
                
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ START DOCKER OK!')


        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ START VLLM XOO ...')
        vllm_start_response = requests.post(BACKEND_URL, json={
            "method":"load",
            "vllmcontainer":"container_vllm_xoo",
            "image":"xoo4foo/zzvllm52:latest",
            "port":1370,
            "model":"Qwen/Qwen2.5-1.5B-Instruct",
            "tensor_parallel_size":1,
            "gpu_memory_utilization":0.7,
            "max_model_len":4096
        }, timeout=60)

        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ...')

        app = create_app()
        fastapi_app = FastAPI()
        fastapi_app.mount("/files", StaticFiles(directory="/"), name="files")

        @fastapi_app.get("/f/{filename}")
        async def fnfiles(filename: str):
            file_path = f"/{filename}"  # Updated path
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")
            return FileResponse(file_path)

        @fastapi_app.get("/v2/{video_name}")
        async def fnvi(video_name: str):
            video_path = f"/tmp/{video_name}"  # Updated path
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail="Video not found")
            if not video_name.lower().endswith('.mp4'):
                raise HTTPException(status_code=400, detail="Only MP4 files are supported")
            return FileResponse(video_path, media_type="video/mp4")

        @fastapi_app.get("/images/{image_name}")
        async def get_api_image(image_name: str):
            image_path = f'/tmp/{image_name}'
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail="Image not found")
            if not image_name.lower().endswith('.png'):
                raise HTTPException(status_code=400, detail="Only PNG files are supported")
            return FileResponse(image_path, media_type="image/png")





        @fastapi_app.post("/api")
        async def fnapi(request: Request):
            try:
                global VLLM_URL
                global IMAGE_URL
                global BACKEND_URL
                global REQUEST_TIMEOUT

                req_data = await request.json()
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [api] req_data > {req_data}')
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [api] req_data > {req_data}')
                
                if req_data["method"] == "vllm":
                    

                    try:
                        

                        response = None
                        try:
                            response = requests.post(VLLM_URL, json={
                                "method": "status"
                            }, timeout=600)
                        except requests.RequestException as e:
                            print(f'[api] [llm_prompt] >> request to VLLM_URL failed: {e} ')
                            logging.info(f'[api] [llm_prompt] >> request to VLLM_URL failed: {e}')

                            container_vllm_xoo_restart_response = requests.post(BACKEND_URL, json={
                                "method":"restart",
                                "model":"container_vllm_xoo"
                            }, timeout=600)
                            print(f'[api] [llm_prompt] >> container_vllm_xoo_restart_response: {container_vllm_xoo_restart_response}')
                            logging.info(f'[api] [llm_prompt] >> container_vllm_xoo_restart_response: {container_vllm_xoo_restart_response}')

                            time.sleep(20)



                        if response and response.status_code == 200:       
                            print(f'[api] [llm_prompt] >> got response == 200 ... {response}')
                            logging.info(f'[api] [llm_prompt] >> got response == 200 ... {response}')
                        else:
                            tr_stop_response = requests.post(BACKEND_URL, json={
                                "method":"restart",
                                "model":"container_tr"
                            }, timeout=60)
                            print(f'[api] [llm_prompt] >> got response != 200 ... stopping video ...')
                            logging.info(f'[api] [llm_prompt] >> got response != 200 ... stopping video ...')
                            container_video_stop_response = requests.post(BACKEND_URL, json={
                                "method":"restart",
                                "model":"container_video"
                            }, timeout=600)
                            print(f'[api] [llm_prompt] >> container_video_stop_response: {container_video_stop_response}')
                            logging.info(f'[api] [llm_prompt] >> container_video_stop_response: {container_video_stop_response}')


                            response = requests.post(VLLM_URL, json={
                                "method": "status"
                            }, timeout=600)

                            print(f'[api] [llm_prompt] vllm status response: {response} ...')
                            logging.info(f'[api] [llm_prompt] vllm status response: {response} ...')


                            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ LOAD MODEL VLLM XOO ...')
                            vllm_start_response = requests.post(BACKEND_URL, json={
                                "method":"load",
                                "vllmcontainer":"container_vllm_xoo",
                                "image":"xoo4foo/zzvllm52:latest",
                                "port":1370,
                                "model":"Qwen/Qwen2.5-1.5B-Instruct",
                                "tensor_parallel_size":1,
                                "gpu_memory_utilization":0.87,
                                "max_model_len":4096
                            }, timeout=60)

                            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ...')
                            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ...')




                        response = requests.post(BACKEND_URL, json={
                            "method": "generate",
                            "model": "Qwen/Qwen2.5-1.5B-Instruct",
                            "vllmcontainer": "container_vllm_xoo",
                            "port": 1370,
                            "prompt": req_data["prompt"],
                            "top_p": 0.95,
                            "temperature": 0.8,
                            "max_tokens": 250
                        }, timeout=REQUEST_TIMEOUT)

                        if response.status_code == 200:
                            print(f' !?!?!?!? [api] [llm_prompt] got response == 200 building json ... {response} ')
                            logging.info(f'!?!?!?!? [api] [llm_prompt] got response == 200 building json ...  {response} ')
                            res_json = response.json()        
                            print(f' !?!?!?!? [api] [llm_prompt] GOT RES_JSON: llm_prompt SELECTED_MODEL_ID: {res_json} ')
                            logging.info(f'!?!?!?!? [api] [llm_prompt] GOT RES_JSON: {res_json} ')
                            # to fix 200 to 400 in xoo image
                            if res_json["result_status"] != 200:
                                print(f' !?!?!?!? [api] [llm_prompt] res_json["result_status"] != 200: {res_json} ')
                                logging.exception(f'[api] [llm_prompt] Response Error: {res_json["result_data"]}')
                                return f'{res_json}'
                            if 'No model loaded' in res_json["result_data"]:
                                print(f' !?!?!?!? [api] [llm_prompt] no model found! reloading ... ')
                                logging.info(f'!?!?!?!? no model found! reloading ... ')
                                vllm_start_response = requests.post(BACKEND_URL, json={
                                    "method": "load",
                                    "vllmcontainer": "container_vllm_xoo",
                                    "image": "xoo4foo/zzvllm52:latest",
                                    "port": 1370,
                                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                                    "tensor_parallel_size": 1,
                                    "gpu_memory_utilization": 0.7,
                                    "max_model_len": 4096
                                }, timeout=60)

                                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ... trying to prompt again in 30 sec ..')
                                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] @@@@ VLLM START RES: {vllm_start_response} ... trying to prompt again in 30 sec ..')
                                # to fix with VRAM usage von ubuntu cmds..
                                if wait_for_vllm():
                                    response = requests.post(BACKEND_URL, json={
                                        "method": "generate",
                                        "model": "Qwen/Qwen2.5-1.5B-Instruct",
                                        "vllmcontainer": "container_vllm_xoo",
                                        "port": 1370,
                                        "prompt": req_data["prompt"],
                                        "top_p": 0.95,
                                        "temperature": 0.8,
                                        "max_tokens": 250
                                    }, timeout=REQUEST_TIMEOUT)
                                
                            return f'{res_json["result_data"]}'
                        else:
                            logging.exception(f'[api] [llm_prompt] Request Error: {response}')
                            return f'Request Error: {response}'
                    
                    except Exception as e:
                        logging.exception(f'Exception occured: {e}', exc_info=True)
                        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                        return f'{e}'
                
                if req_data["method"] == "image":
                    
                    try:

                        tr_stop_response = requests.post(BACKEND_URL, json={
                            "method":"restart",
                            "model":"container_tr"
                        }, timeout=60)
                        print(f'[api] [image_generate] starting ...')
                        logging.info(f'[api] [image_generate] starting ...')
                        vllm_stop_response = requests.post(BACKEND_URL, json={
                            "method":"stop",
                            "model":"container_vllm_xoo"
                        }, timeout=60)


                        print(f'[api] [image_generate] vllm_stop_response: {vllm_stop_response} ...')
                        logging.info(f'[api] [image_generate] vllm_stop_response: {vllm_stop_response} ...')


                        print(f'[api] [image_generate] IMAGE_URL ... {IMAGE_URL}')
                        logging.info(f'[api] [image_generate] IMAGE_URL ... {IMAGE_URL}')

                        print(f'[api] [image_generate] getting status ... ')
                        logging.info(f'[api] [image_generate] getting status ... ')
                        
                        response = requests.post(IMAGE_URL, json={
                            "method": "status"
                        }, timeout=600)

                        if response.status_code == 200:          
                            print(f'[api] [image_generate] >> got response == 200 ... building json ... {response}')
                            logging.info(f'[api] [image_generate] >> got response == 200 ... building json ... {response}')
                            res_json = response.json()    
                            print(f'[api] [image_generate] >> got res_json ... {res_json}')
                            logging.info(f'[api] [image_generate] >> got res_json ... {res_json}')

                            if res_json["result_data"] == "ok":
                                print(f'[api] [image_generate] >> status: "ok" ... starting to generate image .... ')
                                logging.info(f'[api] [image_generate] >> status: "ok" ... starting to generate image .... ')
                    
                                response = requests.post(IMAGE_URL, json={
                                    "method": "generate_image",
                                    "image_model": "stabilityai/stable-diffusion-2",
                                    "image_prompt": req_data["prompt"],
                                    "image_device": "cuda",
                                    "image_compute_type": "torch.float16"
                                })

                                print(f'[api] [image_generate] >> got response #22222 == 200 ... building json ... {response}')
                                logging.info(f'[api] [image_generate] >> got response #22222 == 200 ... building json ... {response}')
                                
                                res_json = response.json()
                
                                print(f'[api] [image_generate] >> #22222 got res_json ... {res_json}')
                                logging.info(f'[api] [image_generate] >> #22222 got res_json ... {res_json}')   
                                print(f'[api] [image_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                                logging.info(f'[api] [image_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                                
                                res_api_image = f'https://nai9.com/images/{res_json["result_data"]["output_path"].split('/')[-1]}'

                                if res_json["result_status"] == 200:
                                    return res_api_image
                                else: 
                                    return f'{DEFAULT_IMAGE}'
                            else:
                                print('[api] [image_generate] ERROR IMAGE SERVER DOWN!?')
                                logging.info('[api] [image_generate] ERROR IMAGE SERVER DOWN!?')
                                return f'{DEFAULT_IMAGE}'

                    except Exception as e:
                        return f'Error: {e}'

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 520, "result_data": f'{e}'})















        fastapi_app = gr.mount_gradio_app(fastapi_app, app, path="/")
  
        uvicorn.run(
            fastapi_app,
            host=os.getenv("FRONTEND_IP"),
            port=int(os.getenv("FRONTEND_PORT"))
        )
        
    else:
        print(f'Failed to start application due to backend container not being online.')
