from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import docker
from docker.types import DeviceRequest
import time
import os
import requests
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import pynvml
import psutil
import logging
# import redis
import redis.asyncio as redis
from collections import Counter


GPU_LIST = []

print(f'** connecting to redis on port: {os.getenv("REDIS_PORT")} ... ')
# r = redis.Redis(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0)
pool = redis.ConnectionPool(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0, decode_responses=True, max_connections=10)
r = redis.Redis(connection_pool=pool)
pipe = r.pipeline()


# created (running time)
# port 
# gpu name 
# gpu uuid
# public or private 
# user 
# model 
# vllm image 
# prompts amount
# tokens

# computed





print(f' %%%%% trying to start docker ...')
client = docker.from_env()
print(f' %%%%% docker started!')
print(f' %%%%% trying to docker network ...')
network_name = "sys_net"
# try:
#     network = client.networks.get(network_name)
# except docker.errors.NotFound:
#     network = client.networks.create(network_name, driver="bridge")
# print(f' %%%%% docker network started! ...')






LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_backend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')
print(f'** connecting to pynvml ... ')
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
print(f'** pynvml found GPU: {device_count}')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] pynvml found GPU: {device_count}')

device_uuids = []
for i in range(0,device_count):
    # print(f'1 i {i}')
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    # print(f'1 handle {handle}')
    current_uuid = pynvml.nvmlDeviceGetUUID(handle)
    device_uuids.append(current_uuid)

# print(f'** pynvml found uuids ({len(device_uuids)}): {device_uuids} ')


DEFAULTS_PATH = "/usr/src/app/utils/defaults.json"
if not os.path.exists(DEFAULTS_PATH):
    logging.info(f' [START] File missing: {DEFAULTS_PATH}')

with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
    defaults_backend = json.load(f)["backend"]
    logging.info(f' [START] SUCCESS! Loaded: {DEFAULTS_PATH}')
    DEFAULT_CONTAINER_STATS = defaults_backend['DEFAULT_CONTAINER_STATS']
    logging.info(f' [START] SUCCESS! Loaded DEFAULT_CONTAINER_STATS: {DEFAULT_CONTAINER_STATS}')
    COMPUTE_CAPABILITIES = defaults_backend['compute_capability']
    logging.info(f' [START] SUCCESS! Loaded COMPUTE_CAPABILITIES: {COMPUTE_CAPABILITIES}')



##########################################################################################################

# spaerter vllm s
def get_vllm_info():
    try:
        global GPU_LIST
        vllm_info = []
        res_container_list = client.containers.list(all=True)
        res_container_list_attrs = [container.attrs for container in res_container_list]
        for c in res_container_list_attrs:
            c_status = f'{c["State"]["Status"] if "State" in c and "Status" in c["State"] else "No status"}'
            
            vllm_info.append({
                "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                "name": c.get("Name", "nadaname"),
                "uid": c.get("Id", "noid"),
                "container_name": c.get("Name", "nadacontainername"),
                "status": c_status,
                "gpu_list": [0,1],
                "mem": f'{GPU_LIST[0]["mem"]}',
                "gpu": f'{GPU_LIST[0]["gpu"]}',
                "temp": f'{GPU_LIST[0]["temp"]}'
            })
        return vllm_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllm_info] {e}')
        return f'{e}'

def get_disk_info():
    try:
        disk_info = []
        partitions = psutil.disk_partitions(all=False)
        processed_devices = set()
        for partition in partitions:
            device = partition.device
            if device not in processed_devices:
                processed_devices.add(device)
                current_disk_info = {}
                try:                
                    current_disk_info['device'] = str(partition.device)
                    current_disk_info['mountpoint'] = str(partition.mountpoint)
                    current_disk_info['fstype'] = str(partition.fstype)
                    current_disk_info['opts'] = str(partition.opts)
                    
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    current_disk_info['usage_total'] = f'{disk_usage.total / (1024**3):.2f} GB'
                    current_disk_info['usage_used'] = f'{disk_usage.used / (1024**3):.2f} GB'
                    current_disk_info['usage_free'] = f'{disk_usage.free / (1024**3):.2f} GB'
                    current_disk_info['usage_percent'] = f'{disk_usage.percent}%'
                    
                except Exception as e:
                    print(f'[ERROR] [get_disk_info] Usage: [Permission denied] {e}')
                    pass
                
                try:                
                    io_stats = psutil.disk_io_counters()
                    current_disk_info['io_read_count'] = str(io_stats.read_count)
                    current_disk_info['io_write_count'] = str(io_stats.write_count)
                    
                except Exception as e:
                    print(f'[ERROR] [get_disk_info] Disk I/O statistics not available on this system {e}')
                    pass
                
                disk_info.append({
                    "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                    "device": current_disk_info.get("device", "0"),
                    "mountpoint": current_disk_info.get("mountpoint", "0"),
                    "fstype": current_disk_info.get("fstype", "0"),
                    "opts": current_disk_info.get("opts", "0"),
                    "usage_total": current_disk_info.get("usage_total", "0"),
                    "usage_used": current_disk_info.get("usage_used", "0"),
                    "usage_free": current_disk_info.get("usage_free", "0"),
                    "usage_percent": current_disk_info.get("usage_percent", "0"),
                    "io_read_count": current_disk_info.get("io_read_count", "0"),
                    "io_write_count": current_disk_info.get("io_write_count", "0")
                })

        return disk_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'

def get_gpu_info():
    try:
        global GPU_LIST
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        global_gpu_list = []
        for i in range(0,device_count):
            current_gpu_info = {}
            current_gpu_info['res_gpu_i'] = str(i)           
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            try:
                res_uuid = pynvml.nvmlDeviceGetUUID(handle)
                current_gpu_info['res_uuid'] = f'{res_uuid}'
            except Exception as e:
                print(f'0 gpu_info {e}')
                current_gpu_info['res_uuid'] = f'0'

            try:
                res_name = pynvml.nvmlDeviceGetName(handle)
                current_gpu_info['res_name'] = f'{res_name}'
            except Exception as e:
                print(f'00 gpu_info {e}')
                current_gpu_info['res_name'] = f'0'

            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                # mem_util = f'{(mem_used / mem_total) * 100} %'
                res_gpu_util = f'{utilization.gpu}%'
                current_gpu_info['res_gpu_util'] = f'{res_gpu_util}'
                
                
                # res_mem_util = f'{utilization.memory}%'
                # current_gpu_info['res_mem_util'] = f'{res_mem_util}'
            except Exception as e:
                print(f'1 gpu_info {e}')

            try: 
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                res_mem_total = f'{mem_info.total / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_total'] = f'{res_mem_total}'
                res_mem_used = f'{mem_info.used / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_used'] = f'{res_mem_used}'
                res_mem_free = f'{mem_info.free / 1024 ** 2:.2f} MB'
                current_gpu_info['res_mem_free'] = f'{res_mem_free}'
                
                res_mem_util = (float(mem_info.used / 1024**2)/float(mem_info.total / 1024**2)) * 100
                current_gpu_info['res_mem_util'] = f'{"{:.2f}".format(res_mem_util)}% ({res_mem_used}/{res_mem_total})'
                current_gpu_info['res_mem_util_perc'] = f'{"{:.2f}".format(res_mem_util)}%'

            except Exception as e:
                print(f'2 gpu_info {e}')
            
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                res_temperature = f'{temperature}°C'
                current_gpu_info['res_temperature'] = f'{res_temperature}'
            except Exception as e:
                print(f'3 gpu_info {e}')
                
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                res_fan_speed = f'{fan_speed}%'
                current_gpu_info['res_fan_speed'] = f'{res_fan_speed}'
            except Exception as e:
                print(f'4 gpu_info {e}')

            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
                res_power_usage = f'{power_usage / 1000:.2f} W'
                current_gpu_info['res_power_usage'] = f'{res_power_usage}'
            except Exception as e:
                print(f'5 gpu_info {e}')
        
            try:
                clock_info_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                res_clock_info_graphics = f'{clock_info_graphics} MHz'
                current_gpu_info['res_clock_info_graphics'] = f'{res_clock_info_graphics}'
            except Exception as e:
                print(f'6 gpu_info {e}')

            try:
                clock_info_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                res_clock_info_mem = f'{clock_info_mem} MHz'
                current_gpu_info['res_clock_info_mem'] = f'{res_clock_info_mem}'
            except Exception as e:
                print(f'7 gpu_info {e}')
   
            try:
                cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
                res_cuda_cores = f'{cuda_cores}'
                current_gpu_info['res_cuda_cores'] = f'{res_cuda_cores}'
            except Exception as e:
                print(f'8 gpu_info {e}')

            res_supported = []
            res_not_supported = []
            
            try:
                compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability_str = f'{compute_capability[0]}.{compute_capability[1]}'
                res_compute_capability = f'{compute_capability_str}'

                if float(res_compute_capability) >= 8:
                    res_supported.append('Bfloat16')
                else:
                    res_not_supported.append('Bfloat16')
            except Exception as e:
                print(f'9 gpu_info {e}')
                res_compute_capability = 0

            if res_compute_capability == 0:
                try:
                    res_name = pynvml.nvmlDeviceGetName(handle)
                    res_name_split = res_name.split(" ")[1:]
                    res_name_splitted_str = " ".join(res_name_split)
                    if res_name.lower() in defaults_backend['compute_capability']:
                        print(f'-> res_name {res_name} exists with compute capability {defaults_backend["compute_capability"][res_name.lower()]}')
                        res_compute_capability = f'{defaults_backend["compute_capability"][res_name.lower()]}'
                    elif res_name_splitted_str.lower() in defaults_backend['compute_capability']:
                        print(f'-> res_name_splitted_str {res_name_splitted_str} exists with compute capability {defaults_backend["compute_capability"][res_name.lower()]}')
                        res_compute_capability = f'{defaults_backend["compute_capability"][res_name_splitted_str.lower()]}'
                    else:
                        print(f'{res_name.lower()} or {res_name_splitted_str.lower()} not found in database')
                except Exception as e:
                    print(f'99 res_compute_capability e: {e}')


            res_supported_str = ",".join(res_supported)
            current_gpu_info['res_supported_str'] = f'{res_supported_str}'
            res_not_supported_str = ",".join(res_not_supported)
            current_gpu_info['res_not_supported_str'] = f'{res_not_supported_str}'

            gpu_info.append({
                "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                "gpu_i": current_gpu_info.get("res_gpu_i", "0"),
                "name": current_gpu_info.get("res_name", "0"),
                "current_uuid": current_gpu_info.get("res_uuid", "0"),
                "gpu_util": current_gpu_info.get("res_gpu_util", "0"),
                "mem_util": current_gpu_info.get("res_mem_util", "0"),
                "mem_total": current_gpu_info.get("res_mem_total", "0"),
                "mem_used": current_gpu_info.get("res_mem_used", "0"),
                "mem_free": current_gpu_info.get("res_mem_free", "0"),
                "temperature": current_gpu_info.get("res_temperature", "0"),
                "fan_speed": current_gpu_info.get("res_fan_speed", "0"),
                "power_usage": current_gpu_info.get("res_power_usage", "0"),
                "clock_info_graphics": current_gpu_info.get("res_clock_info_graphics", "0"),
                "clock_info_mem": current_gpu_info.get("res_clock_info_mem", "0"),
                "cuda_cores": current_gpu_info.get("res_cuda_cores", "0"),
                "compute_capability": current_gpu_info.get("res_compute_capability", "0"),
                "supported": current_gpu_info.get("res_supported", "0"),
                "not_supported": current_gpu_info.get("res_not_supported", "0")
            })

            global_gpu_list.append({
                "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                "i": current_gpu_info.get("res_gpu_i", "0"),
                "name": current_gpu_info.get("res_name", "0"),
                "uuid": current_gpu_info.get("res_uuid", "0"),
                "gpu": current_gpu_info.get("res_gpu_util", "0"),
                "mem": current_gpu_info.get("res_mem_util_perc", "0"),
                "fan_speed": current_gpu_info.get("res_fan_speed", "0"),
                "power_usage": current_gpu_info.get("res_power_usage", "0"),
                "cuda_cores": current_gpu_info.get("res_cuda_cores", "0"),
                "compute_capability": current_gpu_info.get("res_compute_capability", "0"),
                "supported": current_gpu_info.get("res_supported", "0"),
                "not_supported": current_gpu_info.get("res_not_supported", "0"),
                "temp": current_gpu_info.get("res_temperature", "0")
            })

        GPU_LIST = global_gpu_list
        return gpu_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'{e}'


async def update_all():
    while True:
        try:
            # docker
            res_container_list = client.containers.list(all=True)
            res_container_list_attrs = [container.attrs for container in res_container_list]
            
            # redis
            res_vllm, res_fish = await asyncio.gather(
                r.get('vllm_key'),
                r.get('fish_key')
            )
            
            vllm_data_json = json.loads(res_vllm) if res_vllm else None
            fish_data_json = json.loads(res_fish) if res_fish else None

            # update_vllm
            try:
                global GPU_LIST
                vllm_info = []
                res_container_list = client.containers.list(all=True)
                res_container_list_attrs = [container.attrs for container in res_container_list]
                for c in res_container_list_attrs:
                    c_status = f'{c["State"]["Status"] if "State" in c and "Status" in c["State"] else "No status"}'
                    
                    vllm_info.append({
                        "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                        "name": c.get("Name", "nadaname"),
                        "uid": c.get("Id", "noid"),
                        "container_name": c.get("Name", "nadacontainername"),
                        "status": c_status,
                        "gpu_list": [0,1],
                        "mem": f'{GPU_LIST[0]["mem"]}',
                        "gpu": f'{GPU_LIST[0]["gpu"]}',
                        "temp": f'{GPU_LIST[0]["temp"]}'
                    })
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [update_all] {e}')


            # update_fish
            try:                
                updated_fish = []
                for fish in fish_data_json:
                    fish["ts"] = f'{datetime.now().strftime("%H:%M:%S")}'
                    fish_container = [c for c in res_container_list_attrs if c["Name"] == fish["name"]][0]
                    
                    fish_port = next(iter(fish_container["HostConfig"]["PortBindings"]))
                    fish_port_split = fish_port.split('/')[0]
                    fish["port"] = fish_port_split
                                    
                    fish["status"] = fish_container["State"]["Status"]
    
    
                    res_gpu_names = []
                    res_gpu_string = f''
                    res_mem_string = f''
                    res_temp_string = f''
                    for fish_gpu in fish["gpu_list"]:
                        res_gpu_names += [GPU_LIST[fish_gpu]["name"]]
                        res_mem_string += f'{GPU_LIST[fish_gpu]["mem"]} | '
                        res_gpu_string += f'{GPU_LIST[fish_gpu]["gpu"]} | '
                        res_temp_string += f'{GPU_LIST[fish_gpu]["temp"]} | '
                    
                    
                    fish["gpu_names"] = res_gpu_names
                    
                    res_gpu_names_str = (lambda arr: ', '.join(f"{arr.count(item)}x {item}" for item in sorted(set(arr))))(res_gpu_names)
                    fish["gpu_names_str"] = f'{res_gpu_names_str}'
                    
                    fish["gpu"] = f'{res_gpu_string}'
                    fish["mem"] = f'{res_mem_string}'
                    fish["temp"] = f'{res_temp_string}'
                    updated_fish.append(fish)
                
                
            except Exception as e: 
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [update_all] Error: {e}')


            # update_disk
            try:
                data_disk = get_disk_info()
            except Exception as e: 
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [update_all] Error: {e}')


            # update_gpu
            try:
                data_gpu = get_gpu_info()
                res_gpu_arr = []
                curr_gpu_i = 0
                for a_gpu in data_gpu:
                    current_gpu_obj = {
                        f'id': f'{curr_gpu_i}',
                        f'{a_gpu["current_uuid"]}': f'{a_gpu["mem_util"]}',
                        f'ts': f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    }
                    res_gpu_arr.append(current_gpu_obj)
                    curr_gpu_i = curr_gpu_i + 1

            except Exception as e: 
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [update_all] Error: {e}')

            # pipe.setex('gpu_key', 3600, json.dumps(data_gpu))
            pipe.set('fish_key', json.dumps(updated_fish)) \
                .set('gpu_key', json.dumps(data_gpu)) \
                .set('disk_key', json.dumps(data_disk)) \
                .set('vllm_key', json.dumps(vllm_info))
            await pipe.execute()
            
            await asyncio.sleep(1)
            
            
            
            
            
        except Exception as e: 
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [update_vllm] Error: {e}')
            await asyncio.sleep(1)



async def default_vllm():
    try:
        global GPU_LIST
        vllm_info = []
        res_container_list = client.containers.list(all=True)
        res_container_list_attrs = [container.attrs for container in res_container_list]
        res_container_oai = [c for c in res_container_list_attrs if c["Name"]== "/container_vllm_oai"][0]
        res_container_xoo = [c for c in res_container_list_attrs if "/container_vllm_xoo" in c["Name"]]
        
        for default_vllm in res_container_xoo:
            current_port = f'{next(iter(default_vllm["HostConfig"]["PortBindings"]))}'
            current_port_split = current_port.split('/')[0]
            
            current_id = f'{default_vllm["Id"]}'
            current_name = f'{default_vllm["Name"]}'
            current_status = f'{default_vllm["State"]["Status"]}'
            current_vllm = {
                "prompt_active": 0,
                "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                "name": current_name,
                "image_hf": "Qwen/Qwen2.5-1.5B-Instruct",
                "image_vllm": "xoo4foo/zzvllm52:latest",
                "uid": current_id,
                "container_name": current_name,
                "status": current_status,
                "port": current_port_split,
                "gpu_list": [0,1],
                "gpu_names": ["NO GPU"],
                "gpu_names_str": f'0x NVIDIA GPU',
                "mem": f'000000',
                "gpu": f'000000',
                "temp": f'000000',
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.88                
            }
            
            vllm_info.append(current_vllm)
            
            
        vllm2_current_port = f'{next(iter(res_container_oai["HostConfig"]["PortBindings"]))}'
        vllm2_current_port_split = current_port.split('/')[0]
        vllm_2 = {
            "prompt_active": 0,
            "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
            "name": res_container_oai["Name"],
            "image_hf": "Qwen/Qwen2.5-1.5B-Instruct",
            "image_vllm": "vllm/vllm-openai:latest",
            "uid": res_container_oai["Id"],
            "container_name": res_container_oai["Name"],
            "status": res_container_oai["State"]["Status"],
            "port": vllm2_current_port_split,
            "gpu_list": [0],
            "gpu_names": ["NO GPU"],
            "gpu_names_str": f'0x NVIDIA GPU',
            "mem": f'000000',
            "gpu": f'000000',
            "temp": f'000000',
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.88  
        }

        
        vllm_info.append(vllm_2)
            
        pipe.set('fish_key', json.dumps(vllm_info))
        await pipe.execute()
        return vllm_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [get_vllm_info] {e}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GPU_LIST
    print(f'-> getting GPUs start ....')
    res_gpus = get_gpu_info()
    print(f'-> res_gpus: {res_gpus}')
    print(f'-> GPU_LIST: {GPU_LIST}')
    print(f'-> default_vllm start ....')
    res_default = await default_vllm()
    print(f'-> default_vllm: {res_default}')
    asyncio.create_task(update_all())
    yield

app = FastAPI(lifespan=lifespan)




device_request = DeviceRequest(count=-1, capabilities=[["gpu"]])




async def stop_vllm_container():
    try:
        print(f' -> stop_vllm_container')
        res_container_list = client.containers.list(all=True)
        print(f'-> mhmmhmhmh 1')
        vllm_containers_running = [c for c in res_container_list if c.name.startswith("container_vllm") and c.status == "running"]
        print(f'-> found total vLLM running containers: {len(vllm_containers_running)}')
        while len(vllm_containers_running) > 0:
            print(f'stopping all vLLM containers...')
            for vllm_container in vllm_containers_running:
                print(f'-> stopping container {vllm_container.name}...')
                vllm_container.stop()
                vllm_container.wait()
            res_container_list = client.containers.list(all=True)
            vllm_containers_running = [c for c in res_container_list if c.name.startswith("vllm") and c.status == "running"]
        print(f'-> all vLLM containers stopped successfully')
        return 200
    except Exception as e:
        print(f'-> error e: {e}') 
        return 520

                
@app.get("/")
async def root():
    return f'Hello from server {os.getenv("BACKEND_PORT")}!'



# @app.post("/api")
# async def fnapi(request: Request):
#     try:
#         VLLM_URL = f'http://container_vllm_xoo:{os.getenv("VLLM_PORT")}/status'
#         IMAGE_URL = f'http://container_video:{os.getenv("VIDEO_PORT")}/generateimage'

#         req_data = await request.json()
#         print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')
#         logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')
        
#         if req_data["method"] == "vllm":
            
#             try:
#                 response = None
#                 try:
#                     response = requests.post(VLLM_URL, json={
#                         "method": "status"
#                     }, timeout=600)
#                 except requests.RequestException as e:
#                     print(f'[llm_prompt] >> request to VLLM_URL failed: {e} ')
#                     logging.info(f'[llm_prompt] >> request to VLLM_URL failed: {e}')
#                     container_vllm_xoo_restart_response = client.containers.get("container_vllm_xoo")
#                     container_vllm_xoo_restart_response.restart()

#                     print(f'[llm_prompt] >> container_vllm_xoo_restart_response: {container_vllm_xoo_restart_response}')
#                     logging.info(f'[llm_prompt] >> container_vllm_xoo_restart_response: {container_vllm_xoo_restart_response}')

#                     time.sleep(20)



#                 if response and response.status_code == 200:       
#                     print(f'[llm_prompt] >> got response == 200 ... {response}')
#                     logging.info(f'[llm_prompt] >> got response == 200 ... {response}')
#                 else:



#                     print(f'[llm_prompt] >> got response != 200 ... restart video ...')
#                     logging.info(f'[llm_prompt] >> got response != 200 ... restart video ...')


#                     req_container = client.containers.get("container_video")
#                     req_container.restart()


#                     response = requests.post(VLLM_URL, json={
#                         "method": "status"
#                     }, timeout=600)

#                     print(f'[llm_prompt] vllm status response: {response} ...')
#                     logging.info(f'[llm_prompt] vllm status response: {response} ...')





#                     print(f'  * ! * ! *  stopping all vllm containers ....')
#                     res_stop_vllm_container = await stop_vllm_container()
#                     print(f'  * ! * ! *  stopping all vllm containers stopped! -> {res_stop_vllm_container}')

                    
#                     print(f' * ! * ! * starting requested container .... ({"container_vllm_xoo"}) .....  ')
#                     req_container = client.containers.get("container_vllm_xoo")
#                     print(f' * ! * ! * found   req_container: {req_container} ')
#                     print(f'  * ! * ! *  starting container ({"container_vllm_xoo"}) again ...')
#                     req_container.start()
#                     print(f'  * ! * ! *  req_container started @ {datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}')
#                     print(f'  * ! * ! *  checking if status running')













#                     # checking if container has been started 
#                     for i in range(5):
#                         print(f'  * ! * ! *  i -> {i}')   
#                         print(f'  * ! * ! *  reloading ....')    
#                         req_container.reload()
#                         print(f'  * ! * ! *  reloaded!')
#                         print(f'  * ! * ! *  req_container.status -> {req_container.status}')  
#                         if req_container.status == 'running':                        
#                             print(f'  * ! * ! *  SIIIIIIIIIIIIIIIII containorador esta corriendoo! -> breeak')
#                             break
#                         else:
#                             print(f'  * ! * ! *  NOOOOOOOOO EL CONTANIDOR NO ESTA CORRIENDO !!! VAMO ZZZZZZZZZ zzzzzzzz 5 sec')
#                             time.sleep(5)
                        
#                     print(f'  * ! * ! *  VALE VALE Q PASA ....')  
#                     container_name_test1 = req_container.attrs['Name']
#                     print(f'  * ! * ! *  container_name_test1 -> {container_name_test1}')   
#                     container_name_test2 = req_container.attrs.get('Name', 'DefaultName')  
#                     print(f'  * ! * ! *  container_name_test2 -> {container_name_test2}')

#                     VLLM_URL = f'http://container_vllm_xoo:{os.getenv("VLLM_PORT")}'
#                     print(f'  * ! * ! * VLLM_URL -> {VLLM_URL} .... checking /status .....')
#                     try:
#                         res_status = requests.get(f'{VLLM_URL}/status', timeout=10)
#                         print(f'  * ! * ! * res_status -> {res_status}')
#                     except Exception as e:
#                         print(f'  * ! * ! * Error checking status: {str(e)}')
                        
#                     print(f'  * ! * ! * safety sleep 20 sec')
#                     time.sleep(20)
#                     print(f'  * ! * ! * awake!')

                    
#                     try:
#                         response = requests.post(f'{VLLM_URL}/vllm', json={
#                             "req_type":"load",
#                             "model":"Qwen/Qwen2.5-1.5B-Instruct",
#                             "tensor_parallel_size":1,
#                             "gpu_memory_utilization":0.7,
#                             "max_model_len":4096
#                         })
                        
#                         print(f' * ! * ! * trying to load ....  3 response {response}')
                        
#                         if response.status_code == 200:
#                             print(f' * ! * ! * trying to load ....  4 status_code: {response.status_code}')
                            
#                             response_json = response.json()
#                             print(f' * ! * ! * trying to load ....  5 response_json: {response_json}')
#                             print(f' * ! * ! * trying to load ....  6 response_json["result_data"]: {response_json["result_data"]}')
#                             return JSONResponse({"result_status": 200, "result_data": f'{response_json["result_data"]}'})
#                         else:
#                             print(f' * ! * ! * trying to load .... 7 ERRRRR')
#                             return JSONResponse({"result_status": 520, "result_data": f'ERRRRRR'})
                
#                     except Exception as e:
#                             print(f' * ! * ! * trying to load .... 8 ERRRRR')
#                             return JSONResponse({"result_status": 520, "result_data": f'ERRRRRR 8'}) 
                            
#                 VLLM_URL = f'http://container_vllm_xoo:{os.getenv("VLLM_PORT")}/vllm'
#                 print(f'trying request vllm with da URL: {VLLM_URL}')
#                 try:
#                     response = requests.post(VLLM_URL, json={
#                         "req_type":"generate",
#                         "prompt":req_data["prompt"],
#                         "temperature":0.8,
#                         "top_p":0.95,
#                         "max_tokens":250
#                     })
#                     if response.status_code == 200:
#                         logging.info(f' [docker]  status_code: {response.status_code}') 
#                         response_json = response.json()
#                         logging.info(f' [docker]  response_json: {response_json}') 
#                         response_json["result_data"] = response_json["result_data"]
#                         return JSONResponse({"result_status": 200, "result_data": f'{response_json["result_data"]}'})
#                     else:
#                         logging.info(f' [docker] response: {response}')
#                         return JSONResponse({"result_status": 520, "result_data": f'ERRRR response.status_code {response.status_code} response{response}'})
                
#                 except Exception as e:
#                     print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
#                     return JSONResponse({"result_status": 520, "result_data": f'err {str(e)}'})


#             except Exception as e:
#                 print(f' * ! * ! * trying to load .... 9 ERRRRR')
#                 return JSONResponse({"result_status": 599, "result_data": f'ERRRRRR 8'})








                
#                 print(f'[llm_prompt] >> got response != 200 ... stopping video ...')
#                 logging.info(f'[llm_prompt] >> got response != 200 ... stopping video ...')
#                 container_video_stop_response = requests.post(BACKEND_URL, json={
#                     "method":"restart",
#                     "model":"container_video"
#                 }, timeout=600)
#                 print(f'[llm_prompt] >> container_video_stop_response: {container_video_stop_response}')
#                 logging.info(f'[llm_prompt] >> container_video_stop_response: {container_video_stop_response}')


#                 response = requests.post(VLLM_URL, json={
#                     "method": "status"
#                 }, timeout=600)

#             print(f'[llm_prompt] vllm status response: {response} ...')
#             logging.info(f'[llm_prompt] vllm status response: {response} ...')

        
#         if req_data["method"] == "image":
            
#             try:


#                 print(f'[image_generate] starting ...')
#                 logging.info(f'[image_generate] starting ...')
#                 container_vllm_xoo = client.containers.get("container_vllm_xoo")
#                 container_vllm_xoo.stop()
#                 print(f'[image_generate] xoo stopped! ...')
#                 logging.info(f'[image_generate] xoo stopped! ...')


#                 print(f'[image_generate] getting status ... ')
#                 logging.info(f'[image_generate] getting status ... ')
                
#                 response = requests.post(IMAGE_URL, json={
#                     "method": "status"
#                 }, timeout=600)

#                 if response.status_code == 200:          
#                     print(f'[image_generate] >> got response == 200 ... building json ... {response}')
#                     logging.info(f'[image_generate] >> got response == 200 ... building json ... {response}')
#                     res_json = response.json()    
#                     print(f'[image_generate] >> got res_json ... {res_json}')
#                     logging.info(f'[image_generate] >> got res_json ... {res_json}')

#                     if res_json["result_data"] == "ok":
#                         print(f'[image_generate] >> status: "ok" ... starting to generate image .... ')
#                         logging.info(f'[image_generate] >> status: "ok" ... starting to generate image .... ')
            
#                         response = requests.post(IMAGE_URL, json={
#                             "method": "generate_image",
#                             "image_model": "stabilityai/stable-diffusion-2",
#                             "prompt": req_data["prompt"],
#                             "image_device": "cuda",
#                             "image_compute_type": "torch.float16"
#                         })

#                         print(f'[image_generate] >> got response #22222 == 200 ... building json ... {response}')
#                         logging.info(f'[image_generate] >> got response #22222 == 200 ... building json ... {response}')
                        
#                         res_json = response.json()
        
#                         print(f'[image_generate] >> #22222 got res_json ... {res_json}')
#                         logging.info(f'[image_generate] >> #22222 got res_json ... {res_json}')   
#                         print(f'[image_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
#                         logging.info(f'[image_generate] >> #22222 got res_json["result_data"]["output_path"] ... {res_json["result_data"]["output_path"]}')
                        
#                         if res_json["result_status"] == 200:
#                             return f'{res_json["result_data"]["output_path"]}'
#                         else: 
#                             return f'{DEFAULT_IMAGE}'
#                     else:
#                         print('[image_generate] ERROR IMAGE SERVER DOWN!?')
#                         logging.info('[image_generate] ERROR IMAGE SERVER DOWN!?')
#                         return f'{DEFAULT_IMAGE}'

#             except Exception as e:
#                 return f'Error: {e}'

#     except Exception as e:
#         print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
#         return JSONResponse({"result_status": 520, "result_data": f'{e}'})

@app.post("/docker")
async def fndocker(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] req_data > {req_data}')

        if req_data["method"] == "generate":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] generate >>>>>>>>>>>')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] generate >>>>>>>>>>> ')

            if req_data["vllmcontainer"] == "container_vllm_oai":
                VLLM_URL = f'http://{req_data["vllmcontainer"]}:{req_data["port"]}/v1/chat/completions'
                print(f'trying request vllm with da URL: {VLLM_URL}')
                try:
                    response = requests.post(VLLM_URL, json={
                        "model":req_data["model"],
                        "messages": [
                                        {
                                            "role": "user",
                                            "content": f'{req_data["prompt"]}'
                                        }
                        ]
                    })
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] response: {response}')
                    logging.info(f' [docker]  response: {response}') 
                    if response.status_code == 200:
                        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] status_code: {response.status_code}')
                        logging.info(f' [docker]  status_code: {response.status_code}') 
                        
                        
                        response_json = response.json()
                        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] response_json: {response_json}')
                        logging.info(f' [docker]  response_json: {response_json}') 
                        
                        
                        message_content = response_json["choices"][0]["message"]["content"]
                        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [docker] message_content: {message_content}')
                        logging.info(f' [docker]  message_content: {message_content}') 
                        
                        return JSONResponse({"result_status": 200, "result_data": f'{message_content}'})              
                    else:
                        logging.info(f' [docker] response: {response}')
                        return JSONResponse({"result_status": 520, "result_data": f'ERRRR response.status_code {response.status_code} response{response}'})
                
                except Exception as e:
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                    return f'err {str(e)}'
                
                
            if req_data["vllmcontainer"] == "container_vllm_xoo": 
                VLLM_URL = f'http://{req_data["vllmcontainer"]}:{req_data["port"]}/vllm'
                print(f'trying request vllm with da URL: {VLLM_URL}')
                try:
                    response = requests.post(VLLM_URL, json={
                        "req_type":"generate",
                        "prompt":req_data["prompt"],
                        "temperature":float(req_data["temperature"]),
                        "top_p":float(req_data["top_p"]),
                        "max_tokens":int(req_data["max_tokens"])
                    })
                    if response.status_code == 200:
                        logging.info(f' [docker]  status_code: {response.status_code}') 
                        response_json = response.json()
                        logging.info(f' [docker]  response_json: {response_json}') 
                        response_json["result_data"] = response_json["result_data"]
                        return JSONResponse({"result_status": 200, "result_data": f'{response_json["result_data"]}'})
                    else:
                        logging.info(f' [docker] response: {response}')
                        return JSONResponse({"result_status": 520, "result_data": f'ERRRR response.status_code {response.status_code} response{response}'})
                
                except Exception as e:
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                    return JSONResponse({"result_status": 520, "result_data": f'err {str(e)}'})
            
            return JSONResponse({"result_status": 404, "result_data": f'{req_data["vllmcontainer"]} not found!'})
  
        if req_data["method"] == "logs":
            req_container = client.containers.get(req_data["model"])
            res_logs = req_container.logs()
            res_logs_str = res_logs.decode('utf-8')
            reversed_logs = "\n".join(res_logs_str.splitlines()[::-1])
            return JSONResponse({"result": 200, "result_data": reversed_logs})

        if req_data["method"] == "network":
            req_container = client.containers.get(req_data["container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result": 200, "result_data": stats})

        if req_data["method"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        if req_data["method"] == "delete":
            req_container = client.containers.get(req_data["model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result": 200})

        if req_data["method"] == "stop":

            print(f' <>  <>  <>  <>  <>  trying to stop ....  ')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [stop] <>  <>  <>  <>  <>  trying to stop ....  ')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [stop] req_data["model"]: {req_data["model"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [stop] req_data["model"]: {req_data["model"]}')

            req_container = client.containers.get(req_data["model"])
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [stop] req_container: {req_container}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [stop] req_container: {req_container}')

            req_container.stop()
            
            print(f' <>  <>  <>  <>  stopped! ....  ')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [stop] <>  <>  <>  <>  <>  stopped! ....  ')


            return JSONResponse({"result": 200})

        if req_data["method"] == "start":
            req_container = client.containers.get(req_data["model"])
            req_container.start()
            return JSONResponse({"result": 200})

        if req_data["method"] == "restart":
            req_container = client.containers.get(req_data["model"])
            req_container.restart()
            return JSONResponse({"result": 200})

        if req_data["method"] == "load":
            try:
                print(f' * ! * ! * trying to load ....  0 ')
                print(f' * ! * ! * trying to load ....  req_data: {req_data} ')
                print(f' * ! * ! * trying to load ....  req_data["method"]: {req_data["method"]} ')
                print(f' * ! * ! * trying to load ....  req_data["vllmcontainer"]: {req_data["vllmcontainer"]} ')
                print(f' * ! * ! * trying to load ....  req_data["image"]: {req_data["image"]} ')
                print(f' * ! * ! * trying to load ....  req_data["port"]: {req_data["port"]} ')
                print(f' * ! * ! * trying to load ....  req_data["model"]: {req_data["model"]} ')
                print(f' * ! * ! * trying to load ....  req_data["tensor_parallel_size"]: {req_data["tensor_parallel_size"]} ')
                print(f' * ! * ! * trying to load ....  req_data["gpu_memory_utilization"]: {req_data["gpu_memory_utilization"]} ')
                print(f' * ! * ! * trying to load ....  req_data["max_model_len"]: {req_data["max_model_len"]} ')

                print(f'  * ! * ! *  stopping all vllm containers ....')
                res_stop_vllm_container = await stop_vllm_container()
                print(f'  * ! * ! *  stopping all vllm containers stopped! -> {res_stop_vllm_container}')

                
                print(f' * ! * ! * starting requested container .... ({req_data["vllmcontainer"]}) .....  ')
                req_container = client.containers.get(req_data["vllmcontainer"])
                print(f' * ! * ! * found   req_container: {req_container} ')
                print(f'  * ! * ! *  starting container ({req_data["vllmcontainer"]}) again ...')
                req_container.start()
                print(f'  * ! * ! *  req_container started @ {datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}')
                print(f'  * ! * ! *  checking if status running')
                
                # checking if container has been started 
                for i in range(5):
                    print(f'  * ! * ! *  i -> {i}')   
                    print(f'  * ! * ! *  reloading ....')    
                    req_container.reload()
                    print(f'  * ! * ! *  reloaded!')
                    print(f'  * ! * ! *  req_container.status -> {req_container.status}')  
                    if req_container.status == 'running':                        
                        print(f'  * ! * ! *  SIIIIIIIIIIIIIIIII containorador esta corriendoo! -> breeak')
                        break
                    else:
                        print(f'  * ! * ! *  NOOOOOOOOO EL CONTANIDOR NO ESTA CORRIENDO !!! VAMO ZZZZZZZZZ zzzzzzzz 5 sec')
                        time.sleep(5)
                    
                print(f'  * ! * ! *  VALE VALE Q PASA ....')  
                container_name_test1 = req_container.attrs['Name']
                print(f'  * ! * ! *  container_name_test1 -> {container_name_test1}')   
                container_name_test2 = req_container.attrs.get('Name', 'DefaultName')  
                print(f'  * ! * ! *  container_name_test2 -> {container_name_test2}')

                VLLM_URL = f'http://{req_data["vllmcontainer"]}:{req_data["port"]}'
                print(f'  * ! * ! * VLLM_URL -> {VLLM_URL} .... checking /status .....')
                try:
                    res_status = requests.get(f'{VLLM_URL}/status', timeout=10)
                    print(f'  * ! * ! * res_status -> {res_status}')
                except Exception as e:
                    print(f'  * ! * ! * Error checking status: {str(e)}')
                    
                print(f'  * ! * ! * safety sleep 20 sec')
                time.sleep(20)
                print(f'  * ! * ! * awake!')

                
                try:
                    response = requests.post(f'{VLLM_URL}/vllm', json={
                        "req_type":req_data["method"],
                        "model":req_data["model"],
                        "tensor_parallel_size":req_data["tensor_parallel_size"],
                        "gpu_memory_utilization":req_data["gpu_memory_utilization"],
                        "max_model_len":req_data["max_model_len"]
                    })
                    
                    print(f' * ! * ! * trying to load ....  3 response {response}')
                    
                    if response.status_code == 200:
                        print(f' * ! * ! * trying to load ....  4 status_code: {response.status_code}')
                        
                        response_json = response.json()
                        print(f' * ! * ! * trying to load ....  5 response_json: {response_json}')
                        print(f' * ! * ! * trying to load ....  6 response_json["result_data"]: {response_json["result_data"]}')
                        return JSONResponse({"result_status": 200, "result_data": f'{response_json["result_data"]}'})
                    else:
                        print(f' * ! * ! * trying to load .... 7 ERRRRR')
                        return JSONResponse({"result_status": 520, "result_data": f'ERRRRRR'})
            
                except Exception as e:
                        print(f' * ! * ! * trying to load .... 8 ERRRRR')
                        return JSONResponse({"result_status": 520, "result_data": f'ERRRRRR 8'}) 
                            
            except Exception as e:
                print(f' * ! * ! * trying to load .... 9 ERRRRR')
                return JSONResponse({"result_status": 599, "result_data": f'ERRRRRR 8'})

        
        if req_data["method"] == "create":
            try:
                print(f' >>>>> trying to create ....  0 ')
                print(f' >>>>> trying to create ....  req_data: {req_data} ')
                print(f' >>>>> trying to create ....  req_data["method"]: {req_data["method"]} ')
                print(f' >>>>> trying to create ....  req_data["image"]: {req_data["image"]} ')
                print(f' >>>>> trying to create ....  req_data["runtime"]: {req_data["runtime"]} ')
                print(f' >>>>> trying to create ....  req_data["shm_size"]: {req_data["shm_size"]} ')
                print(f' >>>>> trying to create ....  req_data["port"]: {req_data["port"]} ')
                print(f' >>>>> trying to create ....  req_data["max_model_len"]: {req_data["max_model_len"]} ')
                print(f' >>>>> trying to create ....  req_data["tensor_parallel_size"]: {req_data["tensor_parallel_size"]} ')
                print(f' >>>>> trying to create ....  req_data["gpu_memory_utilization"]: {req_data["gpu_memory_utilization"]} ')

                print(f'  >>>>>  stopping all vllm containers ....')
                res_stop_vllm_container = await stop_vllm_container()
                print(f'  >>>>>  stopping all vllm containers stopped! -> {res_stop_vllm_container}')

                
                if req_data["image"] == "vllm/vllm-openai:latest":
                    print(f' !!!!! create found "vllm/vllm-openai:latest" !')
                    res_container = client.containers.run(
                        build={"context": f'./{req_data["container_name"]}'},
                        image=req_data["image"],
                        runtime=req_data["runtime"],
                        ports={
                            f'{req_data["port"]}/tcp': ("0.0.0.0", req_data["port"])
                        },
                        container_name=f'{req_data["container_name"]}',
                        volumes={
                            "/logs": {"bind": "/logs", "mode": "rw"},
                            "/home/cloud/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"},
                            "/models": {"bind": "/root/.cache/huggingface/hub", "mode": "rw"}
                        },
                        shm_size=f'{req_data["shm_size"]}',
                        network=network_name,
                        environment={
                            "NCCL_DEBUG": "INFO"
                        },
                        command=[
                            f'--model {req_data["model"]}',
                            f'--port {req_data["port"]}',
                            f'--tensor-parallel-size {req_data["tensor_parallel_size"]}',
                            f'--gpu-memory-utilization {req_data["gpu_memory_utilization"]}',
                            f'--max-model-len {req_data["max_model_len"]}'
                        ]
                    )
                    container_id = res_container.id
                    
                    
                    vllm_info = []
                    new_vllm = {
                        "prompt_active": 1,
                        "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                        "name": req_data["container_name"],
                        "image_hf": "Qwen/Qwen2.5-1.5B-Instruct",
                        "image_vllm": req_data["image"],
                        "uid": f'{container_id}',
                        "container_name": req_data["container_name"],
                        "status": "news",
                        "port": req_data["port"],
                        "gpu_list": [0,1],
                        "gpu_names": ["NO GPU"],
                        "gpu_names_str": f'0x NVIDIA GPU',
                        "mem": f'000000',
                        "gpu": f'000000',
                        "temp": f'000000',
                        "max_model_len": 4096,
                        "gpu_memory_utilization": 0.88                
                    }
                    vllm_info.append(new_vllm)
                    pipe.set('fish_key', json.dumps(vllm_info))
                    
        
                    return JSONResponse({"result_status": 200, "result_data": str(container_id)})
                
                if "xoo4foo/" in req_data["image"]:
                    print(f' !!!!! create found "xoo4foo/" !')
                    print(f' !!!!! using req_container_name: {req_data["container_name"]} !')

                    res_container = client.containers.run(
                        image=req_data["image"],
                        name=req_data["container_name"],
                        runtime=req_data["runtime"],
                        shm_size=req_data["shm_size"],
                        network=network_name,
                        detach=True,
                        environment={
                            'NCCL_DEBUG': 'INFO',
                            'VLLM_PORT': req_data["port"]
                        },
                        device_requests=[
                            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                        ],
                        ports={f'{req_data["port"]}': req_data["port"]},
                        volumes={
                            '/logs': {'bind': '/logs', 'mode': 'rw'},
                            '/models': {'bind': '/models', 'mode': 'rw'}
                        },
                        command=[
                            "python", "app.py",
                            "--model", req_data["model"],
                            "--port", str(req_data["port"]),
                            "--tensor-parallel-size", str(req_data["tensor_parallel_size"]),
                            "--gpu-memory-utilization", str(req_data["gpu_memory_utilization"]),
                            "--max-model-len", str(req_data["max_model_len"])
                        ]
                    )
                    

                    container_id = res_container.id
                    
                    vllm_info = []
                    new_vllm = {
                        "prompt_active": 1,
                        "ts": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}',
                        "name": req_data["container_name"],
                        "image_hf": "Qwen/Qwen2.5-1.5B-Instruct",
                        "image_vllm": req_data["image"],
                        "uid": f'{container_id}',
                        "container_name": req_data["container_name"],
                        "status": "news",
                        "port": req_data["port"],
                        "gpu_list": [0,1],
                        "gpu_names": ["NO GPU"],
                        "gpu_names_str": f'0x NVIDIA GPU',
                        "mem": f'000000',
                        "gpu": f'000000',
                        "temp": f'000000',
                        "max_model_len": 4096,
                        "gpu_memory_utilization": 0.88                
                    }
                    vllm_info.append(new_vllm)
                    pipe.set('fish_key', json.dumps(vllm_info))
                    
                    
                    return JSONResponse({"result_status": 200, "result_data": str(container_id)})
                            
            except Exception as e:
                print(f' >>>>> trying to load .... 9 ERRRRR')
                return JSONResponse({"result_status": 599, "result_data": f'ERRRRRR 8'})

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 520, "result_data": f'{e}'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("BACKEND_IP")}', port=int(os.getenv("BACKEND_PORT")))