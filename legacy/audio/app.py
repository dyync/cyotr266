import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
from faster_whisper import WhisperModel
import logging



LOG_PATH= './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_audio.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')





current_model = None
def load_audio(req_audio_model,req_audio_device,req_audio_compute_type):
    try:
        global current_model
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] trying to start WhisperModel with req_audio_model: {req_audio_model}')
        if current_model is None:
            current_model = WhisperModel(req_audio_model, device=req_audio_device, compute_type=req_audio_compute_type)
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] [success] WhisperModel started!')
    
    except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] [error] Failed to load WhisperModel')
            raise

def transcribe_audio(audio_model,audio_path,audio_device,audio_compute_type):
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] trying to load WhisperModel audio_model: {audio_model} audio_device: {audio_device} audio_compute_type: {audio_compute_type} audio_path: {audio_path} ...')
        load_audio(audio_model,audio_device,audio_compute_type)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] WhisperModel loaded!')
        
        start_time = time.time()
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] trying to transcribe path: {audio_path} ...')
        segments, info = current_model.transcribe(audio_path)
        full_text = "\n".join([segment.text for segment in segments])
        processing_time = time.time() - start_time
        
        
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] finished transcribing audio from {audio_path}! lang found: {info.language} len text_length: {len(full_text)} in {processing_time:.2f}s ...')
                
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] full_text {full_text}')
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] returning ...')
        
        return f"Detected language: {info.language}\n\n{full_text}\n\nProcessing time: {processing_time:.2f}s"
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] [error]: {e}')
        return f"Error: {e}"


redis_connection = None

def start_redis(req_redis_port):
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Redis started successfully.')
        return r
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Failed to start Redis on port {req_redis_port}: {e}')
        raise







app = FastAPI()
llm_instance = None
             
@app.get("/")
async def root():
    return f'Hello from audio server!'

@app.post("/t")
async def fnaudio(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [audio] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [audio] req_data > {req_data}')

        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": f'ok'})

        if req_data["method"] == "transcribe":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [{req_data["method"]}] trying to transcribe ...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [{req_data["method"]}] trying to transcribe ...')
            result = transcribe_audio(
                    req_data["audio_model"],
                    req_data["audio_path"],
                    req_data["audio_device"],
                    req_data["audio_compute_type"]
            )
            return JSONResponse({"result_status": 200, "result_data": f'{result}'})


    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": f'{e}'})







if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("AUDIO_IP")}', port=int(os.getenv("AUDIO_PORT")))
    

