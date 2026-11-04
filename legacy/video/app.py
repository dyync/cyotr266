import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
import uuid
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import logging
import numpy as np
import re
from datetime import datetime
import sys
import cv2
import scipy.spatial
import random
import math
from sklearn.cluster import KMeans

# Alternative video export implementation
try:
    from diffusers.utils import export_to_video
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False
        print("Warning: Neither OpenCV nor imageio are available for video export")

LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_video.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')

current_pipeline = None


# image
current_model = None
# image
def load_model():
    try:
        global current_model
        
        # Load the base model only (no LoRA adapters, avoids PEFT requirement)
        current_model = StableDiffusionXLPipeline.from_pretrained(
            "fluently/Fluently-XL-Final",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
    
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] [success] Model loaded!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] [error] Failed to load model: {e}')
        raise
# image
def generate_image(
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.0,
    randomize_seed: bool = False
):
    """Generate image using the loaded model"""
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image for prompt: {prompt}')
        
        # Load model if not already loaded
        if current_model is None:
            load_model()
        
        # Randomize seed if requested
        # seed = randomize_seed_fn(seed, randomize_seed)
        
        # Handle negative prompt
        if not use_negative_prompt:
            negative_prompt = ""
        
        start_time = time.time()




        with torch.inference_mode():
            image = current_model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=25,
                    num_images_per_prompt=1,
                    cross_attention_kwargs={"scale": 0.65},
                    output_type="pil",
            ).images[0]

        image.save(output_path)
        # save to temp for api
        tmp_output_path = f'/tmp/{output_path}'
        image.save(tmp_output_path)

        processing_time = time.time() - start_time
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] finished generating image! Saved to {output_path} in {processing_time:.2f}s')
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] changing output_path {output_path}')
        output_path = f'/usr/src/app/video/{output_path}'
        # output_path = f'/image/{output_path}'
        # output_path = f'{output_path}'
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] output_path changed! ->  {output_path}')
        return {
            "output_path": output_path,
            "processing_time": f"{processing_time:.2f}s",
            "status": "success"
        }
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] [error]: {e}')
        return {
            "error": str(e),
            "status": "failed"
        }
# image
def prompt_to_filename(prompt, extension="png", add_timestamp=False):
    filename = prompt.lower()
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.strip('_')

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    return f"{filename}.{extension}"



# vip bleistift
def cv2_bleistift(input_img, buntstift, sigma_s=60, sigma_r=0.07, shade_factor=0.05):
    print(f'[/cv2_pencilsketch] input_img: {input_img}')
    print(f'[/cv2_pencilsketch] buntstift      : {buntstift}')
    print(f'[/cv2_pencilsketch] sigma_s      : {sigma_s}')
    print(f'[/cv2_pencilsketch] sigma_r      : {sigma_r}')
    print(f'[/cv2_pencilsketch] shade_factor : {shade_factor}')

    # Read the image
    bgr = cv2.imread(input_img)
    
    if bgr is None:
        raise ValueError(f"Could not read image from {input_img}")
    
    try:
        # Apply pencil sketch
        dst_gray, dst_color = cv2.pencilSketch(bgr, sigma_s=float(sigma_s), sigma_r=float(sigma_r), shade_factor=float(shade_factor))
        
        # Save grayscale sketch
        output_path_gray = f"pencil_sketch_gray_{uuid.uuid4().hex[:8]}.png"
        success_gray = cv2.imwrite(output_path_gray, dst_gray)
        
        # Save color sketch
        output_path_color = f"pencil_sketch_color_{uuid.uuid4().hex[:8]}.png"
        success_color = cv2.imwrite(output_path_color, dst_color)
        
        if not success_gray or not success_color:
            raise RuntimeError(f"Failed to save one or both sketch images")
        
        print(f"Grayscale sketch saved to: {output_path_gray}")
        print(f"Color sketch saved to: {output_path_color}")
        if buntstift == 'Buntstift':
            return f'{output_path_color}'
        return f'{output_path_gray}'
        
    except Exception as e:
        return f'dragon.png'


#vip oil
def cv2_oilpainting(input_img, pinsel, radius):
    print(f'[/cv2_oilpainting] input_img: {input_img}')
    print(f'[/cv2_oilpainting] pinsel      : {pinsel}')
    print(f'[/cv2_oilpainting] radius      : {radius}')

    bgr = cv2.imread(input_img)

    try:
        oil = cv2.xphoto.oilPainting(bgr, size=int(pinsel), dynRatio=int(radius))
    except Exception as e:
        print(f"[warn] xphoto.oilPainting unavailable, using edgePreservingFilter: {e}", file=sys.stderr)
        oil = cv2.edgePreservingFilter(bgr, sigma_s=60, sigma_r=0.4)
    output_path = f"output_{uuid.uuid4().hex[:8]}.png"

    success = cv2.imwrite(output_path, oil)
    if not success:
        raise RuntimeError(f"Failed to save image to {output_path}")

    return output_path


#vip aquarell
def cv2_aquarell(input_img, sigma_s=60, sigma_r=0.45):
    print(f'[/cv2_stylization] input_img: {input_img}')
    print(f'[/cv2_stylization] sigma_s   : {sigma_s}')
    print(f'[/cv2_stylization] sigma_r   : {sigma_r}')

    # Read the input image
    bgr = cv2.imread(input_img)
    
    if bgr is None:
        raise ValueError(f"Could not load image from {input_img}")
    
    print(f'[/cv2_stylization] image shape: {bgr.shape}')

    try:
        # Apply stylization filter
        stylized = cv2.stylization(bgr, sigma_s=float(sigma_s), sigma_r=float(sigma_r))
    except Exception as e:
        print(f"[warn] stylization failed: {e}", file=sys.stderr)
        # Fallback to edge preserving filter
        stylized = cv2.edgePreservingFilter(bgr, flags=1, sigma_s=60, sigma_r=0.4)

    # Generate output path
    output_path = f"stylized_{uuid.uuid4().hex[:8]}.png"

    # Save the result
    success = cv2.imwrite(output_path, stylized)
    if not success:
        return f'dragon.png'

    print(f'[/cv2_stylization] output saved to: {output_path}')
    return output_path








#vip pointillismus
def cv2_point(input_img, primary_colors):
    print(f'[/cv2_pointillism] input_img       : {input_img}')
    print(f'[/cv2_pointillism] primary_colors  : {primary_colors}')

    img = cv2.imread(input_img)
    if img is None:
        raise ValueError(f"Could not read image from {input_img}")

    radius_width = int(math.ceil(max(img.shape) / 1000))

    clt = KMeans(n_clusters=primary_colors)
    clt.fit(img.reshape(-1, 3))
    palette = clt.cluster_centers_

    complements = 255 - palette
    palette = np.vstack((palette, complements))

    r = 3 // 2
    grid = []
    for i in range(0, img.shape[0], 3):
        for j in range(0, img.shape[1], 3):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j
            grid.append((y % img.shape[0], x % img.shape[1]))
    random.shuffle(grid)

    pixel_colors = np.array([img[y, x] for y, x in grid])
    distances = scipy.spatial.distance.cdist(pixel_colors, palette)
    maxima = np.amax(distances, axis=1)
    distances = maxima[:, None] - distances
    summ = np.sum(distances, axis=1)
    distances /= summ[:, None]

    canvas = img.copy()
    for i, (y, x) in enumerate(grid):
        probs = np.argsort(distances[i])
        color = palette[probs[-1]]
        cv2.ellipse(canvas, (x, y), (radius_width, radius_width), 0, 0, 360, color, -1, cv2.LINE_AA)

    output_path = f"pointillism_{uuid.uuid4().hex[:8]}.png"
    success = cv2.imwrite(output_path, canvas)
    if not success:
        raise RuntimeError(f"Failed to save image to {output_path}")

    return output_path











def load_pipeline(model_id, device, torch_dtype, variant):
    try:
        global current_pipeline
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_pipeline] trying to load pipeline: {model_id}')
        
        if current_pipeline is None:
            current_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                variant=variant
            ).to(device)
            
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_pipeline] [success] Pipeline loaded!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_pipeline] [error] Failed to load pipeline: {e}')
        raise

def export_frames_to_video(frames, output_path, fps=10):
    """Export frames to video using available libraries"""
    if HAS_OPENCV:
        # Use the original OpenCV implementation
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
    elif HAS_IMAGEIO:
        # Use imageio as fallback
        print(f"Using imageio to export video to {output_path}")
        # Convert frames to uint8 if needed
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.cpu().numpy() for frame in frames]
        if frames[0].dtype != np.uint8:
            frames = [(frame * 255).astype(np.uint8) for frame in frames]
        
        # Write video using imageio
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        raise ImportError("Neither OpenCV nor imageio are available. Please install one of them:\n"
                         "pip install opencv-python\nor\npip install imageio")

def generate_video(model_id, input_image_path, device, torch_dtype, variant, 
                  decode_chunk_size=8, motion_bucket_id=180, noise_aug_strength=0.1,
                  output_path="olol.mp4", fps=10):  # Changed path
    try:
        global current_model
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to load pipeline: {model_id}')
        load_pipeline(model_id, device, torch_dtype, variant)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] Pipeline loaded!')

        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [input_image_path] input_image_path: {input_image_path}')
        input_image_path_local = input_image_path.split('/')[-1]
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [input_image_path] input_image_path_local: {input_image_path_local} <- local')
        # Load input image
        # input_image = Image.open(input_image_path)
        input_image = Image.open(input_image_path_local)
        
        start_time = time.time()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] generating video from image: {input_image_path_local}')


        # bringt gor nix
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to empty cache ...1')
        current_model.to("cpu")

        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to empty cache ...2')

        current_model = None

        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to empty cache ...3')
        torch.cuda.empty_cache()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] cache cleared!')



        # Generate video frames
        frames = current_pipeline(
            input_image,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength
        ).frames[0]
        
        # Export to video file
        export_frames_to_video(frames, output_path, fps=fps)
        
        processing_time = time.time() - start_time
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] finished generating video! Saved to {output_path} in {processing_time:.2f}s')
        output_path = f'/usr/src/app/video/{output_path}'
        return {
            "output_path": output_path,
            "processing_time": f"{processing_time:.2f}s",
            "status": "success",
            "frames_generated": len(frames)
        }
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] [error]: {e}')
        return {
            "error": str(e),
            "status": "failed"
        }

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

@app.get("/")
async def root():
    return 'Hello from video generation server!'

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = f"./{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    if not image_name.lower().endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")
    return FileResponse(image_path, media_type="image/png")

@app.get("/vipimages/{image_name}")
async def get_image2(image_name: str):
    image_path = f"./{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    if not image_name.lower().endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")
    return FileResponse(image_path, media_type="image/png")

@app.get("/videos/{video_name}")
async def get_video(video_name: str):
    """
    Serve generated MP4 videos.
    Example: /videos/output_svd.mp4
    """
    # video_path = f"/usr/src/app/videos/{video_name}"  # Updated path
    video_path = f"{video_name}"  # Updated path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    if not video_name.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")
    return FileResponse(video_path, media_type="video/mp4")

@app.post("/generateimage")
async def generateimage(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})
            
        if req_data["method"] == "generate_image":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')


            req_image_path = prompt_to_filename(req_data["image_prompt"])

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] req_image_path: {req_image_path}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] req_image_path: {req_image_path}')

            
            result = generate_image(
                req_data["image_prompt"],
                req_image_path
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

@app.post("/generatevip")
async def generatevip(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})
                     
  
        if req_data["method"] == "oil":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] trying to generate oil...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] trying to generate oil...')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_path"]: {req_data["req_path"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_path"]: {req_data["req_path"]}')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_pinsel"]: {req_data["req_pinsel"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_pinsel"]: {req_data["req_pinsel"]}')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_radius"]: {req_data["req_radius"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_radius"]: {req_data["req_radius"]}')

            
            result = cv2_oilpainting(
                req_data["req_path"],
                req_data["req_pinsel"],
                req_data["req_radius"]
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})
                           
        if req_data["method"] == "aquarell":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] trying to generate aquarell...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] trying to generate aquarell...')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] req_data["req_path"]: {req_data["req_path"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] req_data["req_path"]: {req_data["req_path"]}')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] req_data["req_pinsel"]: {req_data["req_pinsel"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] req_data["req_pinsel"]: {req_data["req_pinsel"]}')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] req_data["req_radius"]: {req_data["req_radius"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] req_data["req_radius"]: {req_data["req_radius"]}')

            
            result = cv2_aquarell(
                req_data["req_path"],
                req_data["req_pinsel"],
                req_data["req_radius"]
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})
                                
        if req_data["method"] == "bleistift":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] trying to generate bleistift...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] trying to generate bleistift...')


            
            result = cv2_bleistift(
                req_data["req_path"],
                req_data["req_vip_cv2_bleistift_buntstift"],
                req_data["req_vip_cv2_bleistift_pinsel"],
                req_data["req_vip_cv2_bleistift_radius"],
                req_data["req_vip_cv2_bleistift_schatten"]
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})
                                            
        if req_data["method"] == "point":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] trying to generate point...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] trying to generate point...')

            
            result = cv2_point(
                req_data["req_path"],
                req_data["req_color"]
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generatevip] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

@app.post("/generatevideo")
async def generatevideo(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})
            
        if req_data["method"] == "generate_video":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to generate video...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to generate video...')
            
            result = generate_video(
                req_data["model_id"],
                req_data["input_image_path"],
                req_data["device"],
                eval(req_data["torch_dtype"]),
                req_data.get("variant", "fp16"),
                req_data.get("decode_chunk_size", 8),
                req_data.get("motion_bucket_id", 180),
                req_data.get("noise_aug_strength", 0.1),
                req_data.get("output_path", "output_svd.mp4"),
                req_data.get("fps", 10)
            )
            
            return JSONResponse({"result_status": 200, "result_data": result})
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host=f'{os.getenv("VIDEO_IP")}', port=int(os.getenv("VIDEO_PORT")))