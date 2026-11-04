# Legacy (pre-2.0) code - reference only

The stack as it was before the 2.0 rebuild (Gradio 5 frontend, custom xoo4foo/zzvllm52
vLLM image, SDXL/SVD video service, faster-whisper on python:slim). Nothing here is
built or run by the new docker-compose.yml. Delete this folder once you no longer need it.

docker-compose.yml, the Dockerfiles and utils/defaults.json were reconstructed verbatim
(content-identical; JSON formatting may differ) because the project is not under git.
TRELLIS (tr) still mounts ./video, so the asset files in video/ stayed in place.
