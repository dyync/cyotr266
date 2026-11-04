# Gradio vLLM Hugging Face

Gradio Frontend using vLLM to download and deploy Hugging Face models. CRUD REST API using Docker SDK and Redis.

### Installation

**Prerequisite**: A GPU which supports CUDA 12.4
**Prerequisite**: Some models require a minimum Bfloat16

To install the containers run the docker compose file:

```bash
sudo docker compose up -d
```

then visit the Gradio Frontend at the port specified in the compose file e.g. [http://localhost:7860](http://localhost:7860)