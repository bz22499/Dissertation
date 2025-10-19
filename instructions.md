Evolutionary Programming Repository
This repository contains two evolutionary programming frameworks: Funsearch (Docker/Containerized) and OpenEvolve (Local Virtual Environment). They must be run in isolation.

1. Funsearch (Docker Environment)
Funsearch is designed for a secure, containerized environment.

Prerequisites
Docker desktop installed and running.

Setup and Build
Navigate to the funsearch subdirectory to prepare the environment and build the image.

Bash

# Navigate to the funsearch directory
cd funsearch

# From the funsearch directory:
docker build . -t funsearch

# Inside the Docker container:
docker run -it -v ./data:/workspace/data -v ./examples:/workspace/examples --env-file .env funsearch






2. OpenEvolve (Virtual Environment)

# Navigate to the openevolve directory
cd openevolve

# From the openevolve directory:
.\.venv\Scripts\Activate.ps1                                                                             

# Run the evolution example
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py `
  --config examples/function_minimization/config.yaml `




To run Ollama open it and choose a model
To check Ollama is running: curl http://localhost:11434/v1/models