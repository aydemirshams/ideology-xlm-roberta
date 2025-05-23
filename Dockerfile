# Use an official Python runtime with CUDA support for GPU
  FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

  # Install Python and pip, plus curl and gdown dependencies
  RUN apt-get update && apt-get install -y \
      python3 \
      python3-pip \
      curl \
      && rm -rf /var/lib/apt/lists/*

  # Set Python 3 as default
  RUN ln -s /usr/bin/python3 /usr/bin/python

  # Set working directory
  WORKDIR /app

  # Copy requirements and install dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  # Copy the script
  COPY xlm_roberta_ideology.py .

  # Download model weights using gdown
  RUN mkdir -p models && \
      gdown https://drive.google.com/uc?id=1mSWpyI205fhvKW-V4OFn_z4S_5TDqWRe -O models/xlm-roberta-large_best.tar.gz && \
      tar -xzf models/xlm-roberta-large_best.tar.gz -C models/ && \
      rm models/xlm-roberta-large_best.tar.gz

  # Set environment variables for CUDA
  ENV CUDA_VISIBLE_DEVICES=0

  # Entrypoint to run predictions on TIRA's dataset
  ENTRYPOINT ["python", "xlm_roberta_ideology.py", "--data-dir", "$inputDataset", "--output-dir", "$outputDir", "--model-dir", "/app/models", "--model-type", "xlm-roberta-large", "--eval-batch-size", "64", "--max-length", "512", "--predict-only", "--test-file", "dataset.tsv"]