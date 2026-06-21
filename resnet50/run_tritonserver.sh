nerdctl run -d \
  --gpus 1 \
  --name tritonserver \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tritonserver:23.01-py3-v0.0.1 \
  CUDA_VISIBLE_DEVICES=1 tritonserver --model-repository=/workspace/models/resnet50 --strict-model-config=false --log-verbose=1 