nerdctl run -d \
  --gpus 1 \
  --name tritonserver \
  -p 127.0.0.1:8000:8000 \
  -v /data01/qyf/models:/models \
  nvcr.io/nvidia/tritonserver:23.01-py3-v0.0.1 \
  CUDA_VISIBLE_DEVICES=0 tritonserver --model-repository=/workspace/models/deeplabv3 --strict-model-config=false --log-verbose=1 