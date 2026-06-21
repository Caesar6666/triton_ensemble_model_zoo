nerdctl run --gpus 1 -v $(pwd):/workspace -it nvcr.io/nvidia/tensorrt:23.01-py3 \
    bash -c \
    "cd /workspace && \
trtexec \
--onnx=yolov26s.onnx \
--minShapes=images:1x3x640x640 \
--optShapes=images:64x3x640x640 \
--maxShapes=images:128x3x640x640 \
--workspace=8192 \
--saveEngine=yolov26s_fp16.plan \
--explicitBatch \
--fp16"