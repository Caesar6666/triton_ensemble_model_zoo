nerdctl run --gpus 1 -v $(pwd):/workspace -it nvcr.io/nvidia/tensorrt:23.01-py3 \
    bash -c \
    "cd /workspace && \
trtexec \
--onnx=resnet50_imagenet_v2.onnx \
--minShapes=input:1x3x224x224 \
--optShapes=input:256x3x224x224 \
--maxShapes=input:512x3x224x224 \
--workspace=8192 \
--saveEngine=resnet50_imagenet_v2_fp16.plan \
--explicitBatch \
--fp16"
