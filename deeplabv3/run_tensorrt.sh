nerdctl run --gpus 1 -v $(pwd):/workspace -it nvcr.io/nvidia/tensorrt:23.01-py3 \
    bash -c \
    "cd /workspace && \
trtexec \
--onnx=deeplabv3_resnet50_coco.onnx \
--minShapes=input:1x3x512x512 \
--optShapes=input:64x3x512x512 \
--maxShapes=input:128x3x512x512 \
--workspace=8192 \
--saveEngine=deeplabv3_resnet50_coco_fp16.plan \
--explicitBatch \
--fp16"
