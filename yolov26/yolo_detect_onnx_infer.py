import cv2
import os
import numpy as np
import onnxruntime


class Yolov26():
    def __init__(self, modelpath, classes=80, size=640, confThreshold=0.25, nmsThreshold=0.5):
        self.net = onnxruntime.InferenceSession(modelpath,providers=['CUDAExecutionProvider'])
        self.confThreshold=confThreshold
        self.nmsThreshold=nmsThreshold
        self.inpWidth = size
        self.inpHeight = self.inpWidth
        self.classes = classes
        self.color = []
        for i in range(classes):
            self.color.append([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def postprocessbox(self, frame, outs, padsize=None):
        
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        newh, neww, padh, padw = padsize
        ratioh, ratiow = frameHeight / newh, frameWidth / neww

        confidences = []
        boxes = []
        classIds = []
        for detection in outs:
            '''与v8、v11最大的不同,detect模型v26输出的是x1,y1,x2,y2,score,class_id'''
            confidence = detection[4]
            classId = detection[5]
            if confidence > self.confThreshold:
                x1 = int( np.ceil( (detection[0] - padw) * ratiow))
                y1 = int( np.ceil( (detection[1] - padh) * ratioh))
                x2 = int( np.ceil( (detection[2] - padw) * ratiow))
                y2 = int( np.ceil( (detection[3] - padh) * ratioh))

                boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                confidences.append(float(confidence))
                classIds.append(classId)
        
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold,self.nmsThreshold)
        box=np.zeros((0,4))
        labels=np.zeros((0,))
        confs=np.zeros((0,))
        if len(idxs)>0:
            box_seq = idxs.flatten()
            box = np.array(boxes)[box_seq]
            labels = np.array(classIds)[box_seq]
            confs = np.array(confidences)[box_seq]
            box[:, 2] += box[:, 0]
            box[:, 3] += box[:, 1]
                
            return box.reshape(-1,4).astype('int32'),labels.astype('int32'),confs
        else:
            return np.array([]),np.array([]),np.array([])
    
    def detect(self,srcimg,crop=False):
        self.srcimg = srcimg
        
        img, newh, neww, padh, padw = self.resize_image(self.srcimg)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)
        out = self.net.run(['output0'], {'images': blob})
        detouts=out[0][0]
        
        boxes,labels,confs = self.postprocessbox(self.srcimg, detouts, padsize=(newh, neww, padh, padw))

        if boxes.shape[0]==0:
            return np.zeros((0,4),dtype=np.int32),np.zeros((0,),dtype=np.int32),np.zeros((0,),dtype=np.float32)
        else:
            return boxes,labels,confs
        
    def draw(self,srcimg,box,label,conf):

        for i in range(box.shape[0]):
            cv2.rectangle(srcimg, (box[i,0], box[i,1]), (box[i,2], box[i,3]), self.color[label[i]], 2)  
            cv2.putText(srcimg,'%d:%.2f'%(label[i],conf[i]), (box[i,0], box[i,1]), cv2.FONT_ITALIC, 1, self.color[label[i]], 1)
        return srcimg


if __name__ == '__main__':
    onnx_path = r'models\yolov26_inference\2\yolo26s.onnx'
    model = Yolov26(onnx_path, classes=80, size=640, confThreshold=0.25, nmsThreshold=0.5)
    image_dir = r'ultralytics/assets'
    save_dir = r'results/onnx'
    os.makedirs(save_dir, exist_ok=True)

    for image_file in os.listdir(image_dir):
        print(image_file)
        if not image_file.endswith(".png") and not image_file.endswith(".jpg") and not image_file.endswith(".jpeg"):
            continue
        image_path = os.path.join(image_dir, image_file)
        srcimg = cv2.imread(image_path)
        boxes,labels,confs = model.detect(srcimg)
        srcimg = model.draw(srcimg,boxes,labels,confs)
        cv2.imwrite(os.path.join(save_dir, image_file), srcimg)
