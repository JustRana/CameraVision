import cv2

# Initialize an empty list for class names
classnames = []
# Recall file thing
classfile = 'files/thing.names'

# Open and read the class file
with open(classfile, 'rt') as a:
    classnames = a.read().rstrip('\n').split('\n')

# Set the paths to the pre-trained model and the configuration file
p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Load the pre-trained model
net = cv2.dnn_DetectionModel(p, v)

# Set input size, scale, mean, and swapRB for the model
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Open the video capture
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()

    
    if ret:
        # Run object detection on the frame
        classIds, confs, bbox = net.detect(frame, confThreshold=0.5)

        # Draw bounding boxes and add labels for detected objects
        if len(classIds) > 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                labelSize, baseLine = cv2.getTextSize(classnames[classId - 1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left = box[0]
                top = max(box[1], labelSize[1] + 10)
                right = left + labelSize[0]
                bottom = top + labelSize[1] - 10
                cv2.rectangle(frame, (left - 1, top - labelSize[1] - 10), (right + 1, bottom + 1), (0, 255, 0),
                              cv2.FILLED)
                cv2.putText(frame, classnames[classId - 1], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1)

        cv2.imshow('Image Recognition', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('CameraVision', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release() 
cv2.destroyAllWindows() 

