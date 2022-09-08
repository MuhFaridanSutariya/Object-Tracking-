import cv2
from tracker import *

#create tracker object
tracker = EuclideanDistTracker()

def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture("data\object_tracking\highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=120)

while True:
    ret, frame = cap.read()
    frame_resized = rescale_frame(frame, scale=.6)
    height, width, _ = frame_resized.shape
    # Extract region of interest(ROI)
    roi = frame_resized[290:400, 300:474]

    # 1.object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # calculate area and remove noise elements
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)

            detections.append([x,y,w,h])

    # 2.object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.putText(roi, str(id), (x,y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame_resized)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()