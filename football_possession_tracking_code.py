import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque
import os

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("footballs.mp4")

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
input_filename = "footballs.mp4"
base_name = os.path.splitext(input_filename)[0]
output_filename = f"{base_name}_annotated.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

print(f"Saving annotated video as: {output_filename}")

id_map = {}
nex_id = 1

trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    res = model.track(frame, classes=[32], persist=True, verbose=False)
    annotated_frame = frame.copy()

    if res[0].boxes is not None and res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            appear[oid] += 1
            
            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = nex_id
                nex_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Fixed putText with all required parameters
                cv2.putText(annotated_frame, f'ID: {sid}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 2)
                
                # Draw center point
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Draw trail
                trail_points = list(trail[oid])
                for i in range(1, len(trail_points)):
                    cv2.line(annotated_frame, trail_points[i-1], trail_points[i], (0, 255, 255), 2)

    # Write the annotated frame to output video
    out.write(annotated_frame)
    
    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       

cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()

print(f"✅ Video saved successfully as: {output_filename}")