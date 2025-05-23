from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import time

results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Define toll rates based on vehicle class (example rates in USD)
TOLL_RATES = {
    2: 2.00,  # Car
    3: 1.50,  # Motorcycle
    5: 5.00,  # Bus
    7: 7.00   # Truck
}

# Load video
cap = cv2.VideoCapture('./2103099-hd_1280_720_60fps.mp4')
vehicles = [2, 3, 5, 7]  # Vehicle class IDs from COCO dataset

# Get video FPS and calculate frame time
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps
print(f"Video FPS: {fps}, Frame time: {frame_time:.4f}s")

# Read frames
frame_nmr = -1
ret = True
last_time = time.time()

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        vehicle_types = {}  # Store vehicle type for each tracked ID
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                vehicle_types[len(detections_) - 1] = int(class_id)
        
        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        print(f"Frame {frame_nmr}: Detected {len(detections_)} vehicles, Tracking {len(track_ids)} vehicles")

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        print(f"Frame {frame_nmr}: Detected {len(license_plates.boxes)} license plates")

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                # Get vehicle type from tracked IDs
                vehicle_type = next((v_type for i, v_type in vehicle_types.items() 
                                   if track_ids[i][4] == car_id), 2)  # Default to car if not found
                
                # Crop and process license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                # Read license plate
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                # Calculate toll tax
                toll_amount = TOLL_RATES.get(vehicle_type, 2.00)  # Default to car rate if unknown
                
                print(f"Frame {frame_nmr}, Car {car_id}: "
                      f"License: {license_plate_text}, Score: {license_plate_text_score}, "
                      f"Vehicle Type: {vehicle_type}, Toll: ${toll_amount:.2f}")
                
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'type': vehicle_type},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        },
                        'toll': toll_amount
                    }
                    print(f"Frame {frame_nmr}: Saved data for car {car_id}")

        # Display frame
        display_frame = frame.copy()
        for car_id, data in results[frame_nmr].items():
            # Draw car bounding box
            cv2.rectangle(display_frame, 
                         (int(data['car']['bbox'][0]), int(data['car']['bbox'][1])),
                         (int(data['car']['bbox'][2]), int(data['car']['bbox'][3])),
                         (0, 255, 0), 2)
            # Draw license plate box and text
            lp = data['license_plate']
            cv2.rectangle(display_frame,
                         (int(lp['bbox'][0]), int(lp['bbox'][1])),
                         (int(lp['bbox'][2]), int(lp['bbox'][3])),
                         (0, 0, 255), 2)
            cv2.putText(display_frame, 
                       f"{lp['text']} (${data['toll']:.2f})",
                       (int(lp['bbox'][0]), int(lp['bbox'][1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('Vehicle Detection', display_frame)
        
        # Control video speed
        current_time = time.time()
        elapsed_time = current_time - last_time
        delay = max(1, int((frame_time - elapsed_time) * 1000))
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            print("Stopped by user")
            break
        elif key == ord('p'):
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
        
        if frame_nmr >= 3000:
            print("Stopped after processing 600 frames")
            break
        
        last_time = current_time  

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Summary
print("\n=== Processing Summary ===")
print(f"Total frames processed: {frame_nmr + 1}")
frames_with_detections = len([k for k, v in results.items() if v])
print(f"Frames with detections: {frames_with_detections}")
total_cars = sum(len(cars) for cars in results.values())
print(f"Total unique cars detected: {total_cars}")
total_toll = sum(data['toll'] for frame in results.values() for data in frame.values())
print(f"Total toll collected: ${total_toll:.2f}")

# Detailed Results
print("\n=== Detailed Results ===")
for frame, cars in results.items():
    if cars:
        print(f"Frame {frame}:")
        for car_id, data in cars.items():
            print(f"  Car {car_id}:")
            print(f"    Car BBox: {data['car']['bbox']}")
            print(f"    License Plate: {data['license_plate']['text']}")
            print(f"    Toll: ${data['toll']:.2f}")

# Write results to CSV
try:
    write_csv(results, './test.csv')
    print("\nCSV file successfully written to './test.csv'")
except Exception as e:
    print(f"\nError writing CSV: {str(e)}")