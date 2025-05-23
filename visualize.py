import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    points = [
        ((x1, y1), (x1, y1 + line_length_y)), ((x1, y1), (x1 + line_length_x, y1)),  # Top-left
        ((x1, y2), (x1, y2 - line_length_y)), ((x1, y2), (x1 + line_length_x, y2)),  # Bottom-left
        ((x2, y1), (x2 - line_length_x, y1)), ((x2, y1), (x2, y1 + line_length_y)),  # Top-right
        ((x2, y2), (x2, y2 - line_length_y)), ((x2, y2), (x2 - line_length_x, y2))   # Bottom-right
    ]
    for pt1, pt2 in points:
        cv2.line(img, pt1, pt2, color, thickness)
    return img

# Load results from interpolated CSVrootsu
results = pd.read_csv('./test_interpolated.csv')

# Load video
video_path = './2103099-hd_1280_720_60fps.mp4'  # Update if your video path is different
cap = cv2.VideoCapture(video_path)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# Dictionary to store the best license plate info for each car
license_plate = {}
for car_id in np.unique(results['car_id']):
    car_data = results[results['car_id'] == car_id]
    max_score = car_data['license_number_score'].astype(float).max()
    best_row = car_data[car_data['license_number_score'].astype(float) == max_score].iloc[0]
    
    if best_row['license_number'] != '0':
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
        ret, frame = cap.read()
        if ret:
            x1, y1, x2, y2 = map(int, map(float, best_row['license_plate_bbox'].split()))
            license_crop = frame[y1:y2, x1:x2, :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
            license_plate[car_id] = {
                'license_crop': license_crop,
                'license_plate_number': best_row['license_number'],
                'toll_amount': float(best_row['toll_amount'])
            }

# Reset video to start
frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Process frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for _, row in df_.iterrows():
            car_id = int(row['car_id'])
            car_x1, car_y1, car_x2, car_y2 = map(int, map(float, row['car_bbox'].split()))
            draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25)

            x1, y1, x2, y2 = map(int, map(float, row['license_plate_bbox'].split()))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

            if car_id in license_plate:
                license_crop = license_plate[car_id]['license_crop']
                H, W, _ = license_crop.shape
                try:
                    # Calculate position for license plate crop and text
                    y_start = max(0, car_y1 - H - 100)  # Ensure it doesn't go above frame
                    y_end = y_start + H
                    if y_end > height:  # If it exceeds frame height, adjust downward
                        y_end = height
                        y_start = y_end - H
                    
                    x_start = int((car_x2 + car_x1 - W) / 2)
                    x_end = x_start + W
                    
                    # Ensure x bounds are within frame
                    x_start = max(0, x_start)
                    x_end = min(width, x_end)
                    
                    # Adjust crop if necessary to fit the adjusted bounds
                    crop_width = x_end - x_start
                    if crop_width > 0 and y_end - y_start == H:
                        frame[y_start:y_end, x_start:x_end, :] = license_crop[:, :crop_width, :]
                    
                    # White background and text below the crop
                    text_y_start = max(0, y_start - 300)  # 300 pixels above crop for text
                    text_y_end = y_start
                    if text_y_end > text_y_start:  # Only draw if there's space
                        frame[text_y_start:text_y_end, x_start:x_end, :] = (255, 255, 255)
                        text = f"{license_plate[car_id]['license_plate_number']} (${license_plate[car_id]['toll_amount']:.2f})"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 8)
                        text_x = max(x_start, int((car_x2 + car_x1 - text_width) / 2))
                        text_y = text_y_start + int((text_y_end - text_y_start + text_height) / 2)
                        if text_x + text_width <= width and text_y - text_height >= 0:
                            cv2.putText(frame, text, (text_x, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 8)
                except Exception as e:
                    print(f"Error displaying license plate for car {car_id} at frame {frame_nmr}: {e}")
                    pass

        out.write(frame)

# Cleanup
out.release()
cap.release()
cv2.destroyAllWindows()
print("Video processing complete. Output saved to './out.mp4'")