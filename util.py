import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Character conversion dictionaries
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path):
    """
    Write results to CSV including toll information.
    """
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,vehicle_type,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score,toll_amount\n')
        
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if all(k in results[frame_nmr][car_id] for k in ('car', 'license_plate', 'toll')):
                    car = results[frame_nmr][car_id]['car']
                    lp = results[frame_nmr][car_id]['license_plate']
                    f.write(f"{frame_nmr},{car_id},"
                           f"[{car['bbox'][0]} {car['bbox'][1]} {car['bbox'][2]} {car['bbox'][3]}],"
                           f"{car['type']},"
                           f"[{lp['bbox'][0]} {lp['bbox'][1]} {lp['bbox'][2]} {lp['bbox'][3]}],"
                           f"{lp['bbox_score']},{lp['text']},{lp['text_score']},"
                           f"{results[frame_nmr][car_id]['toll']}\n")

def license_complies_format(text):
    if len(text) != 7:
        return False
    return ((text[0] in string.ascii_uppercase or text[0] in dict_int_to_char) and
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char) and
            (text[2] in '0123456789' or text[2] in dict_char_to_int) and
            (text[3] in '0123456789' or text[3] in dict_char_to_int) and
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char) and
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char) and
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char))

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 
              5: dict_int_to_char, 6: dict_int_to_char, 2: dict_char_to_int, 
              3: dict_char_to_int}
    for j in range(7):
        license_plate_ += mapping[j].get(text[j], text[j])
    return license_plate_

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for _, text, score in detections:
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, _ = license_plate
    for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]
    return -1, -1, -1, -1, -1