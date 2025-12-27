import cv2
import numpy as np
import pytesseract
import json
import re

class AnswerKeyScanner:
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            try:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            except Exception as e:
                pass

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found")

        # Optimiization: Resize large images to avoid OOM on free tier
        height, width = img.shape[:2]
        if width > 1024:
            scale = 1024 / width
            new_height = int(height * scale)
            img = cv2.resize(img, (1024, new_height), interpolation=cv2.INTER_AREA)

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Adaptive Threshold to get binary image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 2)

        # 3. Grid Line Removal (Critical for Table/Grid OCR)
        # Invert binary image (text becomes white, background black)
        inverted = cv2.bitwise_not(binary)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detect_horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        detect_vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        lines = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0.0)
        
        # Threshold lines back to binary
        (thresh, lines_binary) = cv2.threshold(lines, 127, 255, cv2.THRESH_BINARY)
        
        # Subtract lines from the original inverted image
        # This leaves text + noise, but removes the grid structure
        clean_inverted = cv2.subtract(inverted, lines_binary)
        
        # Invert back to normal (Black text, White background)
        clean_binary = cv2.bitwise_not(clean_inverted)
        
        # 4. Denoise
        # Use a small median blur to remove salt-and-pepper noise from subtraction
        denoised = cv2.medianBlur(clean_binary, 3)

        return denoised

    def run(self, image_path):
        try:
            processed_img = self.preprocess_image(image_path)
            
            # 4. OCR Strategy (Tesseract LSTM)
            # PSM 6: Assume a single uniform block of text (good for grids after line removal)
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDE-'
            
            data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            items = []
            n = len(data['level'])
            for i in range(n):
                txt = data['text'][i].strip()
                if txt:
                    items.append({
                        "text": txt,
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "w": data['width'][i],
                        "h": data['height'][i],
                        "conf": int(data['conf'][i])
                    })
            
            results, confidences = self.parse_items(items)
            
            output = {
                "exam_type": "TYT", 
                "total_questions": len(results),
                "answers": results,
                "confidence": confidences
            }
            return output
            
        except Exception as e:
            return {"error": str(e)}

    def parse_items(self, items):
        # 1. Row Clustering (Y-axis)
        items.sort(key=lambda k: k['y'])
        
        rows = []
        if items:
            curr_row = [items[0]]
            for item in items[1:]:
                center_y_curr = curr_row[-1]['y'] + curr_row[-1]['h']/2
                center_y_item = item['y'] + item['h']/2
                if abs(center_y_item - center_y_curr) < 20: 
                    curr_row.append(item)
                else:
                    rows.append(curr_row)
                    curr_row = [item]
            rows.append(curr_row)

        processed_answers = {}
        processed_confidences = {}

        for row in rows:
            # Sort row by X (Left to Right)
            row.sort(key=lambda k: k['x'])
            
            # Normalize and Classify
            norm_row = []
            for item in row:
                t = item['text'].upper()
                # Remove common noise chars kept by whitelist but invalid
                t = t.replace('|', '').replace('_', '').replace('.', '')
                
                # Check for Combined Format "1-A", "1A", "1 - C"
                # Regex looks for: Digits, optional space/dash, A-E
                match = re.search(r'(\d+)\s*[-]?\s*([ABCDE])', t)
                if match:
                    item['split_q'] = match.group(1)
                    item['split_a'] = match.group(2)
                    item['type'] = 'COMBINED'
                    norm_row.append(item)
                    continue

                is_digit = t.isdigit()
                is_char = t in ['A', 'B', 'C', 'D', 'E']
                
                if (is_digit or is_char) and len(t) < 4: 
                    item['norm'] = t
                    item['type'] = 'NUM' if is_digit else 'ANS'
                    norm_row.append(item)
            
            # 5. Smart Validation & Pairing
            i = 0
            while i < len(norm_row):
                curr = norm_row[i]
                
                # Case 1: Combined Token "1-C"
                if curr.get('type') == 'COMBINED':
                    q_num = curr['split_q']
                    ans = curr['split_a']
                    conf = curr['conf'] / 100.0
                    
                    if self.is_valid_pair(q_num, processed_answers, processed_confidences, conf):
                         processed_answers[q_num] = ans
                         processed_confidences[q_num] = conf
                    i += 1
                    continue

                # Case 2: Separate Tokens "1" then "C"
                if i < len(norm_row) - 1:
                    next_item = norm_row[i+1]
                    if curr.get('type') == 'NUM' and next_item.get('type') == 'ANS':
                        q_num = curr['norm']
                        ans = next_item['norm']
                        conf = min(curr['conf'], next_item['conf']) / 100.0
                        
                        if self.is_valid_pair(q_num, processed_answers, processed_confidences, conf):
                            processed_answers[q_num] = ans
                            processed_confidences[q_num] = conf
                        
                        i += 2 
                        continue
                
                i += 1
        
        # Sort output by question number
        final_answers = {}
        final_conf = {}
        
        def try_int(k):
            try: return int(k)
            except: return 9999

        sorted_keys = sorted(processed_answers.keys(), key=try_int)
        for k in sorted_keys:
            final_answers[k] = processed_answers[k]
            final_conf[k] = processed_confidences[k]
            
        return final_answers, final_conf
    
    def is_valid_pair(self, q_num, processed_answers, processed_confidences, conf):
        # Update if not exists OR if new confidence is higher
        if q_num not in processed_answers:
            return True
        if conf > processed_confidences[q_num]:
            return True
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
    else:
        scanner = AnswerKeyScanner()
        print(json.dumps(scanner.run(sys.argv[1]), indent=2))
