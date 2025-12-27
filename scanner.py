import cv2
import numpy as np
import pytesseract
import json
import re

class AnswerKeyScanner:
    def __init__(self, tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'):
        try:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        except Exception as e:
            # We don't print here to keep JSON output clean on stdout
            pass

    def preprocess_image(self, image_path):
        """
        2. PREPROCESSING (Strict Order)
        - Grayscale
        - Adaptive Threshold
        - Median Blur
        - Canny Edge
        - Homography
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found")

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Adaptive Threshold (Gaussian)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

        # 3. Median Blur (Noise Reduction)
        blurred = cv2.medianBlur(thresh, 3)

        # 4. Canny Edge Detection (for layout analysis/homography)
        edged = cv2.Canny(blurred, 75, 200)

        # 5. Perspective Transform (Homography)
        # Find contours on Edge Image to detect the document page
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        doc_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

        if doc_cnt is not None:
            # Warp the original grayscale.
            warped_gray = self.four_point_transform(gray, doc_cnt.reshape(4, 2))
            
            # Re-apply threshold pipeline on warped image for optimal OCR
            warped_thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
            warped_thresh = cv2.medianBlur(warped_thresh, 3)
            
        else:
            # Fallback if no page detected
            warped_thresh = blurred

        return warped_thresh

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def run(self, image_path):
        try:
            processed_img = self.preprocess_image(image_path)
            
            # 4. OCR Strategy (Tesseract LSTM)
            # Whitelist: Numbers + A-E. 
            # Note: Headers (TURKCE etc) will be filtered out by whitelist.
            # This follows "OCR'yi SADECE: Soru numaraları, A/B/C/D/E harfleri üzerinde çalıştır"
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDE'
            
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
                # Replacement map
                t = t.replace('8', 'B').replace('0', 'D').replace('€', 'E').replace('|', 'I')
                
                is_digit = t.isdigit()
                is_char = t in ['A', 'B', 'C', 'D', 'E']
                
                if (is_digit or is_char) and len(t) < 4: 
                    item['norm'] = t
                    item['type'] = 'NUM' if is_digit else 'ANS'
                    norm_row.append(item)
            
            # 5. Smart Validation & Pairing
            i = 0
            while i < len(norm_row) - 1:
                curr = norm_row[i]
                next_item = norm_row[i+1]
                
                # Pair: Num + Ans
                if curr['type'] == 'NUM' and next_item['type'] == 'ANS':
                    q_num = curr['norm']
                    ans = next_item['norm']
                    conf = min(curr['conf'], next_item['conf']) / 100.0
                    
                    # Duplicate Handling: Max 1 answer per question.
                    # "En koyu / en yüksek confidence olanı seç"
                    if q_num in processed_answers:
                        existing_conf = processed_confidences[q_num]
                        if conf > existing_conf:
                            processed_answers[q_num] = ans
                            processed_confidences[q_num] = conf
                    else:
                        processed_answers[q_num] = ans
                        processed_confidences[q_num] = conf
                    
                    i += 2 
                else:
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
    else:
        scanner = AnswerKeyScanner()
        print(json.dumps(scanner.run(sys.argv[1]), indent=2))
