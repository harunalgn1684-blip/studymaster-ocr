# OMR System & API

This project provides an Optical Mark Recognition (OMR) scanner for answer keys, available both as a standalone script and a REST API.

## Prerequisites
1. **Python 3.x**: Ensure Python is installed.
2. **Tesseract OCR**: You MUST install Tesseract OCR separately.
   - Download: [Tesseract-OCR-w64-setup.exe](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to default location: `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Installation
Open a terminal in this directory (`vision_backend`) and run:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Standalone Script (CLI)
Run the scanner on a single image:
```bash
python scanner.py absolute/path/to/answer_key.jpg
```

### 2. Web API (Server)
Start the API server to allow mobile/web apps to use the scanner:
```bash
python api.py
```
- Server runs on: `http://0.0.0.0:5000`
- **Health Check**: `GET /health`
- **Scan Endpoint**: `POST /scan`
  - Body: `form-data` with key `image` (file).

## API Response Example
```json
{
  "exam_type": "TYT",
  "total_questions": 120,
  "answers": {
    "1": "A",
    "2": "C"
  },
  "confidence": {
    "1": 0.95
  }
}
```

## Troubleshooting
- **Image not found**: Check the path.
- **Tesseract not found**: Edit `tesseract_cmd` in `scanner.py`.
- **Empty result**: Ensure image lighting is even.
