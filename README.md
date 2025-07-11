# LeafMed Model Server

This is the model server component of LeafMed, providing AI-powered leaf disease detection and classification services through RESTful APIs.

## Features

- Leaf detection using YOLO object detection
- Disease classification using Vision Transformer (ViT)
- Multiple API versions for different use cases
- Base64 image processing
- Cross-Origin Resource Sharing (CORS) support

## Prerequisites

- Python 3.8+
- pip package manager
- Required model files in the `model/` directory:
  - `model.keras`: TensorFlow model for v1 API
  - `yolo11x_leaf.pt`: YOLO model for leaf detection
  - `model-v2/`: Vision Transformer model files

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the [`model`](https://drive.google.com/drive/folders/1C87Jo9auMfdu5NpxlxdCaU00GzL30X91?usp=drive_link) directory and paste it in the root directory

## Environment Variables

Create a `.env` file with the following variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)

## API Endpoints

### 1. V2 Detect Endpoint

**POST** `/v2/detect`

- Detects and crops leaf from the input image
- Request body: `{ "image": "base64_encoded_image" }`
- Response:
  ```json
  {
    "leaf_detected": true,
    "cropped_leaf": "base64_encoded_cropped_image",
    "box": { "x1": 0, "y1": 0, "x2": 100, "y2": 100 }
  }
  ```

### 2. V2 Predict Endpoint

**POST** `/v2/predict`

- Classifies leaf disease using Vision Transformer
- Request body: `{ "image": "base64_encoded_image" }`
- Response:
  ```json
  {
    "label": "disease_classification"
  }
  ```

### 3. V1 Predict Endpoint (Legacy)

**POST** `/v1/predict`

- Legacy disease classification using TensorFlow
- Request body: `{ "image": "base64_encoded_image" }`
- Response:
  ```json
  {
    "index": 0,
    "label": "disease_name",
    "confidence": 0.9999
  }
  ```

## Testing

Use the provided `ping.py` script to test the API endpoints:

```bash
python ping.py
```

The script tests both detection and prediction endpoints using a sample image.

## Model Information

- **YOLO Model**: Used for leaf detection and cropping
- **Vision Transformer**: Primary model for disease classification
- **TensorFlow Model**: Legacy model for basic classification

## Error Handling

The API includes error handling for:

- Missing images in requests
- Invalid base64 encoding
- Image decoding errors
- Model prediction failures

## Development

To run the server in development mode:

```bash
python app.py
```

The server will start with debug mode enabled by default.
