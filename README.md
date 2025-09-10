# IndoBERT Hoax Detection

A machine learning system for detecting hoaxes in Indonesian text using transformer models (IndoBERT/XLM-R).

## Overview

This project implements a hoax detection system specifically designed for Indonesian language content. It leverages state-of-the-art transformer models (IndoBERT and XLM-RoBERTa) to classify text as either legitimate or hoax content.

## Project Structure

```bash
hoax-detection/
├── api/                  # FastAPI application for serving predictions
│   ├── app/              # Application code
│   │   ├── models/       # Model loading and inference
│   │   ├── preprocessing/# Text preprocessing utilities
│   │   ├── routers/      # API endpoints
│   │   ├── utils/        # Training and evaluation utilities
│   │   └── main.py       # FastAPI application
│   ├── requirements.txt  # API dependencies
│   └── run.py            # API entry point
└── README.md             # This file
```

## Features

- **Text Preprocessing**: Cleaning, stopword removal, and stemming for Indonesian text
- **Model Training**: Fine-tuning capabilities for IndoBERT and XLM-R models
- **Model Evaluation**: Performance metrics and evaluation utilities
- **RESTful API**: FastAPI-based web service for hoax prediction
- **Multi-model Support**: Compatible with both IndoBERT and XLM-RoBERTa models

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/w33ladalah/indobert-hoax-detection.git
   cd indobert-hoax-detection
   ```

2. Set up the API:

   ```bash
   cd api
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a directory for model storage:

   ```bash
   mkdir -p models/hoax_detection_model
   ```

### Running the API

Start the FastAPI application:

```bash
python run.py
```

The API will be available at [http://localhost:8888](http://localhost:8888). You can access the interactive API documentation at [http://localhost:8888/docs](http://localhost:8888/docs).

## API Usage

### Endpoints

- `GET /`: Welcome message and API information
- `POST /predict`: Predict whether a text is a hoax or not

### Example Request

```bash
curl -X POST "http://localhost:8888/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Jakarta akan tenggelam pada tahun 2023"}'
```

### Example Response

```json
{
  "label": "hoax",
  "confidence": 0.92,
  "processed_text": "jakarta tenggelam tahun 2023"
}
```

## Model Training

To train a hoax detection model, you need a labeled dataset. The dataset should be a CSV file with at least two columns: one for the text and one for the label (hoax/not_hoax).

```python
from app.utils.training import train_and_evaluate

results = train_and_evaluate(
    data_path="path/to/your/dataset.csv",
    model_path="models/hoax_detection_model",
    pretrained_model="indolem/indobert-base-uncased",  # or "xlm-roberta-base"
    text_column="text",
    label_column="label",
    epochs=3,
    batch_size=8
)
```

## Supported Models

- IndoBERT: `indolem/indobert-base-uncased`
- XLM-RoBERTa: `xlm-roberta-base`

## License

This project is licensed under the MIT License.

## Acknowledgements

- [IndoLEM](https://indolem.github.io/) for the IndoBERT model
- [Hugging Face](https://huggingface.co/) for the transformer libraries
