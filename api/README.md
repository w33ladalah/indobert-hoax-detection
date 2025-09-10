# Hoax Detection API

A FastAPI-based web application for detecting hoaxes in Indonesian text using transformer models (IndoBERT/XLM-R).

## Features

- Text preprocessing (cleaning, stopword removal, stemming)
- Model loading and fine-tuning capabilities
- Model evaluation and persistence
- RESTful API for hoax prediction
- Support for both IndoBERT and XLM-R models

## Project Structure

```bash
api/
├── app/
│   ├── models/
│   │   └── model.py          # Model loading, fine-tuning, and inference
│   ├── preprocessing/
│   │   └── text_processor.py # Text preprocessing utilities
│   ├── routers/
│   │   └── prediction.py     # API endpoints
│   ├── utils/
│   │   └── training.py       # Training and evaluation utilities
│   └── main.py               # FastAPI application
├── requirements.txt          # Project dependencies
└── run.py                    # Application entry point
```

## Installation

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a directory for model storage:

   ```bash
   mkdir -p models/hoax_detection_model
   ```

## Usage

### Running the API

Start the FastAPI application:

```bash
python run.py
```

The API will be available at [http://localhost:8000](http://localhost:8000). You can access the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### API Endpoints

- `GET /`: Welcome message and API information
- `POST /predict`: Predict whether a text is a hoax or not

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Jakarta akan tenggelam pada tahun 2023"}'
```

#### Example Response

```json
{
  "label": "hoax",
  "confidence": 0.92,
  "processed_text": "jakarta tenggelam tahun 2023"
}
```

## Model Training

To train a hoax detection model, you need a labeled dataset. The dataset should be a CSV file with at least two columns: one for the text and one for the label (hoax/not_hoax).

You can use the training utility as follows:

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

print(f"Evaluation results: {results['eval_results']}")
```

## Environment Variables

You can configure the application using the following environment variables:

- `MODEL_PATH`: Path to the fine-tuned model (default: "models/hoax_detection_model")
- `PRETRAINED_MODEL`: Pretrained model to use (default: "indolem/indobert-base-uncased")

## Supported Models

- IndoBERT: `indolem/indobert-base-uncased`
- XLM-RoBERTa: `xlm-roberta-base`

## License

This project is licensed under the MIT License.
