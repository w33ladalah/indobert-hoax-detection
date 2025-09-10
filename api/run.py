import uvicorn
import os

if __name__ == "__main__":
    # Set environment variables if needed
    # os.environ["MODEL_PATH"] = "models/hoax_detection_model"
    # os.environ["PRETRAINED_MODEL"] = "indolem/indobert-base-uncased"

    # Run the FastAPI application
    uvicorn.run("app.main:app", host="0.0.0.0", port=8888, reload=True)
