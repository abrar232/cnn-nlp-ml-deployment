# CNN & NLP ML Deployment Project

End-to-end machine learning deployment project combining a Computer Vision classifier and an NLP ticket triage system, served via FastAPI and Streamlit.

## Projects

### 🌱 CNN Plant Seedling Classifier
- Trained a PyTorch CNN on 12 plant seedling classes achieving 82% test accuracy
- Applied data augmentation to improve generalisation
- Tracked experiments and registered models using MLflow

### 🎫 NLP Customer Support Ticket Triage
- Fine-tuned DistilBERT (HuggingFace Transformers) for 5-class ticket classification
- Implemented Regex NER for entity extraction
- Integrated Claude API to auto-generate draft responses for support agents

## Deployment Stack
- **Backend:** FastAPI with JSON request/response handling
- **Frontend:** Streamlit multi-page app
- **Containerisation:** Docker & docker-compose
- **CI/CD:** GitHub Actions pipeline for automated testing and build
- **Experiment Tracking:** MLflow

## Tech Stack
Python, PyTorch, HuggingFace Transformers, FastAPI, Streamlit, Docker, MLflow, GitHub Actions
