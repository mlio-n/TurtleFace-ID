# TurtleFace ID — Sea Turtle Individual Recognition System

## 🐢 Overview

**TurtleFace ID** identifies individual sea turtles using their post-ocular scute (scale) patterns — unique to each turtle like a fingerprint. It uses a Siamese Network for few-shot learning, so only 1–3 photos per turtle are needed.

## 🏗️ Architecture

```
turtlefaceid/
├── agents/      → IdentificationAgent (pipeline orchestrator)
├── detectors/   → FaceDetector
├── extractors/  → ScuteExtractor
├── models/      → SiameseNetwork + ContrastiveLoss
├── matchers/    → IdentityMatcher
├── database/    → TurtleDatabase (FAISS vector store)
└── utils/       → ImageUtils, Visualizer
```

## 🚀 Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 📚 References

- Jean et al. (2010) — Post-ocular scute uniqueness
- Carter et al. (2014) — TORSOOI database
- Chopra et al. (2005) — Contrastive Loss
- He et al. (2016) — ResNet50
