# CSIR-NEERI YOLO Agri-Sect Project

## Project Overview
This project is a deep learning-based agricultural pest classification system using YOLO models.  
It compares multiple YOLO variants and evaluates performance using accuracy, precision, recall, F1-score, inference time, and model size.

---

## Models Used
- YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l, YOLOv11x

---

## Dataset
- Custom agricultural insect dataset
- Includes multiple pest and insect classes
- Augmented dataset used for training

Dataset is organized in YOLO classification format with structured image folders used for training, validation, and testing.
---

## Project Structure
- `agri_sect.py` → Training script
- `detect.py` → Inference script
- `data.yaml` → Dataset configuration
- `yolo_dataset/` → Training dataset
- `runs/` → Training logs
- `YOLO_Results/` → Final evaluation graphs

---

## Results Included
- Accuracy comparison
- Precision / Recall / F1-score
- Training vs validation loss
- Confusion matrix
- Model size comparison
- Inference speed comparison
- Class-wise performance graphs

---
## 📸 Sample Output

The model generates predictions for each image in the test set.

### Output includes:
- Predicted class label
- Confidence score
- Top-5 probabilities for classification
- Saved prediction images in output folder

## Installation

```bash
pip install -r requirements_current.txt
```

---

## Run Training
```bash
python agri_sect.py
```

---

## Run Detection
```bash
python detect.py
```

---

## Important Notes
- `.venv/` is excluded from Git (do not upload)
- `.pt` models are included for YOLO experiments
- Large dataset folders are ignored in Git

---

## Author
Kashish Taklikar  
GitHub: https://github.com/taklikarkashish