# Computer Vision Training

Self-directed program in computer vision and deep learning. One folder per week, each ending in a shipped deliverable with its own README of results.

## Structure
| Folder | Topic | Deliverable |
|---|---|---|
| `week1_foundations/` | ML & CV foundations | CIFAR-10 CNN trained from scratch with logged curves, plus a training-loop explainer |
| `week2_classification/` | Image classification & transfer learning | Fine-tuned ResNet on a custom dataset, augmentation ablation, confusion matrix, write-up |
| `week3_detection/` | Detection, segmentation, tracking & anomaly detection | YOLO detector with mAP, segmentation demo, tracked video, anomaly-detection demo |
| `week4_Docunderstanding/` | Document understanding | Invoice extraction to a common JSON schema (self-hosted OCR/layout vs managed service) with a comparison matrix |
| `week5_vlms/` | Vision-language models | VLM for VQA, explanations and schema-valid extraction (three-way benchmark), plus caption + CLIP retrieval utilities |

## Progress
| Week | Status | Highlights |
|---|---|---|
| 1 | Done | CNN reached ___% test accuracy on CIFAR-10 |
| 2 | In progress | Fine-tuned model ___% vs from-scratch ___% |
| 3 | Not started | |
| 4 | Not started | |
| 5 | Not started | |

(Fill in the numbers when each week is finished.)

## Setup
Run on Google Colab with a T4 GPU, or locally with Python 3.12+:
```
pip install -r requirements.txt
```
Each week's folder has its own README with how to reproduce the results.

## Conventions
- One commit per working day, message format `Day N: what I did`
- Datasets, model weights and experiment logs are not committed (see `.gitignore`)
- Notebooks are saved with their outputs so the results are visible on GitHub
