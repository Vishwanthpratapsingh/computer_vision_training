# Week 2: Image Classification and Transfer Learning

Fine-tuning a pretrained ResNet-18 on a custom dataset, and measuring what data augmentation adds.

> TEMPLATE: replace every `___` and every line in (parentheses) with your own results and words
> after running `week2_deliverable.ipynb`. Delete this note when done.

## Deliverable
- `week2_deliverable.ipynb`: the full notebook (code, curves, ablation table, confusion matrix, write-up), saved with outputs

## Dataset
- 
- Task / domain: ___
- Classes: ___ | Images: ___ | Split: 70% train / 15% val / 15% test (seed 0)
- Notes (class imbalance, image sizes, anything odd): ___

## Method
- Model: ResNet-18, 224x224 input, ImageNet mean/std normalization, Adam, batch size 64
- From scratch: random init, lr 1e-3, 15 epochs
- Frozen backbone (feature extraction): only the new head trains, lr 1e-3, 8 epochs
- Full fine-tuning: backbone lr 1e-4, head lr 1e-3, 8 epochs
- Augmentation (Albumentations): horizontal flip, shift/scale/rotate (up to 15 degrees), colour jitter; training set only
- Best epoch chosen by validation accuracy; test set used once per run at the end

## Results (test accuracy)
| Run | Weights | Trained | Aug | Epochs | Val acc | Test acc |
|---|---|---|---|---|---|---|
| A scratch / no aug | random | all | no | 15 | ___ | ___ |
| B scratch / aug | random | all | yes | 15 | ___ | ___ |
| C frozen / no aug | ImageNet | head only | no | 8 | ___ | ___ |
| D finetune / no aug | ImageNet | all | no | 8 | ___ | ___ |
| E finetune / aug | ImageNet | all | yes | 8 | ___ | ___ |

(Paste the table the notebook prints in section 8 instead of retyping.)

### Augmentation ablation
| Model | No aug | With aug | Effect |
|---|---|---|---|
| From scratch | ___ | ___ | ___ pts |
| Fine-tuned | ___ | ___ | ___ pts |

### Confusion matrix and per-class results
- Selected model (by validation accuracy): ___
- Weakest classes: ___
- Most common confusions: ___ -> ___
- ![confusion matrix](confusion_matrix.png)  (save the plot from the notebook and commit it, or delete this line)

## Success gate
- [ ] Fine-tuned model beats the from-scratch baseline by a clear margin (___ pts)
- [ ] Ablation table shows a measurable augmentation effect (___ pts) with an interpretation below

## Reflection: what moved the metric most
Rank the changes by how much test accuracy each one gained, using your own numbers.

| Rank | Change | Gain | Why I think it helped |
|---|---|---|---|
| 1 | ___ | ___ pts | ___ |
| 2 | ___ | ___ pts | ___ |
| 3 | ___ | ___ pts | ___ |

(Write 4-6 sentences. Things worth covering:
- how much the pretrained weights mattered compared with everything else, and why a small dataset benefits so much
- frozen vs full fine-tuning, and what that says about how close your data is to ImageNet
- whether augmentation helped the scratch model more than the fine-tuned one, and whether the difference is bigger than single-seed noise (about 1 point)
- what happened with a too-high backbone learning rate in run F, if you ran it
- what you would try next and what you would not trust yet, for example only one seed and one backbone)

## Limitations
- Single seed per run, so differences under about 1 point may be noise
- One architecture (ResNet-18) and one dataset
- ___

## How to reproduce
1. Open `week2_deliverable.ipynb` in Google Colab with a T4 GPU runtime.
2. Set `DATA_DIR` to the dataset folder (one sub-folder per class).
3. Run all cells (about 20-30 minutes).
