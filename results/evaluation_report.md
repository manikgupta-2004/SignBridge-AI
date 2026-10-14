# SignBridge AI — Model Evaluation & Diagnostic Report

**Date:** 2026-10-11 14:27:18
**Evaluated Model:** `signbridge_efficientnet.keras` (24.00 MB)

## 1. Executive Summary

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **81.06%** |
| **Macro F1-Score** | **0.5341** |
| **Macro Precision** | **0.6046** |
| **Macro Recall** | **0.5323** |
| **Inference Latency (Mean)** | **44.1 ms** |
| **Inference Throughput** | **~22.7 FPS** |
| **Evaluation Test Samples** | **602** |

## 2. Top Confused Sign Pairs

Certain ASL alphabet letters share identical finger arrangements with subtle wrist rotation or thumb placement differences:

| Ground Truth Sign | Mispredicted Sign | Misclassification Count |
| :--- | :--- | :--- |
| **T** | **O** | 20 |
| **Z** | **S** | 20 |
| **U** | **R** | 18 |
| **nothing** | **G** | 11 |
| **W** | **K** | 9 |

## 3. Full Per-Class Classification Report

```
              precision    recall  f1-score   support

           A       1.00      1.00      1.00        32
           B       1.00      1.00      1.00        32
           C       0.00      0.00      0.00         0
           D       1.00      1.00      1.00        32
           E       1.00      1.00      1.00        32
           F       0.00      0.00      0.00         0
           G       0.74      1.00      0.85        32
           H       0.00      0.00      0.00         0
           I       1.00      1.00      1.00        32
           J       0.94      1.00      0.97        32
           K       0.00      0.00      0.00         0
           L       1.00      1.00      1.00        32
           M       0.75      0.75      0.75        32
           N       0.00      0.00      0.00         0
           O       0.53      1.00      0.70        32
           P       0.00      0.00      0.00         0
           Q       1.00      1.00      1.00        32
           R       0.64      1.00      0.78        32
           S       0.00      0.00      0.00         0
           T       1.00      0.12      0.22        32
           U       1.00      0.44      0.61        32
           V       0.00      0.00      0.00         0
           W       1.00      0.72      0.84        32
           X       0.00      0.00      0.00         0
           Y       0.93      0.78      0.85        32
           Z       1.00      0.19      0.32        32
         del       0.00      0.00      0.00         0
     nothing       1.00      0.44      0.61        32
       space       1.00      1.00      1.00        26

    accuracy                           0.81       602
   macro avg       0.60      0.53      0.53       602
weighted avg       0.92      0.81      0.81       602

```

## 4. The Real-World Domain Shift (The Kaggle Gap)

> **Why 99%+ Dataset Accuracy Does NOT Guarantee 99% Webcam Accuracy:**

The Kaggle ASL Alphabet dataset consists of curated hand images captured under relatively uniform studio conditions with consistent camera distances, static backgrounds, and controlled lighting.

When transitioning to a live laptop webcam in the real world, several severe distribution shifts occur:
1. **Background Clutter & Hand Silhouette:** Webcams capture bedroom walls, lamps, curtains, and shirts. Without hand segmentation, edge filters can activate on background objects.
2. **Illumination & Shadow Variability:** Warm tungsten lighting, overhead fluorescent glare, or harsh side-lighting create shadows on finger joints not present in the dataset.
3. **Camera Angle & Focal Perspective:** Laptop webcams are often tilted upward at the user's chin. Finger foreshortening alters relative landmark positions.
4. **Skin Tone & Morphology Variations:** Hand sizes, finger thicknesses, and wrist angles vary across individuals.

**Solution Implemented in SignBridge AI:**
- MediaPipe Hand Detection isolates and crops only the hand region, normalizing background variance.
- Temporal smoothing filters out single-frame confidence dips.
- Calibration tool (`scripts/collect_webcam_data.py`) allows instant collection of personal calibration images for fine-tuning.
