# Skin Lesion Classification with YOLOv8 Localization and a Dual-View Swin-T + MobileNetV3 Network

A research pipeline for classifying dermoscopic images into the 7 HAM10000 classes. It localizes the lesion, builds two views of it, fuses their features, rejects uncertain cases, assigns a risk level and writes an explainable PDF report.

> **Research prototype. Not a medical device and not a diagnostic tool.** Always consult a dermatologist.

## Pipeline

```
Dermoscopic image
  -> 1. Lesion localization (YOLOv8n, single class "lesion")
  -> 2. Dual-view generation (lesion crop 1.1x box, context crop 1.6x box, 224x224)
  -> 3. Feature extraction (Swin-T on lesion view, MobileNetV3 on context view)
  -> 4. Feature fusion (projection + concatenation + MLP head, 7 classes)
  -> 5. Safety guardrail (confidence and entropy rejection -> "Unknown")
  -> 6. Risk categorization (High / Moderate / Low / Unknown)
  -> 7. Report generation (Grad-CAM, SHAP, branch ablation, PDF)
```

| Module | What it does |
|---|---|
| Localization | YOLOv8n trained on boxes derived from the HAM10000 lesion masks (image size 384) |
| Dual view | Tight lesion crop and wider context crop around the detected box |
| Feature extraction | Swin-T (768-d) for the lesion, MobileNetV3-Large (1280-d) for the context |
| Fusion | Each branch is projected to 512-d, concatenated (1024-d), then a 2-layer MLP head |
| Guardrail | A case is "Unknown" if max probability < 0.60 or normalized entropy > 0.60 |
| Risk | MEL, BCC = High; AKIEC = Moderate; BKL, NV, DF, VASC = Low; Low is raised to Moderate if P(MEL) >= 0.20 |
| Report | Two-page PDF with prediction, risk, Grad-CAM (both branches), SHAP, region summary |

The guardrail is a rejection rule on the model's own confidence and entropy. It is not a separately trained out-of-distribution detector.

## Dataset

[HAM10000](https://doi.org/10.1038/sdata.2018.161) (10,015 images, 7 classes) with the lesion segmentation masks. The data is **not** included in this repository.

- HAM10000 images and metadata: Kaggle `kmader/skin-cancer-mnist-ham10000`
- Lesion segmentation masks: Kaggle "HAM10000 Lesion Segmentations"
- Optional external demo images: PH2 dataset

HAM10000 is licensed CC BY-NC 4.0 (non-commercial use with attribution).

Split: stratified 80/20 (8,012 train / 2,003 validation, seed 42). Class imbalance is handled by class-wise oversampling of the training set (21,565 images after oversampling).

## Setup

```bash
pip install -r requirements.txt
```

A GPU is needed for training. The notebook was developed on Kaggle (Tesla T4).

## How to run

1. Open full_pipeline.ipynb on Kaggle (or Colab) and attach the datasets above.
2. Run the cells in order: setup, data split, YOLO training, crop generation, dual-view training, evaluation, guardrail and triage, report.
3. To generate a report for one image, set IMD063 to the path of a dermoscopic image file (not a folder) and run the report cell. It writes technical_report_IMDO63.pdf.

## Pretrained Weights

- **Dual-View Training (best_model.pth):** [Download best_model.pth](https://huggingface.co/Srinithi09/skin-cancer-detection-model/resolve/main/best_model.pth)

- **YOLOv8 Weights (best.pt):** [Download best.pt](https://huggingface.co/Srinithi09/skin-cancer-detection-model/resolve/main/best.pt)
## Results

Final dual-view model (25 epochs, AdamW with warm-up and cosine schedule, label smoothing 0.1), validation split:

| Accuracy | Macro-F1 | MCC | Macro AUC |
|---|---|---|---|
| 94.56% | 0.913 | 0.895 | 0.982 |

Effect of YOLO localization, same split and the same 8-epoch recipe (MCC):

| Model | Full image | YOLO lesion crop |
|---|---|---|
| Swin-T | 0.8382 | 0.8699 |
| MobileNetV3 | 0.7457 | 0.7874 |
| MobileNetV2 (baseline) | 0.7230 | 0.7425 |
| Swin-T + MobileNetV3 (dual-view) | 0.8278 | 0.859 |

Test-time augmentation (horizontal and vertical flips) did not improve results (accuracy 94.6% with and without), so it is not used.

### Limitations

- **Train/validation overlap.** The split is random by image. About 40% of validation images (791 of 2,003) come from lesions that also appear in the training set, so all absolute numbers are optimistic. The with-YOLO versus without-YOLO comparisons use the same split, so they remain a fair comparison. Re-splitting by `lesion_id` is recommended before reporting final numbers.
- Checkpoints were selected on the validation set, which adds a further small optimism.
- Each result comes from a single run with one seed. Differences of about 0.02 to 0.04 MCC on 2,003 images should be read as indications, not proof.
- DF and VASC have only about 20 to 30 validation images each, so their per-class scores are unstable.
- HAM10000 under-represents darker skin tones, and results may not generalize to other devices or populations.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── full_pipeline.ipynb
└── LICENSE
```
## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

## Licenses and third-party terms

- Code: APACHE-2.0
- Ultralytics YOLOv8 is released under AGPL-3.0. Check its terms before redistributing code or weights derived from it.
- HAM10000: CC BY-NC 4.0. Use is non-commercial.

## Citation

If you use this work, please cite the dataset:

Tschandl, P., Rosendahl, C., Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018).

## Author

Srinithi S, CSE,Mepco Schlenk Engineering college, srinithivishali0912_bcs27@mepcoeng.ac.in
