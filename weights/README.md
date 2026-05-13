# Model Weights

The compiled NCNN model weights are **not stored in this repository** because they are large binary files.

## Download

Download from the **[GitHub Releases](../../releases)** page for this repository.

| Model | File | Purpose |
|---|---|---|
| Detection | `det_ncnn_model.zip` | YOLOv8n grape cluster detection |
| Segmentation | `seg_ncnn_model.zip` | YOLOv8n-seg stem segmentation |

## After Downloading

```bash
# Extract detection model
unzip det_ncnn_model.zip -d weights/det_ncnn_model/

# Extract segmentation model
unzip seg_ncnn_model.zip -d weights/seg_ncnn_model/
```

The `setup_phoenix.sh` script creates the correct directory structure automatically — run it first.

## Directory Structure (after extraction)

```
weights/
├── det_ncnn_model/
│   ├── model.param
│   └── model.bin
└── seg_ncnn_model/
    ├── model.param
    └── model.bin
```

## Re-training

Training data, annotations, and training scripts are not included in this repository.
Contact the author for access to the training dataset.
