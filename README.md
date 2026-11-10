# Traffic Sign Recognition using CNN

A complete Python/TensorFlow project for PyCharm. It trains a custom
Convolutional Neural Network on the 43-class GTSRB dataset and includes:

- CNN model training
- Automatic training/validation split
- Data augmentation
- Class-imbalance handling
- Best-model checkpointing
- Training graphs and CSV logs
- Single-image prediction
- Tkinter desktop GUI
- Centre-box webcam recognition
- Classification report and confusion matrix

## Important limitation

This is a **traffic-sign classification/recognition** project, not a full
traffic-sign detector. The input should contain one cropped traffic sign.
The webcam script uses a centre box, and the sign must be placed inside it.

To locate signs automatically anywhere in a road scene, a separate object
detector such as YOLO would be required.

## Recommended Python version

Use Python 3.10 or Python 3.11, 64-bit.

## 1. Open in PyCharm

1. Extract this project.
2. In PyCharm, choose **File > Open**.
3. Select the `traffic_sign_cnn_pycharm` folder.
4. Create or select a Python 3.10/3.11 virtual environment.

## 2. Install packages

Open the PyCharm terminal in the project folder:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Test TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 3. Download and extract GTSRB

Download the GTSRB training dataset from the official benchmark page or from a
commonly used GTSRB mirror.

Extract it inside the project's `dataset` folder.

The loader accepts either of these common layouts:

### Common Kaggle layout

```text
dataset/
└── Train/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── ...
    └── 42/
```

### Official GTSRB layout

```text
dataset/
└── GTSRB/
    └── Final_Training/
        └── Images/
            ├── 00000/
            ├── 00001/
            ├── ...
            └── 00042/
```

PPM, PNG, JPG, JPEG, BMP, and WebP images are supported.

## 4. Train the CNN

Run `train.py` in PyCharm, or run:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --epochs 30 --batch-size 64
```

The best model is saved here:

```text
models/traffic_sign_cnn.keras
```

Training results are saved in:

```text
outputs/
├── training_log.csv
├── history.csv
└── training_curves.png
```

## 5. Use the desktop GUI

After training, run:

```bash
python app.py
```

Click **Select Image**, choose a cropped traffic-sign image, and click
**Predict**.

## 6. Predict one image from the terminal

Open a file-selection dialog:

```bash
python predict.py
```

Or pass the image path:

```bash
python predict.py --image "test_images/sign.jpg"
```

## 7. Use the webcam

```bash
python webcam.py
```

Hold or display one traffic sign inside the centre square.

If the wrong camera opens:

```bash
python webcam.py --camera 1
```

Change the confidence threshold:

```bash
python webcam.py --threshold 0.80
```

## 8. Evaluate the trained model

```bash
python evaluate.py
```

This creates:

```text
outputs/
├── classification_report.txt
└── confusion_matrix.png
```

## Common errors

### `Could not find GTSRB training class folders`

The dataset is missing or extracted into the wrong location. Confirm that
`dataset/Train/0` through `dataset/Train/42` exist, or use the official layout
shown above.

### `No module named tensorflow`

PyCharm is using the wrong interpreter, or dependencies were not installed in
the active virtual environment. Select the same interpreter where
`pip install -r requirements.txt` was run.

### TensorFlow does not detect the NVIDIA GPU on native Windows

Modern TensorFlow releases do not support CUDA GPU execution directly on native
Windows. Training still works on the CPU. WSL2 is needed for current official
TensorFlow CUDA support on Windows.

### Poor prediction on a full road photograph

This model was trained as a classifier. Crop the traffic sign tightly before
prediction. A full-scene detector is a different project.
