# ResNet-50 Training and Knowledge Distillation on Animal Dataset

This repository implements a pipeline to train a ResNet-50 on an animal image dataset, evaluate its performance, and distill its knowledge to a smaller MLP model using knowledge distillation techniques.

## Features

- **ResNet-50 Training:** Trains a ResNet-50 classifier on the Animal dataset.
- **Evaluation Metrics:** Measures Top-1, Top-5 accuracy, and F1-score.
- **Knowledge Distillation:** Distills ResNet-50 knowledge into a smaller MLP model using KL Divergence loss.
- **Visualization:** Plots training loss and accuracy for both models.


## Project Structure

```
ASSN3/
├── data/
│   └── Animals_data/animals/animals/  # Dataset folder
├── models/
│   ├── resnet_classifier.pth         # Saved ResNet model
│   ├── mlp_resnet.pth                # Saved distilled MLP model
└── scripts/
    └── main.py                       # Main script for training and distillation
```


## Installation

### Prerequisites

- Python 3.8 or later
- PyTorch
- torchvision
- matplotlib
- tqdm

### Install Dependencies
```bash
pip install -r requirements.txt
```


## Dataset

The Animal dataset should be organized in the following structure:
```
data/
└── Animals_data/
    └── animals/
        └── animals/
            ├── class1/
            │   ├── img1.jpg
            │   └── ...
            ├── class2/
            │   ├── img1.jpg
            │   └── ...
            └── ...
```

---

## Usage

### 1. Train ResNet-50
To train the ResNet-50 model and save it:
```python
TRAIN_RESNET = True
DISTILL_RESNET = False
```

### 2. Distill ResNet-50 to MLP
To distill the ResNet-50 knowledge into an MLP:
```python
DISTILL_RESNET = True
```

### 3. Run the Script
Run the main script:
```bash
python main.py
```


## Key Functions

### Data Preparation
- **`prepare_datasets()`**: Prepares train/test splits with data augmentation.

### Model Training
- **`train_classifier(num_epochs)`**: Trains the ResNet-50 model.
- **`train_student(teacher_model, supervision_loss, temperature, epochs)`**: Trains the student MLP model using knowledge distillation.

### Evaluation
- **`test(model)`**: Evaluates the model and computes Top-1, Top-5 accuracy, and F1-score.

### Visualization
- **`plot_metrics(train_losses, top1_accuracies, top5_accuracies)`**: Plots training loss and accuracy curves.


## Configuration

Modify the following constants in the script to customize training:

```python
NUM_CLASSES = 90              # Number of classes in the dataset
NUM_EPOCHS = 50               # Number of epochs for training
TEMPERATURE = 2.0             # Temperature for distillation
TRAIN_RESNET = False          # Toggle ResNet training
DISTILL_RESNET = True         # Toggle MLP distillation
```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
