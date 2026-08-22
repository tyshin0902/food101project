# Food-101 Image Classification with Custom CNN & Optuna

A computer vision project focused on building an **end-to-end image classification pipeline** using PyTorch, from dataset preprocessing and augmentation to CNN architecture design, hyperparameter optimization, and model evaluation.

Rather than relying on a pretrained model, this project explores the full training workflow with a **custom convolutional neural network** and uses **Optuna** to systematically search for better training and architecture settings.

---

## Project Overview

The goal of this project is to classify food images from the **Food-101 dataset** into one of 101 food categories.

The project was designed as a hands-on study of the complete deep-learning workflow:

1. Load and inspect a real-world image dataset
2. Build separate preprocessing pipelines for training and evaluation
3. Apply data augmentation to improve robustness
4. Design a custom CNN architecture in PyTorch
5. Train and validate the model
6. Use Optuna to optimize important hyperparameters
7. Rebuild the model with the best configuration
8. Evaluate performance on a held-out test set
9. Visualize training loss and validation accuracy

---

## What I Implemented

### 1. Dataset Preparation

I used the Hugging Face version of the **ETH Zürich Food-101 dataset** and created separate datasets for training, validation, and testing.

* Original training split → 80% training / 20% validation
* Original validation split → test dataset
* Fixed random seed (`42`) for reproducible splitting
* Converted all images to RGB before applying transforms
* Built a custom `collate_fn` for PyTorch `DataLoader`

This allowed the data pipeline to work cleanly between Hugging Face `datasets` and PyTorch.

---

### 2. Image Preprocessing & Data Augmentation

I created two different preprocessing pipelines depending on whether an image was being used for training or evaluation.

#### Training pipeline

The training set uses several augmentation techniques:

* Resize to `224 × 224`
* Random crop with reflected padding
* Random horizontal flip
* Color jitter
* Random grayscale conversion
* Random perspective transformation
* Random erasing
* Tensor conversion
* Dataset-specific normalization

These augmentations introduce controlled variation into the training data so the model does not learn only the exact appearance of the original images.

#### Validation / Test pipeline

Validation and test images use only deterministic preprocessing:

* Resize to `224 × 224`
* Convert to tensor
* Normalize

This keeps evaluation consistent while augmentation remains limited to training data.

Normalization values used in the notebook:

```text
Mean: (0.5450, 0.4435, 0.3436)
Std:  (0.2695, 0.2718, 0.2765)
```

I also implemented code for calculating channel-wise mean and standard deviation from the dataset instead of relying on generic normalization statistics.

---

## Custom CNN Architecture

Instead of starting with transfer learning, I built a CNN from scratch to better understand how convolutional image classifiers are constructed.

The network contains two convolutional blocks followed by a classification head.

### Convolutional Block 1

```text
Conv2D
→ Batch Normalization
→ ReLU
→ Max Pooling
→ Dropout
```

### Convolutional Block 2

```text
Conv2D
→ Batch Normalization
→ ReLU
→ Max Pooling
→ Dropout
```

The model then uses adaptive average pooling and fully connected layers to produce predictions for all **101 Food-101 classes**.

Several architectural values are parameterized so that Optuna can modify the network automatically during experimentation.

---

## Hyperparameter Optimization with Optuna

One of the main focuses of this project was moving beyond manually guessing training settings.

I built an Optuna objective function that creates and trains a new CNN for each trial, evaluates it on the validation set, and returns validation accuracy as the optimization target.

### Search Space

| Hyperparameter        | Search Range / Options       |
| --------------------- | ---------------------------- |
| Learning rate         | `1e-4` to `1e-2` (log scale) |
| Optimizer             | Adam / RMSprop / SGD         |
| Dropout rate          | `0.1` to `0.5`               |
| Conv block 1 channels | 16 / 32                      |
| Conv block 2 channels | 32 / 64                      |

Each Optuna trial trains for **3 epochs** to make experimentation faster.

The study is configured to maximize validation accuracy.

### Trial Pruning

I also used Optuna's `MedianPruner`.

Instead of allowing every weak trial to finish, intermediate validation accuracy is reported after each epoch. Trials that are unlikely to become competitive can be stopped early.

This introduced me to an important practical idea in machine learning experimentation: **compute should be spent on promising configurations rather than treating every experiment equally.**

---

## Final Training & Evaluation Pipeline

After Optuna completes the search, the notebook:

1. Retrieves `study.best_params`
2. Reconstructs the CNN using the best architecture settings
3. Selects the best optimizer and learning rate
4. Trains the final model for 10 epochs
5. Records training loss after each epoch
6. Measures validation accuracy after each epoch
7. Evaluates the final network on the held-out test dataset

The notebook also visualizes:

* Training loss vs. epoch
* Validation accuracy vs. epoch

This makes it easier to inspect the model's learning behavior instead of looking only at a final accuracy value.

---

## Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **Hugging Face Datasets**
* **Optuna**
* **NumPy**
* **Matplotlib**

---

## What I Learned

This project helped me connect several machine-learning concepts that are often learned separately into one working pipeline.

### Data matters before modeling

I learned how preprocessing choices can directly affect what a model sees during training. Building separate training and evaluation transforms made the distinction between **augmentation** and **normalization** much clearer.

### CNN architecture is a set of design decisions

Building a CNN from scratch helped me understand the role of convolution, pooling, batch normalization, activation functions, dropout, and the final classifier instead of treating a neural network as a black box.

### Hyperparameter tuning can be treated as an optimization problem

Optuna showed me how architecture and training choices such as learning rate, optimizer, dropout, and channel size can be searched systematically rather than selected only through trial and error.

### Validation must be separated from final testing

Creating dedicated training, validation, and test datasets reinforced the importance of using validation data for model decisions while preserving a separate dataset for final evaluation.

### Efficient experimentation is part of model development

Using short Optuna trials and pruning demonstrated how real model development involves balancing model quality with computational cost.

---

## Project Structure

```text
food101project.ipynb
│
├── Configuration
│   ├── Imports
│   ├── Random seed
│   └── Device selection
│
├── Preprocessing
│   ├── Dataset loading
│   ├── Dataset visualization
│   ├── Data augmentation
│   ├── Normalization
│   ├── Train / validation split
│   └── DataLoaders
│
├── Modeling
│   ├── Custom CNN
│   └── Forward pass
│
├── Hyperparameter Optimization
│   ├── Optuna objective
│   ├── Search space
│   ├── Validation loop
│   └── Trial pruning
│
└── Evaluation
    ├── Best-parameter reconstruction
    ├── Final training
    ├── Test evaluation
    └── Learning-curve visualization
```

---

## Project Status

This notebook documents the full experimental pipeline and is still being iterated on.

One implementation detail currently requires correction before the present CNN definition can be trained end-to-end: the model applies `AdaptiveAvgPool2d((1, 1))`, while the first fully connected layer is currently defined using the pre-pooling `56 × 56 × channels` feature size. The classifier input dimension should be updated to match the pooled tensor before reporting final experimental accuracy.

Because of that, this repository does **not** claim a final benchmark result that has not been produced by the current implementation.

---

## Next Steps

Planned improvements include:

* Correct and validate the final classification head
* Run a larger Optuna study with more trials
* Compare the custom CNN with transfer-learning models
* Add per-class evaluation metrics and a confusion matrix
* Analyze failure cases and misclassified images
* Experiment with deeper architectures and learning-rate scheduling
* Save the best model checkpoint for inference

---

## Why I Built This Project

My main goal was not only to make an image classifier, but to understand the process behind one.

By implementing the dataset pipeline, image augmentation, CNN architecture, training loop, validation process, automated hyperparameter search, pruning, and evaluation myself, I was able to study how the individual components of a computer vision system interact as one complete machine-learning workflow.
