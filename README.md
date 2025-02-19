## Problem Statement

Develop a CNN-based model for melanoma detection. Melanoma causes 75% of skin cancer deaths but is highly treatable if detected early. Automating image analysis can assist dermatologists and reduce manual effort.

## Table of Contents

- [General Info](#general-information)
- [Model Architecture](#model-architecture)
- [Model Summary](#model-summary)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)
- [Collaborators](#collaborators)

## General Information

The dataset contains 2,357 malignant and benign skin lesion images from the International Skin Imaging Collaboration (ISIC). Images are evenly distributed across categories. The Augmentor Python package was used to balance class representation through data augmentation.

![datasetgraph](./class_distribution.png)

## Skin Cancer Types

![skincancertypes](./skin_cancer_types.png)

The goal is to classify skin cancer types accurately.

## Model Architecture

1. **Data Augmentation**: Enhances dataset diversity via transformations like rotation, scaling, and flipping.
2. **Normalization**: Scales pixel values to [0,1] using `Rescaling(1./255)`.
3. **Convolutional Layers**: Three `Conv2D` layers (16, 32, 64 filters) extract features, each followed by ReLU activation.
4. **Pooling Layers**: `MaxPooling2D` reduces spatial dimensions while preserving key features.
5. **Dropout Layer**: `Dropout(0.2)` prevents overfitting by randomly deactivating neurons.
6. **Flatten Layer**: Converts feature maps into a 1D vector for classification.
7. **Fully Connected Layers**: Two `Dense` layers (128 neurons, ReLU activation) for final predictions.
8. **Output Layer**: Outputs class probabilities based on `target_labels`.
9. **Model Compilation**: Uses Adam optimizer and Sparse Categorical Crossentropy loss for multi-class classification.
10. **Training**: Runs for 50 epochs with `ModelCheckpoint` (saving best model) and `EarlyStopping` (stopping if validation accuracy plateaus).

## Technologies Used

- [Python]
- [Matplotlib]
- [Numpy]
- [Pandas]
- [Seaborn]
- [TensorFlow]

## Collaborators

Created by [@Sagar-Satuluru](https://github.com/Sagar-Satuluru)

