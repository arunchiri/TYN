# Titanic Yield Network (TYN)

A hierarchical neural network for predicting Titanic passenger survival.

## Overview

TYN (Titanic Yield Network) is a deep learning model that combines hierarchical reasoning with adaptive computation to predict passenger survival on the Titanic.

## Features

- Hierarchical two-level neural architecture
- Adaptive computation time with early stopping
- Efficient learning on limited data
- Support for passenger features: Age, Sex, Class, Fare, Family Size, and more

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Place the following files in the project root:
- `train.csv` - Training data with survival labels
- `test.csv` - Test data for predictions
- `gender_submission.csv` - Sample submission format

## Training

```bash
python train.py
```

## Evaluation & Submission

```bash
python evaluate.py
```

## Project Structure

```
TYN/
├── models/              # Model architecture files
├── config/              # Configuration files
├── dataset/             # Data processing scripts
├── utils/               # Utility functions
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Results

Model predictions are saved to `submission.csv`

## License

MIT License
