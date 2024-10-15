# Welcome

Welcome to the **Mines x Invent 2024 Data Challenge**! This repository is designed to help Mines Paris students get started quickly with the challenge.

## 🏗️ Setup Instructions

### 1. Clone the repo

```console
git clone https://github.com/LouisStefanuto/hackathon-mines-invent-2024.git
```

### 2. Create a Virtual Environment

First, you'll need to create a Python virtual environment. Make sure you're using Python 3.12:

```console
conda create -n env_challenge python=3.12
conda activate env_challenge
```

> **Note:** For those familiar with Poetry, you can by-pass the Conda step. We add it to simplify the install.

### 3. Install Dependencies

Next, install the required dependencies for the project:

```console
pip install poetry
poetry install --with dev
```

For more information about Poetry, check out their [**documentation**](https://python-poetry.org).

For more information about Poetry, check out their [**documentation**](https://python-poetry.org).

## 📊 Dataset

The challenge is based on the PASTIS dataset from Vivien Sainte Fare Garnot and Loic Landrieu.

The dataset for this challenge is quite large, but for initial experiments, we recommend starting with the mini dataset.

The data is available on Kaggle in the [Data section](https://www.kaggle.com/competitions/data-challenge-invent-mines-2024/data). It contains:

- a **mini Dataset**: 10 samples, to test the starter kit
- and the **full Dataset**: much larger (40Gb), download it at home.

## 🧪 Running the Demo

To get a quick start, check out the `demo.ipynb` notebook. It will guide you through loading and visualizing the dataset, helping you familiarize yourself with the data.

## Training example

We also provide a minimal training pipeline. Note it contains the bare minimal material.

```console
python baseline.train.py
```

## 📚 Documentation

This static documentation website is built using MkDocs and hosted on GitHub Pages. Make sure to document your work thoroughly—your future self will thank you for presenting your project clearly! 😇
