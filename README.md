# NLP Restaurant Reviews Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![NLTK](https://img.shields.io/badge/NLTK-3.6-green.svg)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-yellow.svg)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-1.4-orange.svg)](https://pandas.pydata.org/)

## Overview

This project implements Natural Language Processing techniques to analyze restaurant reviews and classify them as positive or negative. Using text preprocessing, bag-of-words modeling, and machine learning classification, the system can automatically determine customer sentiment from written reviews.

## Table of Contents

- [NLP Restaurant Reviews Sentiment Analysis](#nlp-restaurant-reviews-sentiment-analysis)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Features](#features)
  - [Methodology](#methodology)
  - [Text Preprocessing](#text-preprocessing)
  - [Model Implementation](#model-implementation)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dependencies](#dependencies)

## Introduction

Understanding customer sentiment is crucial for businesses in the restaurant industry. This project leverages Natural Language Processing (NLP) techniques to automatically analyze and classify customer reviews as positive (liked) or negative (disliked). By processing textual data, the system can identify patterns and sentiments in customer feedback that would be time-consuming to analyze manually.

## Dataset

The project uses the "Restaurant_Reviews.tsv" dataset, which contains:
- 1000 restaurant reviews
- Binary classification labels (1 for positive, 0 for negative)
- Tab-separated values format

Sample reviews from the dataset:
- "Wow... Loved this place." (Positive)
- "Crust is not good." (Negative)
- "Not tasty and the texture was just nasty." (Negative)

## Features

- Text preprocessing pipeline for cleaning and preparing review text
- Removal of stopwords and non-alphabetic characters
- Word stemming to reduce words to their root form
- Bag-of-Words model for feature extraction
- Machine learning model for sentiment classification
- Train-test split evaluation methodology

## Methodology

The project follows these key steps:

1. **Data Loading**: Import the restaurant reviews dataset
2. **Text Preprocessing**: Clean and prepare the text data
3. **Feature Extraction**: Convert text to numerical features using Bag-of-Words
4. **Model Training**: Train a machine learning classifier on the processed data
5. **Evaluation**: Assess the model's performance on test data

## Text Preprocessing

The text preprocessing pipeline includes several important steps:

1. **Cleaning**: 
   - Removing non-alphabetic characters using regular expressions
   - Converting all text to lowercase
   - Splitting text into individual words

2. **Stopword Removal**:
   - Filtering out common English words (e.g., "the", "a", "an") that don't contribute to sentiment
   - Using NLTK's stopwords corpus

3. **Stemming**:
   - Reducing words to their root form using Porter Stemmer
   - Example: "loving", "loved", "loves" → "love"

4. **Rejoining**:
   - Combining the processed words back into a single text string for feature extraction

## Model Implementation

The project implements the following components:

1. **Feature Extraction**:
   - Using CountVectorizer to create a Bag-of-Words model
   - Setting a maximum of 1500 features to keep dimensionality manageable
   - Converting text data into numerical feature vectors

2. **Train-Test Split**:
   - Dividing the dataset into 80% training and 20% testing data
   - Using a fixed random seed for reproducibility

3. **Classification Model**:
   - The code is set up to implement machine learning classification
   - Note: The classification model training and evaluation steps are not shown in the provided code snippet

## Installation

1. Clone this repository
```bash
git clone https://github.com/cnosmn/restaurant-project-NLP.git
cd restaurant-project-NLP
```

2. Install required packages
```bash
pip install pandas numpy matplotlib scikit-learn nltk
```

3. Download necessary NLTK data
```python
import nltk
nltk.download("stopwords")
```

## Usage

1. Place your restaurant reviews dataset in the project directory
2. Run the preprocessing and model training script
```bash
jupyter sentiment_analysis.ipynb
```

3. For analyzing new reviews, you can use the trained model:
```python
# Example code for prediction (after model is trained)
new_review = "The food was delicious and the service was excellent!"
processed_review = preprocess_text(new_review)
vectorized_review = cv.transform([processed_review]).toarray()
prediction = model.predict(vectorized_review)
print("Positive review" if prediction[0] == 1 else "Negative review")
```

## Dependencies

- Python 3.x
- NLTK: For natural language processing tasks
- scikit-learn: For machine learning models and feature extraction
- Pandas: For data manipulation
- NumPy: For numerical operations
- Matplotlib: For data visualization
- Regular expressions (re): For text cleaning