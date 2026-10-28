# Fake_News_Detection_MS24011

## About the Project

This project explores whether a news article can be classified as fake or real based on its text.

I used the WELFake dataset and started with exploratory data analysis (EDA) to understand the data, examine the distribution of fake and real news, check missing values and duplicates, and look at differences in headline and article lengths.

After exploring the dataset, I cleaned the article text and used TF-IDF to convert it into numerical features. I then trained three machine learning models and compared their performance. I also used confusion matrices and misclassified examples to understand where the models make mistakes.

## Dataset

The project uses the **WELFake Dataset**, which contains news articles with their text and corresponding labels.

The labels used in this project are:

- `0` — Fake News
- `1` — Real News

The dataset is stored in CSV format as `WELFake_Dataset.csv`.

The dataset is included in this repository in Raw data folder.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

I explored the dataset to understand its structure and identify patterns before training the models.

This included:

- Checking the number of rows and columns.
- Examining the distribution of fake and real news.
- Checking missing values and duplicate rows.
- Inspecting repeated headlines and article texts.
- Comparing headline and article word counts.
- Finding frequently occurring words in both classes.
- Visualizing text-length distributions and category-wise differences.

### 2. Text Preprocessing

I cleaned the news text before using it for machine learning.

The preprocessing steps included:

- Converting text to lowercase.
- Removing URLs.
- Removing punctuation and non-alphabetic characters.
- Removing English stop words.
- Handling missing text by replacing it with an empty string for text processing.

### 3. Train-Test Split

I split the dataset into training and testing sets using a **70:30 ratio**.

I used stratified sampling to preserve the distribution of the two labels in both sets, with a fixed random state for reproducibility.

### 4. TF-IDF Feature Extraction

I used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert the cleaned article text into numerical features.

The vectorizer was configured to use:

- A maximum of 20,000 features.
- Unigrams and bigrams.
- A minimum document frequency of 2.
- A maximum document frequency of 95%.

This representation allows the models to learn patterns from individual words and pairs of words in the articles.

### 5. Machine Learning Models

I trained and compared three classifiers:

- **Logistic Regression** — a linear classification model used as a baseline.
- **Multinomial Naive Bayes** — a probabilistic classifier commonly used for text classification.
- **Linear Support Vector Machine (Linear SVM)** — a linear model that learns a decision boundary between fake and real news.

### 6. Model Evaluation

I evaluated the models on the test set and compared their results using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score

I also generated confusion matrices to examine how often fake news was correctly identified, how often real news was correctly identified, and where the models confused the two classes.

### 7. Error Analysis

I used the Linear SVM model to examine misclassified articles and identify false positives and false negatives.

This helped me look beyond the overall scores and investigate examples where the model predicted the wrong label.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Regular expressions
- Jupyter Notebook / Google Colab

## How to Run the Project
1. Clone or download this repository.
2. Download the dataset and place `WELFake_Dataset.csv` from the Raw data folder.
3.  Open `Fake_news_project.ipynb` in Jupyter Notebook or upload it to Google Colab.
4. Run the cells in order, from data loading and EDA through model training, evaluation, and error analysis.

## Repository Contents

- `Fake_news_project.ipynb` — notebook containing data exploration, text preprocessing, model training, evaluation, and error analysis.
- `WELFake_Dataset.csv` — the input dataset, which must be downloaded separately.



---

*An academic project exploring text classification and machine learning using Python.*
