# Udacity Data  Science Starbucks Capstone Challenge

![](./reports/readme_image2.png)

This project is a the capstone project of Udacity [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
## Table of Contents
1. [Project Overview](#overview)
2. [Project Components](#components)
3. [Installation](#installation)
4. [File Descriptions](#descriptions)
5. [Instructions](#instructions)
6. [Project Structure](#structure)
7. [Results](#results)
8. [Acknowledgements](#acknowledgements)

## 1. Project Overview <a name="overview"></a>
This dataset simulates customer interactions with the Starbucks rewards mobile app. Periodically, Starbucks sends out promotional offers to app users. These offers can range from simple drink advertisements to actual incentives like discounts or buy-one-get-one-free deals. However, not all users may receive an offer during a given period. 

The challenge is to merge transaction, demographic, and offer data to identify which demographic groups are most responsive to different types of offers. Please note that this dataset is a simplified representation of the actual Starbucks app, as the underlying simulator only features one product, while Starbucks offers a wide variety of products.

## 2. Project Components <a name="components"></a>
The objective of this project is to construct a machine learning model capable of predicting whether customers will react to a specific type of offer. The project follows these steps:

- Cleanse the three provided datasets. This involves handling missing values, removing duplicates, and ensuring data types are correct.

- Conduct exploratory data analysis. The questions investigated include:
  - How many customers in this dataset received offers, and what types of offers were they?
  - Customer Analysis: What are the demographics of the customers (age, gender, income, etc.)?
  - Offer Analysis: How effective are the offers? Which types of offers are more successful in driving orders?

- Combine the three cleaned datasets and prepare the data for model training. This involves feature engineering and selection, as well as data normalization.

- Develop machine learning models to predict offer usage:
  - Define the features and target, and create training and testing datasets.
  - Train the Classifier. We'll be using a Gradient Boosting Classifier, which often provides high accuracy.
  - Refine the Model. We'll tune the hyperparameters to improve the model's performance.
  - Determine feature importance. This will give us insights into which features are most influential in predicting offer usage.
  - Predict the test data and generate a confusion matrix. This will allow us to evaluate the model's performance in terms of precision, recall, and F1 score.

- Draw conclusions. We'll summarize the findings and discuss the implications.
  - Suggest potential improvements. We'll propose ways to enhance the model's performance and the strategy for sending offers.


## 3. Installation <a name="installation"></a>
- The code are runing on python 3.11
- Data Processing Libraries: NumPy, Pandas, Math, Json
- Data Visualization Libraries: Matplotlib, Plotly
- Meachine Learning Library: Sciki-Learn


## 4. File Descriptions <a name="descriptions"></a>

- `portfolio.json` - Offer metadata, including id, type, difficulty, reward, duration, and channels.
- `profile.json` - Customer demographics, including age, membership start date, gender, id, and income.
- `transcript.json` - Transaction records, including event type, customer id, time, and value (either an offer id or transaction amount).


## 5. Project Structure <a name="structure"></a>

Here is the structure of the project workspace:

- `.gitignore` - Specifies which files and directories to ignore in git version control
- `data/` - Contains the raw data files and the data cleaning script
  - `data_cleaning.py` - Python script for cleaning the data
  - `portfolio.json` - Contains offer ids and metadata about each offer
  - `profile.json` - Contains demographic data for each customer
  - `transcript.json` - Contains records for transactions, offers received, offers viewed, and offers completed
- `models/` - Contains the scripts for training the classifier, refining the model, and generating the confusion matrix
  - `confusion_matrix.py` - Python script for generating the confusion matrix
  - `model_refinement.py` - Python script for refining the model
  - `train_classifier.py` - Python script for training the classifier
- `README.md` - This file, providing an overview of the project
- `Starbucks_Capstone_notebook.ipynb` - Jupyter notebook containing the complete analysis


## 6. Instructions <a name="instructions"></a>
- The complete analysis is contained within the Jupyter notebook in the root directory.
- The original datasets, provided as three JSON files, are located in the 'data' folder.

## 7. Results <a name="results"></a>
The project's introduction and main findings are detailed in a blog post, which can be accessed [here](https://medium.com/@casasvil/udacity-data-science-starbucks-capstone-challenge-2551df6af8f3).

## 8. Acknowledgements <a name="acknowledgements"></a>
This project serves as the capstone project for the Udacity Data Science Nanodegree. The dataset used is a simulated representation of customer behavior on the Starbucks rewards mobile app.
