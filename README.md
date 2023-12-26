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

```bash
pip install -r requirements.txt
```


## 4. File Descriptions <a name="descriptions"></a>

**Portfolio Data (`portfolio.json`):** Contains details about each offer, including offer ID, type (BOGO, discount, informational), difficulty, reward, duration, and distribution channels. These details are crucial for categorizing and analyzing the effectiveness of different offers.
- **Contents**: 10 offers with details on reward, channels, difficulty, duration, offer type, and ID.
- **Key Points**: Offers are varied, with different rewards (mean 4.2), distribution channels (e.g., email, mobile), difficulty levels, and durations (average 6.5 days).
- **Usage**: Critical for understanding offer effectiveness and tailoring marketing strategies.

**Profile Data (`profile.json`):** Provides demographic data of customers, such as age, date of app account creation, gender, ID, and income. This dataset allows for the segmentation of customers and understanding their response patterns.
- **Contents**: Demographics of 17,000 customers, including gender, age, ID, membership start date, and income.
- **Key Points**: Gender and income have missing values; age data (mean 62.53 years) includes placeholders (e.g., age 118).
- **Usage**: Essential for customer segmentation and personalized marketing; aids in loyalty and membership analysis.

**Transcript Data (`transcript.json`):** Records transactions, offers received, offers viewed, and offers completed, along with timestamps and monetary values. This dataset is pivotal for tracking customer behavior in response to offers.
- **Contents**: 306,534 customer interactions, detailing person ID, event type, value details, and time.
- **Key Points**: Covers a wide range of interactions, with event times averaging 366 hours.
- **Usage**: Offers insights into behavioral responses to offers and overall effectiveness of marketing campaigns.


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
The project's introduction and main findings are detailed in a blog post, which can be accessed [here](https://github.com/jan-casas/udacity-starbucks/blob/main/blog.md).

## 8. Acknowledgements <a name="acknowledgements"></a>
This project serves as the capstone project for the Udacity Data Science Nanodegree. The dataset used is a simulated representation of customer behavior on the Starbucks rewards mobile app.
