# CAP 5771 - Data Science Group Project

Customer Retention for the Google Merchandise Store

Group 10: James Diffenderfer, Anshika Saxena, Jay Patel, Sanchit Deora

# Github Contents

Here we describe the layout of the files included in the github repository for this project. The files for the coding portion of the project are separated over two main directories corresponding to the two primary processes required for the project: **Data Preprocessing** and **Modeling/Evaluation**. We have also included a directory containing a pdf copy of the project presentation and report called **Documents**.

### Data Preprocessing

This directory contains all of the code that was used to preprocess the original Google Merchandise Store customer data included with the Kaggle Challenge [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction/data). There are three main files that were used to preprocess the original train_v2.csv and test_v2.csv data files downloaded from Kaggle and these files were run in the order they are stated below.

1. `Subsampling-Data.ipynb`: Creates labels required for predicting customer retention and performs subsampling of original data sets. Following these steps, a .csv file is saved for the subsampled training and testing sets that can then be loaded in the next step in the daa preprocessing pipeline.

2. `Feature_Engg_data.ipynb`: Using the subsampled data sets from step 1, this code performs feature engineering so that the modeling/evaluation phase of the project works better. In order for this code to run, it requires the `countries.csv` file which is used to create longitude and latitude features for the data. At the end of this file .csv files are saved for the training and test data sets that can then be loaded in the next step of the data preprocessing pipeline.

3. `Cleaning_Encoding_FeatureSelection.ipynb`: This is the final file in the data preprocessing pipeline. This code cleans the data (deals with NaNs, missing values, etc.) and determines importance of features so that insignificant features can be removed. At the end of this code, .csv files are saved for the training and testing data sets that can then be used for modeling and evaluation.

### Modeling and Evaluation

This directory contains all of the code that was used to create and evaluate models for predicting customer retention based on the preprocessed Google Merchandise Store data. The directory contains a subdirectory for each type of model that was used. Instead of describing each file here, we just describe the subdirectories and briefly outline their contents. Note that these subdirectories can all be run independently of each other except for the `Ensemble-Method` subdirectory which requires that all of the other models are pretrained and saved so that they can be reloaded for the ensemble method.

* **Baseline-Models-and-Clustering**: This directory contains all of the code for creating and testing the following models:

  * Linear Regression
  * Gaussian Naive Bayes Classifier
  * Multinomial Naive Bayes Classifier
  * KMeans Clustering


* **Neural-Network-Model-and-Data**: This directory contains all of the code used to create, test, and reload the following deep neural network models:

  * Deep Neural Network
  * Deep Neural Network with Batch Normalization


* **Support-Vector-Machine**: This directory contains all of the code used to create and test a linear support vector machine model for classifying whether or not customers return to the Google Merchandise Store.

* **Trees**: This directory contains all of the code used to create and test the following tree based models for classifying whether or not customers return to the Google Merchandise Store:

  * Random Forest
  * XGBoost Tree

* **Ensemble Method**: This directory contains the code and necessary files for an ensemble method that makes use of various models developed as part of the project. The contents of this directory are:

  * `Ensemble_Method_1`: Code that loads Random Forest and XGBoost models and combines them into an ensemble method
  
  * `Ensemble_Method_2`: Code that loads Logistic Regression, Support Vector Machine, Random Forest and Gaussian Naive Bayes Classifiers to create an ensemble Voting Classifier.

  * **NN-models**: Directory containing the necessary files for loading the previously trained neural network models.

  * **Trained_Trees**: Directory containing the necessary file for loading the XGBoost tree. The file required to load the random forest was excluded due to the large size of the file. 

### Documents

This directory contains two files that summarize the key details from the project:

* `Report.pdf`: A report outlining the pipeline for the project and key takeaways from the final models.

* `Presentation.pdf`: An in-class presentation that was given for the project.
