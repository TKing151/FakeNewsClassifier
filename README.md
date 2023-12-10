# FakeNewsClassifier
### Small project surveying 5 machine learning methods for fake news detection. 

## The Process

### Step 1: installations and imports
#### The first snippet installs nltk. the second begins by importing necessary libraries for a machine learning project focused on fake news detection. The libraries include NumPy and Pandas for data manipulation, NLTK for natural language processing tasks, and various modules from scikit-learn for machine learning tasks. Stopwords are common words (e.g., "and," "the") that are often removed from text data during natural language processing to focus on more meaningful words.

### Step 2: Load in the data
#### Third snippet creates a dataframe from the Fake News Prediction Dataset from Kaggle. After loading the data, the code uses the value_counts method on the 'label' column to count the occurrences of each unique value in the 'label' column. This is done to understand the distribution of labels in the dataset, particularly for binary classification tasks common in fake news detection. 

### Step 3: Text Cleaning
#### In this code snippet, a new column called 'cleaned_text' is added to the existing DataFrame. This column is created by applying a lambda function to the 'text' column. The lambda function processes each text entry in the 'text' column by splitting it into individual words and joining them back together, excluding any words that are present in the set of English stopwords (stop_words), which was previously downloaded using NLTK. After creating the 'cleaned_text' column, the original 'text' column is dropped from the DataFrame using the drop method.

### Step 4: train-test split
#### The dataset is split into input features (X) and target variable (y) for training a machine learning model. Next, the train_test_split function from scikit-learn is used to split the dataset into training and testing sets. The input features (X) and target variable (y) are split into training and testing sets (X_train, X_test, y_train, y_test). The test_size parameter is set to 0.2, indicating that 20% of the data will be used for testing, and the remaining 80% for training. 

### Step 5: Vectorizer
#### The CountVectorizer essentially converts the text data into a bag-of-words representation, where each document is represented as a vector of word frequencies.

### Step 6: testing the 5 models
#### Next, a dictionary called models is defined, containing the various machine learning models for classification, including Random Forest, Support Vector Machine (SVC), Naive Bayes, Logistic Regression, and k-Nearest Neighbors. A loop iterates through each model, fits it to the training data, makes predictions on the testing data, and computes the accuracy. From this, we see that logistic regression is the most accurate. 

### Step 7: Focusing in on logistic regression
#### Because logistic regression was most accurate, we will now try to improve on it. For this,  the StandardScaler standardizes the features by removing the mean and scaling to unit variance. The parameter with_mean is set to False, indicating that the scaler should not center the data before scaling. The logistic regression model is re-fitted using the scaled training data (X_train_scaled), updating the model with the scaled features. This step is essential when using certain machine learning algorithms, like logistic regression, that can be sensitive to the scale of input features. Scaling ensures that all features have a similar influence on the model, contributing to better model performance. The class_weight parameter is set to 'balanced'. The 'balanced' option automatically adjusts the weights of the classes based on the number of samples in each class. In binary classification, where there are typically two classes (e.g., fake and real news), 'balanced' means that the model will give more weight to the minority class to address class imbalance.

### Step 8: Test the improved logistic regression
#### Once again, predictions are made, and the accuracy is calculated as well as other measures. Here, we see that precision, recall, accuracy, and f1 score are all 93%. 

### Step 9: visualized confusion matrix
#### A confusion matrix is computed for the predictions made by the logistic regression model on the test data. The confusion_matrix function from scikit-learn is used, taking the true labels (y_test) and the predicted labels (logistic_predictions) as inputs.









