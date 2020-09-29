"""
Classifier Trainer

Syntax:
> python train_classifier.py <destination_db> <pickle_model>

Example:
> python train_classifier.py ../data/disaster_response_db.db message_classifier.pkl

Arguments:
    1) Path to database created in data pipeline
    2) Path to pickle model to be stored 
"""

# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Load Data from the Database 
    
    Arguments:
        database_filepath -> Path  of database built from data pipeline 
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_categories',engine)
    
    #Remove child alone as it has all zeros only
    #df = df.drop(['child_alone'],axis=1)
   
    # convert 2 to 1 (majority) in related 
    #df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns # This will be used for visualization purpose
    return X, y, category_names


def tokenize(text):
    """
    Tokenize the text
    
    Arguments:
        text -> text that needs to be tokenized
    Output:
        clean_tokens -> tokenized text
    """
   
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
   
    detected_urls = re.findall(url_regex, text)
    
    
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'urlplaceholder')

    
    tokens = nltk.word_tokenize(text)
    
   
    lemmatizer = nltk.WordNetLemmatizer()

    
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

def build_pipeline():
    """
    Build initial machine learning pipeline
    
    Arguments moc: A Classifier
        
    Output:
        pipeline built for multi output classification
    """
    
    pipeline1 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

   # Improve with GridSearch


    parameters = {
     'classifier__estimator__n_estimators': [50, 100, 200 ]
    }

    cv = GridSearchCV(pipeline1, parameters, n_jobs=-1)
    
    return cv

def average_classification_report (y_true, y_pred): 
    """ Calculates mean score for each class from classification report as a measure of the model
       Returns a dataframe : 
       average f1-score 
       average precision 
       average recall 
    """
    #instantiating a dataframe
    report = pd.DataFrame ()
    
    for col in y_true.columns:
        #returning dictionary from classification report
        class_dict = classification_report (output_dict = True, y_true = y_true.loc [:,col], y_pred = y_pred.loc [:,col])
    
        #converting from dictionary to dataframe
        eval_df = pd.DataFrame (pd.DataFrame.from_dict (class_dict))
        
        #print (eval_df)
        
        #dropping unnecessary columns
        eval_df.drop(['macro avg', 'weighted avg'], axis =1, inplace = True)
        
        #dropping unnecessary row "support"
        eval_df.drop(index = 'support', inplace = True)
        
        #calculating mean values
        av_eval_df = pd.DataFrame (eval_df.transpose ().mean ())
        
        #transposing columns to rows and vice versa 
        av_eval_df = av_eval_df.transpose ()
    
        #appending result to report df
        report = report.append (av_eval_df, ignore_index = True)    
    
    #renaming indexes for convinience
    report.index = y_true.columns
    
    return report

def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    Given a pipeline Predict classification of test set
    
    Arguments:
        pipeline -> ML pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    Y_pred = pipeline.predict(X_test)

    Y_pred_df = pd.DataFrame( Y_pred, columns = Y_test.columns) 
    report = average_classification_report(Y_test,Y_pred_df)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print(report)

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))


def save_model(pipeline, pickle_filepath):
    """
    Save Pipeline function
    
    save the model
    
    Arguments:
        pipeline -> model
        pickle_filepath -> path and filename of model to be saved
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))

def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl \n\
Arguments Description: \n\
1) Path to SQLite destination database (e.g. disaster_response_db.db)\n\
2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")

if __name__ == '__main__':
    main()
