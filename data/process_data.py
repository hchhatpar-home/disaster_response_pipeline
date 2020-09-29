"""
Data Wrangling for Disaster Response Pipeline

Syntax : python process_data.py <messages_file> <categories_file> <destination_database>


Arguments Description:
    1) messages_file : Path to  messages file (e.g. disaster_messages.csv)
    2) categories_file : Path to categories file  (e.g. disaster_categories.csv)
    3) destination_database : Path to destination database (e.g. disaster_response_db.db)
"""

# Import all the relevant libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
 
def load_data(messages_filepath, categories_filepath):
    """
    Load Messages and Categories and creates a data frame
    
    Arguments:
        messages_filepath -> Path to messages file
        categories_filepath -> Path to categories file
    Output:
        df -> Data frame with  messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge dataframes
    df = pd.merge(messages,categories,on='id')

    return df 

def clean_data(df):
    """
    Takes a dataframe and splits categories and converts them into binary 0 or 1
    
    Arguments:
        df -> A messages and categories merged data frame
    Outputs:
        df -> Cleaned data frame 
    """
    
    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
   
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
 
    categories.columns = category_colnames

    # Convert category values to numbers
    for column in categories:
        # set each value to be last character of string
        categories[column] = categories[column].astype(str).str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)

    #Drop Duplicates
    df.drop_duplicates(inplace=True)

    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)

    # convert 2 to 1 (majority) in related 
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    
    return df

def save_data(df, database_filepath):
    """
    Save Data to SQLite Database 
    
    Arguments:
        df -> DataFrame with merged and cleaned messages and categories
        database_filepath -> Path to destination db filename
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('message_categories', engine, index=False, if_exists='replace')
 

def main():
    """
    Main function which is the etl piple line to load, clean and save data:
    """
    
    
    # Check if program is supplied with correct arguments to work
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] 

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning  data ...')
        df = clean_data(df)
        
        print('Saving data to DB : {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else: # Print the usage
        print("Please use correct arguments : \nUsage:\n\
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db \n\
Arguments Description: \n\
1) Path to  file containing messages (e.g. disaster_messages.csv)\n\
2) Path to  file containing categories (e.g. disaster_categories.csv)\n\
3) Path to  destination database (e.g. disaster_response_db.db)")

if __name__ == '__main__':
    main()
