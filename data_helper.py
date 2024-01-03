from preprocessing import cleanup_df as cleanup_df2
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning) 

import os
import pandas as pd

def read_data_part(type, part):
    files = os.listdir(f"lingspam_public/{type}/part{part}")

    df = pd.DataFrame(columns=["subject", "message", "is_spam"])
    for file in files:
        with open(f"lingspam_public/{type}/part{part}/{file}", "r") as f:
            subject = f.readline().replace("Subject: ", "").strip()
            message = f.read().strip()
            is_spam = 1 if file.startswith("spm") else 0

        df = df.append({"subject": subject, "message": message, "is_spam": is_spam}, ignore_index=True)
    
    return df

def cleanup_df(df):
    df['subject'] = df['subject'].str.lower()
    df['message'] = df['message'].str.lower()

    df['is_spam'] = df['is_spam'].astype('int')

    df.fillna(df['subject'].mode().values[0],inplace=True)
    df.fillna(df['message'].mode().values[0],inplace=True)

    # feature engineering
    df['merged']=df['subject']+" "+df['message']

    #REPLACING NUMBERS BY 'NUMBERS'
    df['merged']=df['merged'].str.replace(r'\d+(\.\d+)?', 'numbers')
    #CONVRTING EVERYTHING TO LOWERCASE
    df['merged']=df['merged'].str.lower()
    #REPLACING NEXT LINES BY 'WHITE SPACE'
    df['merged']=df['merged'].str.replace(r'\n'," ") 
    # REPLACING EMAIL IDs BY 'MAILID'
    df['merged']=df['merged'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','MailID')
    # REPLACING URLs  BY 'Links'
    df['merged']=df['merged'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','Links')
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df['merged']=df['merged'].str.replace(r'£|\$', 'Money')
    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df['merged']=df['merged'].str.replace(r'\s+', ' ')
    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df['merged']=df['merged'].str.replace(r'^\s+|\s+?$', '')
    #REPLACING CONTACT NUMBERS
    df['merged']=df['merged'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','contact number')
    #REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 
    df['merged']=df['merged'].str.replace(r"[^a-zA-Z0-9]+", " ")
    
    df = cleanup_df2(df)
    
    # df['cleaned'] = df['merged'] -> without cleaning the data
    
    # cleanup
    df.drop('subject',axis=1,inplace=True)
    df.drop('message',axis=1,inplace=True)
    df.drop('merged',axis=1,inplace=True)

    return df

def visualize(df):
    labels=df['is_spam'].value_counts().index.tolist()
    values=df['is_spam'].value_counts().values.tolist()
    plt.figure(figsize=(10,5),dpi=140)
    plt.pie(x=values,labels=labels)
    plt.legend(["0 = NO SPAM",'1 = SPAM'])
    plt.show()

def get_train_dataset(type):
    df = pd.DataFrame(columns=["subject", "message", "is_spam"])

    for part in range(1, 10):
        df = df.append(read_data_part(type, part), ignore_index=True)
    
    return cleanup_df(df)

def get_test_dataset(type):
    return cleanup_df(read_data_part(type, 10))