import re
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')

def cleanup_df(df):
    stop = stopwords.words('english')
    df['cleaned'] = df['merged'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df