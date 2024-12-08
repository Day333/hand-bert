import warnings
import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')


def dataSet_preprocess(review):
    raw_text = BeautifulSoup(review,'html').get_text()
    letters = re.sub('[^a-zA-Z]',' ',raw_text)
    words = letters.lower().split()
    return words

def get_stopwords(stop_words_file):
    with open(stop_words_file,encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list


def data_provider(path):
    data = pd.read_csv(path)

    review_data = []
    sentiment_data = []

    for review in data['review']:
        review_data.append(' '.join(dataSet_preprocess(review)))

    for sentiment in data['sentiment']:
        sentiment_data.append(sentiment)

    data["review"] = pd.DataFrame(review_data)
    data["sentiment"] = pd.DataFrame(sentiment_data)

    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, Y_train, Y_test = train_test_split(data["review"], data['sentiment'], test_size=0.2)

    return data, X_train, X_test, Y_train, Y_test



def preprocessing_for_bert(data, tokenizer, max_len):
    inputs = []
    masks = []
    
    for text in data:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=max_len,
            truncation=True,  # Truncate to max_len if necessary
            padding='max_length',  # Pad sequences to max_len
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        inputs.append(encoding['input_ids'])
        masks.append(encoding['attention_mask'])

    inputs = torch.cat(inputs, dim=0)
    masks = torch.cat(masks, dim=0)

    return inputs, masks
