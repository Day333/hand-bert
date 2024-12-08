
from data import data_provider, preprocessing_for_bert
from model import initialize_model
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torchtext import data
from transformers import BertTokenizer
from model import train
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    bert_path = '/mnt/models/bert-base-uncased'
    data_path = '/mnt/data/IMDB_Dataset.csv'

    data, X_train, X_test, Y_train, Y_test = data_provider(data_path)

    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)

    MAX_LEN = 512
    encoded_comment = [tokenizer.encode(sent, add_special_tokens=True, padding='max_length', truncation=True, max_length=MAX_LEN) for sent in data.review.values]    

    train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer, MAX_LEN)
    test_inputs, test_masks = preprocessing_for_bert(X_test, tokenizer, MAX_LEN)

    train_labels = torch.tensor(Y_train.values, dtype=torch.long)
    test_labels = torch.tensor(Y_test.values, dtype=torch.long)

    batch_size = 16
    epochs = 1

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    bert_classifier, optimizer, scheduler = initialize_model(device, train_dataloader, bert_path, epochs)

    print("Start training and testing:\n")
    train(bert_classifier, device, optimizer, scheduler, train_dataloader, test_dataloader, epochs=1, evaluation=True)