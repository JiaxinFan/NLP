import time
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, data_file, label_file, max_len, tag):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tag = tag
        datalist = []
        if label_file != None:
            for data, label in zip(data_file, label_file):
                ids = data.strip().split(",")
                fpath = self.tag + "/"
                if (Path(fpath + ids[0] + ".json")).is_file():
                    tweets = dict()
                    temp = []
                    for i in ids:
                        if (Path(fpath + i + ".json")).is_file():
                            temp.append(
                                json.load(open(fpath + i + ".json", "r")))
                    tweets["text"] = processText(temp, self.tag)
                    tweets["label"] = convert_label(label)
                    datalist.append(tweets)
        else:
            for data in data_file:
                ids = data.strip().split(",")
                fpath = self.tag + "/"
                if (Path(fpath + ids[0] + ".json")).is_file():
                    tweets = dict()
                    temp = []
                    for i in ids:
                        if (Path(fpath + i + ".json")).is_file():
                            temp.append(
                                json.load(open(fpath + i + ".json", "r")))
                    tweets["text"] = processText(temp, self.tag)
                    datalist.append(tweets)

        self.dataset_df = pd.DataFrame(datalist)

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        if self.tag != 'test':
            text = self.dataset_df.iloc[index]["text"]
            label = self.dataset_df.iloc[index]["label"]
            return text, label
        else:
            input_ids = self.dataset_df.iloc[index]["text"]
            return input_ids

    def my_collate(self, batch):
        texts = []
        labels = []
        encode_data = dict()

        if self.tag != 'test':
            for text, label in batch:
                texts.append(text)
                labels.append(label)
        else:
            for text in batch:
                texts.append(text)

        tokens = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        if self.tag != 'test':
            encode_data['input_ids'] = tokens.input_ids
            encode_data['attn_masks'] = tokens.attention_mask
            encode_data['label'] = torch.LongTensor(labels)
        else:
            encode_data['input_ids'] = tokens.input_ids
            encode_data['attn_masks'] = tokens.attention_mask

        return encode_data

def processText(datalist, tag):
    if tag != 'test':
        datalist = sorted(datalist, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
    else:
        datalist = sorted(datalist,
                          key=lambda x: time.mktime(time.strptime(x["created_at"], '%a %b %d %H:%M:%S +0000 %Y')))
    processed_text = ""
    for item in datalist:
        new_text = []
        for text in item["text"].split(" "):
            text = text.replace('\n', '').replace('\r', '').lower()
            if text.startswith('@') and len(text) > 1:
                text = '@user'
            if text.startswith('http'):
                text = 'http'
            new_text.append(text)
        processed = " ".join(new_text)
        processed_text = processed_text + processed
    return processed_text

def convert_label(label):
    if label == 'nonrumour':
        return 0
    else:
        return 1
