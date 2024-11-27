import pandas as pd
import os
import math
import numpy as np
import re
import random
from tqdm import tqdm


def read_data():
    all_data = pd.read_csv(os.path.join("data.csv"))
    all_text = all_data["content"].tolist()
    return all_text


def resplit_text(text_list):
    result = []
    sentence = ""
    for text in text_list:
        if len(text) < 3:
            continue
        if sentence == "":
            if random.random() < 0.2:
                result.append(text + "。")
                continue

        if len(sentence) < 30 or random.random() < 0.2:
            sentence += text + "，"
        else:
            result.append(sentence[:-1] + "。")
            sentence = text

    return result


def split_text(text):
    # patten = r"。|？"
    patten = r"[，、：；。？]"
    sp_text = re.split(patten, text)
    new_sp_text = resplit_text(sp_text)
    return new_sp_text


def build_neg_pos_data(text_list):
    all_text1, all_text2 = [], []
    all_label = []

    for tidx, text in enumerate(text_list):
        if tidx == len(text_list) - 1:
            break
        all_text1.append(text)
        all_text2.append(text_list[tidx + 1])
        all_label.append(1)

        c_id = [i for i in range(len(text_list)) if i != tidx and i != tidx + 1]

        other_idx = random.choice(c_id)

        other_text = text_list[other_idx]
        all_text1.append(text)
        all_text2.append(other_text)
        all_label.append(0)

    return all_text1, all_text2, all_label


def build_task2_dataset(text_list):
    all_text1 = []
    all_text2 = []
    all_label = []

    for text in tqdm(text_list):
        sp_text = split_text(text)
        if len(sp_text) <= 2:
            continue
        text1, text2, label = build_neg_pos_data(sp_text)

        all_text1.extend(text1)
        all_text2.extend(text2)
        all_label.extend(label)

    pd.DataFrame({"text1": all_text1, "text2": all_text2, "label": all_label}).to_csv(
        os.path.join("..", "data", "task2.csv"), index=False)


def build_word_2_index(all_text):
    if os.path.exists("index_2_word.txt") == True:
        with open("index_2_word.txt", encoding="utf-8") as f:
            index_2_word = f.read().split("\n")
            word_2_index = {w: idx for idx, w in enumerate(index_2_word)}
            return word_2_index, index_2_word
    word_2_index = {"[PAD]": 0, "[unused1]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "[UNK]": 5, }

    for text in all_text:
        for w in text:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)
    index_2_word = list(word_2_index)

    with open("index_2_word.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(index_2_word))

    return word_2_index, index_2_word


def get_data():
    all_data = pd.read_csv(os.path.join(".", "data", "task2.csv"))

    text1 = all_data["text1"].tolist()
    text2 = all_data["text2"].tolist()
    label = all_data["label"].tolist()

    return text1, text2, label


if __name__ == "__main__":
    all_text = read_data()
    build_task2_dataset(all_text)
    word_2_index = build_word_2_index(all_text)
    print("")
