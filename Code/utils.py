from typing import Sequence
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
import csv


def load_pretrained_embeddings(embedding_path, skip_line=False):
    embeddings_index = {}
    with open(embedding_path, encoding='utf-8') as f:
        if skip_line:
            next(f) # or f.readline()

        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index


def semeval_to_csv(f_in: str, f_out: str):
    root = ET.parse(f_in).getroot()

    with open(f_out, 'w', encoding='utf-8', newline='') as file:
        columns = ("context_left", "target", "context_right", "polarity")
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for sentence in root.iter('sentence'):
            sent = sentence.find('text').text
            
            for opinion in sentence.iter('Opinion'):
                sentiment = opinion.get('polarity')
                if sentiment == "positive":
                    polarity = 1
                elif sentiment == "neutral":
                    polarity = 0
                elif sentiment == "negative":
                    polarity = -1
                else:
                    polarity = None

                start = int(opinion.get('from'))
                end = int(opinion.get('to'))

                # skip implicit targets
                if start == end == 0: 
                    continue

                context_left = sent[:start] 
                context_right = sent[end:]
                writer.writerow({"context_left": context_left, "target": sent[start:end], "context_right": context_right, "polarity": polarity})
        

def semeval_data(f_in):
    data = pd.read_csv(f_in).fillna('') # replaces NaN values with empty strings
    # can be unpacked like this: left, target, right, label = semeval_data(f_in)
    return data.T.values


def vocabulary_index(f_in: str, vectorizer=None):
    from tensorflow.keras.layers import TextVectorization

    if vectorizer is None:
        vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200) # hyperparameters here

    root = ET.parse(f_in).getroot()
    samples = []

    for sentence in root.iter('sentence'):
        samples.append(sentence.find('text').text)
    
    text_ds = tf.data.Dataset.from_tensor_slices(samples).batch(128)
    vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    return vectorizer, word_index

def csv_to_input(inpath, column_names: Sequence[str]):
    df = dd.read_csv(inpath).compute()
    return [df[column_name] for column_name in column_names]
    
def main():
    semeval_to_csv("ExternalData\ABSA16_Restaurants_Train_SB1_v2.xml", "ExternalData\sem_train_2016.csv")
    semeval_to_csv("ExternalData\EN_REST_SB1_TEST.xml.gold", "ExternalData\sem_test_2016.csv")


if __name__ == "__main__":
    main()