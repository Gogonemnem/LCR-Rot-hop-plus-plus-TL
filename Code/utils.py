import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
import csv


def load_pretrained_embeddings(embedding_path, skip_line=False):
    embeddings_index = {}
    with open(embedding_path) as f:
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

    with open(f_out, 'w', newline='') as file:
        columns = ("context_left", "target", "context_right", "polarity")
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for sentence in root.iter('sentence'):
            sent = sentence.find('text').text
            
            for opinion in sentence.iter('Opinion'):
                sentiment = opinion.get('polarity')
                if sentiment == "positive":
                    polarity = 1
                elif sentiment == "conflict":
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


def vocabulary_index(f_in: str):
    from tensorflow.keras.layers import TextVectorization

    root = ET.parse(f_in).getroot()
    samples = []

    for sentence in root.iter('sentence'):
        samples.append(sentence.find('text').text)
    
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200) # hyperparameters here
    text_ds = tf.data.Dataset.from_tensor_slices(samples).batch(128)
    vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    return vectorizer, word_index


def main():
    path = "C:/Users/Gonem/CodeProjects/seminar-ba-qm/Wallaart-HAABSA/data/externalData/absa-2015_restaurants_trial.xml"
    embed_path = "C:/Users/Gonem/CodeProjects/seminar-ba-qm/Wallaart-HAABSA/data/externalData/glove.6B.50d.txt"
    data_path = "sem_trial_2015.csv"
    # semeval_to_csv(path, data_path)


    vectorizer, word_index = vocabulary_index(path)
    embeddings_index = load_pretrained_embeddings(embed_path)

    num_tokens = len(vectorizer.get_vocabulary()) + 2
    embedding_dim = 50
    # hits = 0
    # misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
    #         hits += 1
    #     else:
    #         misses += 1
    # print("Converted %d words (%d misses)" % (hits, misses))

    from tensorflow.keras.layers import Embedding

    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    # int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
    # embedded_sequences = embedding_layer(int_sequences_input)
    # model = tf.keras.Model(int_sequences_input, embedded_sequences)
    # print(model.summary())

    left, target, right, polarity = semeval_data(data_path)

    # x_train = vectorizer(np.array([[s] for s in target])).numpy()
    print(vectorizer(np.array([s for s in target])).numpy())
    print(vectorizer(target).numpy())
    # x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

    # y_train = np.array(polarity, dtype=np.float32)
    # y_val = np.array(val_labels)
    # model.compile(loss="MeanSquaredError", optimizer=tf.keras.optimizers.SGD())
    # model.fit(x_train, y_train)


if __name__ == "__main__":
    main()