import numpy as np
import tensorflow as tf

import utils

class Embedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, *args, **kwargs):
        # ensure you have a embedding_dim
        self.embedding_dim = embedding_dim
        super().__init__(*args, **kwargs)

class GloveEmbedding(Embedding):
    def __init__(self, glove_file, vocab_files):
        embedding_dim = 300 # TODO: Make modifiable for other gloves
        super().__init__(embedding_dim)

        self.glove_file = glove_file
        self.vocab_files = vocab_files
        self.embeddings_index = utils.load_pretrained_embeddings(glove_file)
        self.vectorizer = None
        
        for vocab_file in self.vocab_files:
            self.vectorizer, word_index = utils.vocabulary_index(vocab_file, self.vectorizer)
        
        num_tokens = len(self.vectorizer.get_vocabulary()) + 2

        embedding_matrix = np.zeros((num_tokens, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"

                # padding when sentence (or target phrase) is too short
                # no idea what oov is though
                embedding_matrix[i] = embedding_vector
        self.embedding = tf.keras.layers.Embedding(num_tokens, self.embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), trainable=False)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        vectors = self.vectorizer(inputs)
        embeddings = self.embedding(vectors)
        return embeddings


class BERTEmbedding(Embedding):
    import tensorflow_hub as hub
    import tensorflow_text as text
    def __init__(self, pp_url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                 bert_url='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4') -> None:
        embedding_dim = 768 # TODO: Make modifiable for other berts
        # print("loading bert")
        self.preprocessor = hub.KerasLayer(pp_url)
        self.bert = hub.KerasLayer(bert_url)

        # print("loaded bert")
        super().__init__(embedding_dim)
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs):
        embeddings = self.preprocessor(inputs)
        # print("preprocessoring done")
        output = self.bert(embeddings)
        # print("embedding done")

        # This is what Trusca used, not the 'best' one though
        # Best results are last four concatenated
        average_last_four = tf.math.reduce_mean(output['encoder_outputs'][-4:], axis=0)
        del output
        average_last_four = average_last_four #[:,:,:self.embedding_dim]  # for dim = 200
        # print(tf.shape(average_last_four))
        return average_last_four



###########################################################3 OLD CODE

# import string
# from transformers import AutoTokenizer, TFBertModel
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# print("loading bert tokenizer")
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# print("loaded bert tokenizer")

# print("loading bert model")
# model = TFBertModel.from_pretrained('bert-base-uncased')
# print("loaded bert model")

# sent = ["Hello, my dog is cute", "Hi there"]
# # sent = "Hello, my dog is cute"
# print(tokenizer(sent))
# print(tf.constant(tokenizer.encode(sent))[None, :])
# print(tokenizer(sent)['input_ids'])
# # input_ids = tf.constant(tokenizer.encode(sent))[None, :]
# input_ids = tokenizer(sent)['input_ids']
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
# print("sentence", sent)
# # print("encoded", tokenizer.encode(sent))
# print("encoded", tokenizer(sent))
# print("encoded batch 1", tf.constant(tokenizer.encode(sent))[None, :])
# # print("shape" + tf.shape(outputs))
# print("input_id", input_ids)
# print("decoded", tokenizer.decode(input_ids[0]))
# print("outputs", outputs)



# class HuggingfaceBERTLayer(tf.keras.layers.Layer):
#     def __init__(self, bert_preprocess: str='bert-base-uncased', bert_model: str='bert-base-uncased', *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.bert_preprocess = bert_preprocess
#         self.bert_model = bert_model
        
    
#     def build(self, input_shape):
#         print("loading bert tokenizer")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.bert_preprocess, do_lower_case=True)
#         print("loaded bert tokenizer")

#         print("loading bert model")
#         self.model = TFBertModel.from_pretrained(self.bert_model)
#         print("loaded bert model")

#         super().build(input_shape)

#     def call(self, inputs):


####################################################################################

if __name__ == '__main__':
    # load the pre-processing model 
    preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    # Use BERT pre-processing on a batch of raw text inputs.
    embeddings = preprocess(['Blog writing is awesome. If you understand it.', 'Hi there', 'Oh wow'])
    
    bert = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')
    output = bert(embeddings)

    test = tf.math.reduce_mean(output['encoder_outputs'][-4:], axis=0)
    print(test[:,:,:300])