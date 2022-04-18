from embedding import BERTEmbedding
from tensorflow_addons.metrics.f_scores import F1Score
from transferlearningexperimental import TL
import utils
import tensorflow as tf

yelp_mult_train_path = 'ExternalData/yelp/mult/train-*.csv'
yelp_mult_test_path = 'ExternalData/yelp/mult/test-*.csv'

sem_rest_mult_train_path = 'ExternalData/semeval_2016/restaurant/mult/train-*.csv'
sem_rest_mult_test_path = 'ExternalData/semeval_2016/restaurant/mult/test-*.csv'

doc_train_x, doc_train_y = utils.csv_to_input(yelp_mult_train_path, ['text', 'polarity'])
doc_test_x, doc_test_y = utils.csv_to_input(yelp_mult_test_path, ['text', 'polarity'])

*asp_train_x, asp_train_y = utils.csv_to_input(sem_rest_mult_train_path, ['context_left', 'target', 'context_right', 'polarity'])
*asp_test_x, asp_test_y = utils.csv_to_input(sem_rest_mult_test_path, ['context_left', 'target', 'context_right', 'polarity'])

x_train = {'asp': asp_train_x, 'doc': doc_train_x}
y_train = {'asp': tf.one_hot(asp_train_y+1, 3, dtype='int32'), 'doc': tf.one_hot(doc_train_y+1, 3, dtype='int32')}

x_test = {'asp': asp_test_x, 'doc': doc_test_x}
y_test = {'asp': tf.one_hot(asp_test_y+1, 3, dtype='int32'), 'doc': tf.one_hot(doc_test_y+1, 3, dtype='int32')}

f1 = F1Score(num_classes=3, average='macro')

emb = BERTEmbedding()
model = TL(embedding_layer=emb, hop=3, hierarchy=(False, True), hidden_units=300)
model.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights={'asp': 1, 'doc': 0.5}, metrics=['accuracy', f1])
model.fit(x_train, y_train, epochs=5, batch_size=32)
