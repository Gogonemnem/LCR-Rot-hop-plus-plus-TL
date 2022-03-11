import string
from transformers import AutoTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("loading bert model")
model = TFBertModel.from_pretrained('bert-base-uncased', do_lower_case=True)
print("loaded bert model")


print("loading bert tokenizer")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
print("loaded bert tokenizer")



sent = "Hello, my dog is cute"
input_ids = tf.constant(tokenizer.encode(sent))[None, :]  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

print(input_ids)
print(tokenizer.decode(input_ids))
print(outputs)