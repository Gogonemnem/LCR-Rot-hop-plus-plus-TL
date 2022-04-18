import tensorflow as tf
import tensorflow_text as text

# This is just an example on how to load/save models
# I accidentally saved the whole model, only weights is fine enough
model = tf.keras.models.load_model("Trained_Model")
model.save_weights("Trained_Weights/weights")

