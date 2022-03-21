embedding_path = "ExternalData/glove.6B.300d.txt"
training_path = "ExternalData/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml"
validation_path = "ExternalData/ABSA15_Restaurants_Test.xml"

train_data_path = "ExternalData/sem_train_2015.csv"
test_data_path = "ExternalData/sem_test_2015.csv"

from embedding import GloveEmbedding
import utils

embedding = GloveEmbedding(embedding_path, [train_data_path, test_data_path])

left, target, right, polarity = utils.semeval_data(train_data_path)

print(left)