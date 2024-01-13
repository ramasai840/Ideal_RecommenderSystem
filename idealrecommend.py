import surprise

from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy
from surprise import SVD

movie_data = Dataset.load_builtin('ml-100k')
trainingdata, testdata = train_test_split(movie_data, test_size = .3, random_state = 40)

movie_recommender = KNNBasic()
movie_recommender.fit(trainingdata)

SVD_recommender = SVD()
SVD_recommender.fit(trainingdata)

predictions = movie_recommender.test(testdata)
svd_predictions = SVD_recommender.test(testdata)


print("sample test data --- ", predictions[0])
accuracy.rmse(predictions)
accuracy.rmse(svd_predictions)
