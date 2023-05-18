from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

#Gathering and preparing the data:
iris_data = load_iris()

#Training the data and fitting the data in the model:
knn_model = KNeighborsClassifier( n_neighbors = 3 )
knn_model.fit( iris_data.data, iris_data.target )

#Predicting the data:
pred = knn_model.predict( [ [ 4, 6, 8, 10 ], ] )
print( iris_data.target_names[ pred ] )