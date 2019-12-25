# we get the built in data set iris_flower that exist in the sklearn library
from sklearn.datasets import load_digits

# load data set of iris to variable 'iris'
iris = load_digits()

# x, y feature and response
x = iris.data
y = iris.target

# split training set and testing set
from sklearn.model_selection import train_test_split

# train_test_split parameter first and second take feature and response and test size take the size of
# testing set in our 150*.4 = 60 row
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=1)

print(y)
print(x_train.shape)  # it should be 90 row
print(x_test.shape)  # it should be 60 row

print(y_train.shape)  # it should be 90 row
print(y_test.shape)  # it should be 60 row


# training in the train set
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# making prediction
y_predict = knn.predict(x_test)

# to ensue the prediction is good
from sklearn import metrics

print("knn model accuracy : ", metrics.accuracy_score(y_test, y_predict))
""""
# out put sample or user input sample
sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
predict = knn.predict(sample)
pred_species = [iris.target_names[p] for p in predict]

print("Predictions:", pred_species)

# saving the model to after use
from sklearn.externals import joblib
joblib.dump(knn, 'iris_knn.pkl')

# when we want to import it we use
# knn = joblib.load('iris_knn.pkl')

"""