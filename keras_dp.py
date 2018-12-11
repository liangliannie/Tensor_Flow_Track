# Importing Keras Sequential Model


from keras.models import Sequential
from keras.layers import Dense
import numpy

# Initializing the seed value to a integer.
seed = 7

numpy.random.seed(seed)

# Loading the data set (PIMA Diabetes Dataset)
dataset = numpy.loadtxt('/Users/lli51/Downloads/dppython/datasets/pima-indians-diabetes.csv', delimiter=",")

# Loading the input values to X and Label values Y using slicing.
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Initializing the Sequential model from KERAS.
model = Sequential()

# Creating a 16 neuron hidden layer with Linear Rectified activation function.
model.add(Dense(16, input_dim=8, init='uniform', activation='relu'))

# Creating a 8 neuron hidden layer.
model.add(Dense(8, init='uniform', activation='relu'))

# Adding a output layer.
model.add(Dense(1, init='uniform', activation='sigmoid'))



# Compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# Fitting the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))