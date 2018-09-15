# import the necessary packages
from mydlchest.nn import neuralnetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# load the MNIST dataset apply min/max scaling to scale the pixel intensities
# to the range[0, 1] (each image is represented by an 8x8 feature vector)
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print ("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

# construct the train test splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# convert the label from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# training
print("[INFO} training...")
nn = neuralnetwork.NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

# evaluation
print("[INFO] evaluating...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))
