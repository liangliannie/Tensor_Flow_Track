import tensorflow as tf
# import google
import os
print(os.path)
# tf.enable_eager_execution()
# tf.add(1, 2)
# hello = tf.constant('Hello, TensorFlow!')
# print(hello)

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test, y_test)
# 
# from matplotlib_venn import venn2
# _ = venn2(subsets = (3, 2, 1))
# from pyspark.sql import SparkSession