#araf
import tensorflow as tf
import numpy as np
#  function 
atributes = [[-0.4, -0.4],
[0.6, -0.4],
[-0.4, 0.6],
[0.6, -0.4],
[-0.4, 0.6]]

labels = [0,
1,
0,
1,
0]

data=np.array(atributes,'float32')
target=np.array(labels,'float32')
feature_columns = [tf.contrib.layers.real_valued_column("")]
learningRate = 0.1
epoch= 10000 #learning time

classifier= tf.contrib.learn.DNNClassifier(
         feature_columns = feature_columns
		,hidden_units =[3] #2 attributes and 1 hidden 
		,activation_fn = tf.nn.sigmoid
		,optimizer=tf.train.GradientDescentOptimizer(learningRate)
)

classifier.fit(data,target,steps=epoch)

def test_set():
  return np.array(atributes, np.float32)

predictions = classifier.predict_classes(input_fn = test_set)

index = 0
for i in predictions:
   print(data[i], "-> actual: " , target[index] , ", predict :",i)
   index = index + 1 
   
