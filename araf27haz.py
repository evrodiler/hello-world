#araf
import tensorflow as tf
import numpy as np
#  function 
atributes = [[-0.146739130434783, 0.00416666666666664],
[0.298913043478261, 0.322348484848485],
[0.309782608695652, 0.318560606060606],
[0.309782608695652, 0.314772727272727],
[-0.690217391304348, -0.374621212121212],
[0.233695652173913, 0.269318181818182],
[0.125, 0.193560606060606],
[-0.146739130434783, -0.677651515151515],
[-0.146739130434783, -0.0147727272727273],
[-0.146739130434783, -0.355681818181818]
]

labels = [0,1,1,1,0,0,0,1,1,0]

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
   
