import numpy as np
def sigmoid(z):   
    s= 1./(1 + np.exp(-1*z))    
    return s

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))    
    return w, b
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
# fwd prop
def propagate(w, b, X, Y):
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)     
    #z= w.T * x -> 1x2 2x4
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation 
   
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
      
    # BACKWARD PROPAGATION (TO FIND GRAD)   
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
   
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
#andrew örnek
#w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
#and örneği
w, b, X, Y = np.random.randn(2,1) * 0.01, 0, np.array([[0,0,1,1],[0,1,0,1]]), np.array([[0,0,0,1]])

grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations): 
        grads, cost = propagate(w, b, X, Y)  
        dw = grads["dw"]
        db = grads["db"]
       
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db           
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


def predict(w, b, X):   
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)       
    A = sigmoid(np.dot(w.T, X) + b)  
    
    for i in range(A.shape[1]):        
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0          
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

print("X:",np.shape(X))
print("predictions = " + str(predict(w, b, X)))
