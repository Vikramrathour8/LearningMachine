import numpy as np
import h5py
import matplotlib.pyplot as plt
np.random.seed(1)

def load_data():
	train_dataset=h5py.File("Data/train_catvnoncat.h5","r")
	train_set_x_orig=np.array(train_dataset['train_set_x'][:])
	train_set_y_orig=np.array(train_dataset['train_set_y'][:])
	
	test_dataset=h5py.File("Data/test_catvnoncat.h5","r")
	test_set_x_orig=np.array(test_dataset['test_set_x'][:])
	test_set_y_orig=np.array(test_dataset['test_set_y'][:])
	
	classes=np.array(test_dataset['list_classes'][:])
	
	train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
	test_set_y_orig=test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
	
	return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.rand(n_h,n_x)
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)
    b2 = np.zeros((n_y,1))
 
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1)) 
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    
   
def sigmoid(Z):
	A=1/(1+np.exp(-Z))
	activation_cache=Z
	return A,activation_cache

def relu(Z):
	A=np.maximum(0,Z)
	activation_cache=Z
	return A,activation_cache

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dim[l],layer_dim[l-1])
        parameters['b' + str(l)] = np.random.zeros((layer_dim[l],1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))   
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)    # number of layers in the neural network  
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A,parameters['W'+str(l)],parameters['b'+str(l),'relu'])
        caches.append(cache)
    AL, cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = np.sum(np.dot(Y,np.log(AL).T)+np.dot(1-Y,np.log(1-AL).T))/(-m)    
    cost = np.squeeze(cost)      # To make sure to remove 1 unneccasary dimension
    assert(cost.shape == ())   
    return cost

def sigmoid_backward(dA,activation_cache):
	Z=activation_cache
	A=1/(1+np.exp(-Z))
	dZ=dA * A * (1-A)
	return dZ

def relu_backward(dA,activation_cache):
	Z=activation_cache
	dZ=np.array(dA,copy=True)
	dZ[Z<=0]=0
	return dZ
	

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)
    db = np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db



def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = -np.divide(Y,AL)+np.divide(1-Y,1-AL)
    current_cache = caches[L]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = cache[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+1)],current_cache,'relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    for i in range(1,L):
	parameters['W'+str(i)]-=learning_rate*grads['dW'+str(i)]
	parameters['b'+str(i)]-=learning_rate*grads['db'+str(i)]
		
    return parameters


def predict(test_case_x,test_case_y,parameters):
	m=test_case_x.shape[1]
	AL,cache=L_model_forward(test_case_x,parameters)
	new_prob=np.where(AL>0.5,1,0)
	print ("Accuracy: "+ str(np.sum(new_prob==test_case_y)/m))
	return new_prob
	
	
def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]    
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
