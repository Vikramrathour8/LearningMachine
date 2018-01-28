import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from read import load_train_data,load_train_labels,load_test_data,load_test_labels
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


train_features=load_train_data()
train_labels=load_train_labels()
test_features=load_test_data()
test_labels=load_test_labels()
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

train_features=train_features.reshape(train_features.shape[0],-1).T
train_labels=train_labels.reshape(1,train_labels.shape[0])
test_features=test_features.reshape(test_features.shape[0],-1).T
test_labels=test_labels.reshape(1,test_labels.shape[0])
test_labels = convert_to_one_hot(test_labels, 10)
train_labels = convert_to_one_hot(train_labels, 10)

def create_placeholders(n_x,n_y):
	X=tf.placeholder(tf.float32,[n_x,None])
	Y=tf.placeholder(tf.float32,[n_y,None])
	return X,Y

def initialize_parameters(layers_dim):
	L=len(layers_dim)
	parameters={}
	for l in range(L-1):
		parameters['W'+str(l+1)]=tf.get_variable('W'+str(l+1),[layers_dim[l+1],layers_dim[l]  ],initializer=tf.contrib.layers.xavier_initializer(seed=1))
		parameters['b'+str(l+1)]=tf.get_variable('b'+str(l+1),[layers_dim[l+1],1],initializer=tf.zeros_initializer())
	return parameters

def propagate_forward(X,parameters):
	L=len(parameters)/2
	A_prev=X
	for l in range(L-1):
		Z1=tf.add(tf.matmul(parameters['W'+str(l+1)],A_prev),parameters['b'+str(l+1)])
		A_prev=tf.nn.relu(Z1)
	Z3=tf.add(tf.matmul(parameters['W'+str(L)],A_prev),parameters['b'+str(L)])
	return Z3

def compute_cost(Z3,Y):
	logits=tf.transpose(Z3)
	labels=tf.transpose(Y)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
	return cost

def models(train_features,train_labels,test_features,test_labels,learning_rate=0.001,num_epoch=40,minibatch_size=50):
	n_x,m=train_features.shape
	n_y=train_labels.shape[0]
	costs=[]
	seed=3
	X,Y=create_placeholders(n_x,n_y)
	layers_dim=[n_x,500,300,10]
	parameters=initialize_parameters(layers_dim)
	Z3=propagate_forward(X,parameters)
	cost=compute_cost(Z3,Y)
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epoch):
			epoch_cost = 0.                       # Defines a cost related to an epoch
            		num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
           		seed = seed + 1
            		minibatches = random_mini_batches(train_features, train_labels, minibatch_size, seed)

            		for minibatch in minibatches:

                		(minibatch_X, minibatch_Y) = minibatch
                
                		_ , minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
               			epoch_cost += minibatch_cost / num_minibatches

			if (epoch%10)==0:
				print ("Cost after " +str(epoch) + "iterations are" + str(epoch_cost))
				costs.append(epoch_cost)
		costs=np.squeeze(costs)
		plt.plot(costs)
		plt.xlabel("Number of Iteration")
		plt.ylabel("Costs")
		plt.show()
		parameters=sess.run(parameters)	
		print ("Parameters have been trained!")

		correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print ("Train Accuracy:", accuracy.eval({X: train_features, Y: train_labels}))
		print ("Test Accuracy:", accuracy.eval({X: test_features, Y: test_labels}))
		
		return parameters
	
parameters=models(train_features,train_labels,test_features,test_labels)

