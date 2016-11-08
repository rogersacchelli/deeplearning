# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 192



NODES = 1024
train_subset = 10000
beta = 0.003

graph = tf.Graph()

with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights_0 = tf.Variable(
        tf.truncated_normal([image_size * image_size, NODES]))
    biases_0 = tf.Variable(tf.zeros([NODES]))

    print ('shapes w: %s' % str(weights_0.get_shape()))
    print ('shapes b: %s' % str(biases_0.get_shape()))

    relu_train = tf.nn.relu(tf.matmul(tf_train_dataset,weights_0) + biases_0)
    relu_train_0 = tf.nn.relu(tf.matmul(tf_train_dataset,weights_0) + biases_0)

    '''
    Problem 3
    Introduce Dropout on the hidden layer of the neural network.
    Remember: Dropout should only be introduced during training, not evaluation,
    otherwise your evaluation results would be stochastic as well.
    TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
    What happens to our extreme overfitting case?

    '''

    relu_train = tf.nn.dropout(relu_train,0.9)
    relu_valid = tf.nn.relu(tf.matmul(tf_valid_dataset,weights_0) + biases_0)
    relu_test = tf.nn.relu(tf.matmul(tf_test_dataset,weights_0) + biases_0)

    relu_train_0 = tf.nn.dropout(relu_train_0, 0.9)
    relu_valid_0 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_0) + biases_0)
    relu_test_0 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_0) + biases_0)

    weights_1 = tf.Variable(
        tf.truncated_normal([image_size*image_size,NODES]))

    biases_1 = tf.Variable(tf.zeros([NODES]))

    relu_train_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)

    relu_train_1 = tf.nn.dropout(relu_train_1, 0.9)
    relu_valid_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    relu_test_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)

    weights_2 = tf.Variable(
        tf.truncated_normal([NODES,num_labels]))

    biases_2 = tf.Variable(tf.zeros([num_labels]))

    logits_train = tf.matmul(relu_train_1, weights_2) + biases_2
    logits_valid = tf.matmul(relu_valid_1, weights_2) + biases_2
    logits_test = tf.matmul(relu_test_1, weights_2) + biases_2

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits_train, tf_train_labels))

    '''
    Problem 1
    Introduce and tune L2 regularization for both logistic and neural network models.
    Remember that L2 amounts to adding a penalty on the norm of the weights to the loss.
    In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t).
    The right amount of regularization should improve your validation / test accuracy.
    '''

    loss = loss + (beta/2)*(tf.nn.l2_loss(weights_0)+tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2))

    '''
    Problem 4
    Try to get the best performance you can using a multi-layer model!
    The best reported test accuracy using a deep network is 97.1%.
    One avenue you can explore is to add multiple layers.
    Another one is to use learning rate decay:
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    '''

    global_step = tf.Variable(0,trainable=False)
    start_learning_rate = 0.9

    leaning_rate = tf.train.exponential_decay(start_learning_rate,global_step=global_step,decay_steps=50,decay_rate=0.01,staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(leaning_rate).minimize(loss)

    train_prediction = tf.nn.softmax(logits_train)
    valid_prediction = tf.nn.softmax(logits_valid)
    test_prediction = tf.nn.softmax(logits_test)


num_steps = 4501

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))



'''
Problem 2
Let's demonstrate an extreme case of overfitting.
Restrict your training data to just a few batches. What happens?
'''



