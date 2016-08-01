##This creates different neural networks using tensorflow without tflearn
# read through to try and understand better how tensorflow works
import tensorflow as tf
import h5py


def get_data() :
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("mnist_data/", one_hot = True)


#so tensorflow is pretty much writing a computer program within a computer program
def softmax_regression() :

    x = tf.placeholder(tf.float32, [None, 784])
    #placeholder is a variable, but it acts like an input
    #None means any number can be passed in, and these specify dimensions

    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    #variables are modifiable by the computation

    y = tf.nn.softmax(tf.matmul(x, w) + b)
    #matmul matrix mulitplication
    # softmax() spits out a tensor

    y_holder = tf.placeholder(tf.float32, [None, 10])
    # a spot to hold the answers

    cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_holder * tf.log(y) , reduction_indices=[1]))

    #reduce sum and reduce mean compute mean and sum. I think the reduce means that it collapse it along an axis
    # instead of just finding a scalar sum and mean. 

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #it can implement all of the work for backprop automatically, pretty sweet!
    # I am guessing 0.5 is the learning rate? yep
    # not sure what type of thing train_step is, GradientDescentOptimizer is a class
    # docs say minimize returns an "Operation" still do not have tensorflows workflow straight in my head, as
    # I am not sure what that means

    init = tf.initialize_all_variables()
    #not sure how this works, I guess tensorflow keeps a static list of all tensors and operations so far 

    sess = tf.Session()
    sess.run(init) #so it is at this point that all variables have been initialized

    mnist = get_data() #get our data

    for i in range(1000) :
        batch_xs, batch_ys = mnist.train.next_batch(100) #fancy function
        sess.run(train_step, feed_dict={x : batch_xs, y_holder : batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_holder, 1))
    #computer the onehot vectors to see if we guessed right
    # one hot vectors are single dimension vectors which are all zeros expect one column, which has a one

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #so change the booleans to 0.0 or 1.0 and then get out a percentage

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_holder : mnist.test.labels}))

# now lets make a simple network
def softmax_regression2() :

    #so it looks like what you do is first create a "graph" with all the operations you need to perform
    # and then you create a session. A session is the connection to the backend c++ code that actually runs the graph 
    # efficiently. Normally you will first create the graph, and then launch the session to connect it

    mnist = get_data() 

    input = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])

    #So these are just placing things into a default graph that tensorflow automatically has

    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable (tf.zeros([10]))

    prediction = tf.nn.softmax(tf.matmul(input, w) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sume(labels * tf.log(prediction), reduction_indices=[1]))

    #so I guess cross_entropy is an operation. You have to run it in a session

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #another operations, the session must run it

    sess = tf.Session()
    sess.run(tf.initialize_all_variables()) #this gets the graph ready to go

    for i in range(1000) :
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={input: batch[0], labels : batch[1]})
        #so I guess the operations know how to run themselves? I feel like there are lurking
        # global variables I should be aware of going on somewhere
        # on a different note, you can replace any tensor in your graph with feed_dict, not just placeholders . . .
        # useful for loading weights and biases?

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={input:mnist.test.images, labels:mnist.test.labels}))

    #okay so I guess that both correct_prediction and accuracy were more nodes in that graph
    # and eval actually ran the entire graph to get the results we wanted. What is the difference
    # between run and eval? Oh it is the same thing. eval() just gets the current default session
    # and then runs it using that session . . .


    #Now we make an actual CNN

def deep_net_mnist() :
    mnist = get_data() 

    input = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])

    reshaped_input = tf.reshape(input, [-1, 28, 28, 1]) 

    #okay I think that the -1 means to adjust as necessary?
    # why can't we just intially have the shape as [None, 28, 28 1] ?


    def create_weight(shape) :
        arr = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(arr)

    def create_bias(shape) :
        arr = tf.constant(0.1, shape=shape)
        return tf.Variable(arr)

    def conv2d(x, w) :
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        #this is what I need to understand, if I can make this make sense, I will figure out what is going on

    def max_pool_2x2(x) :
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

    #I am guessing that these returns operations? maybe . . .

    # layer 1

    l1_weights = create_weight([5, 5, 1, 32])
    l1_bias = create_bias([32])

    l1_conv = tf.nn.relu(conv2d(reshaped_input, l1_weights) + l1_bias)
    l1_pool = max_pool_2x2(l1_conv)

    #so I guess this just connects the graph together behind the scenes

    # layer 2
    l2_weights = create_weight([5, 5, 32, 64])
    l2_bias = create_bias([64])

    l2_conv = tf.nn.relu(conv2d(l1_pool, l2_weights) + l2_bias)
    l2_pool = max_pool_2x2(l2_conv)

    # fully connected layer
    fc3_weights = create_weight([7 * 7 * 64, 1024])
    fc3_bias = create_bias([1024])

    flat = tf.reshape(l2_pool, [-1, 7 * 7 * 64])
    # there's that -1 again . . . hmmm

    fc3_activations = tf.nn.relu(tf.matmul(flat, fc3_weights) + fc3_bias)

    # dropout

    keep_prob = tf.placeholder(tf.float32)
    fc3_drop = tf.nn.dropout(fc3_activations, keep_prob) #pretty nifty

    # label layer

    sm4_weights = create_weight([1024, 10])
    sm4_bias = create_bias([10])

    prediction = tf.nn.softmax(tf.matmul(fc3_drop, sm4_weights) + sm4_bias)

    # final functions for prediction and back propagation
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(prediction), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(prediction, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #now get the ball rolling

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(20000) :

        batch = mnist.train.next_batch(50)
        if i % 100 == 0 :
            train_accuracy = sess.run(accuracy, feed_dict={input:batch[0], labels:batch[1], keep_prob:1.0})
            print("iter {}, accuracy : {}".format(i, train_accuracy))
        sess.run(train_step, feed_dict={input:batch[0], labels:batch[1], keep_prob:0.5})


    final_acc = accuracy.eval(feed_dict={input:mnist.test.images, labels:mnist.test.labels, keep_prob:1.0}, session=sess)

    print("final accuracy : {}".format(final_acc))

def efftest() :
    data = h5py.File('forms_out.h5', 'r')

    def create_weight(shape) :
        arr = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(arr)

    def create_bias(shape) :
        arr = tf.constant(0.1, shape=shape)
        return tf.Variable(arr)

    def conv2d(x, w) :
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        #this is what I need to understand, if I can make this make sense, I will figure out what is going on

    def max_pool_2x2(x) :
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

    input = tf.placeholder(tf.float32, [None, 256, 256, 3])
    labels = tf.placeholder(tf.float32, [None, 11])

    l1_weights = create_weight([3,3,3,32])
    l1_bias = create_bias([32])

    l1_conv = tf.nn.relu(conv2d(input, l1_weights) + l1_bias)
    l1_pool = max_pool_2x2(l1_conv)

    fc2_input = tf.reshape(l1_pool, [-1, 128 * 128 * 32])
    fc2_weights = create_weight([128 * 128 * 32, 30])
    fc2_bias = create_bias([30])

    fc2_layer = tf.nn.relu(tf.matmul(fc2_input, fc2_weights) + fc2_bias)

    fc3_weights = create_weight([30, 11])
    fc3_bias = create_bias([11])

    prediction = tf.nn.softmax(tf.matmul(fc2_layer, fc3_weights) + fc3_bias)

    loss_function = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(prediction), reduction_indices=[1]))

    back_prop = tf.train.GradientDescentOptimizer(0.01).minimize(loss_function)

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(prediction, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session() 
    sess.run(tf.initialize_all_variables())

    for i in range(300) :
        batch_start = (i % 10) * 70
        sess.run(back_prop, feed_dict={input:data['train_image'][batch_start:batch_start + 70], labels:data['train_label'][batch_start: batch_start + 70]})

        if (i % 30 == 0) :
            current_accuracy = sess.run(accuracy, feed_dict={input:data['test_image'], labels:data['test_label']})
            print('iter {}, accuracy {}'.format(i, current_accuracy))

    final_accuracy = sess.run(accuracy, feed_dict={input:data['test_image'], labels:data['test_label']})
    print("final accuracy : {}".format(final_accuracy))
        
efftest() 
