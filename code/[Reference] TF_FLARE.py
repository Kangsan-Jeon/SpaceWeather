import tensorflow as tf

epoch = 1000
batch_size_train = 32
batch_size_test = 32
image_size = 224
n_category = 10
n_channel = 3
learning_rate = 2e-4


def weight_variable(shape, name = None) :
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def conv_layer(x, W, strides, padding, name = None) :
    return tf.nn.conv2d(x, W, strides = strides, padding = padding, name = name)

def pool_layer(x, ksize, strides, padding, name = None) :
    return tf.nn.max_pool(x, ksize = ksize, strides = strides, padding = padding, name = name)

def VGG16(mdi_image, n_category, n_channel, keep_prob, phase_train):

    W1 = weight_variable([3, 3, n_channel, 64])
    L1 = conv_layer(mdi_image, W1, [1, 1, 1, 1], 'SAME')
    L1 = tf.nn.relu(L1)
    
    W2 = weight_variable([3, 3, 64, 64])
    L2 = conv_layer(L1, W2, [1, 1, 1, 1], 'SAME')
    L2 = tf.nn.relu(L2)
    L2 = pool_layer(L2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


    W3 = weight_variable([3, 3, 64, 128])
    L3 = conv_layer(L2, W3, [1, 1, 1, 1], 'SAME')
    L3 = tf.nn.relu(L3)
    
    W4 = weight_variable([3, 3, 128, 128])
    L4 = conv_layer(L3, W4, [1, 1, 1, 1], 'SAME')
    L4 = tf.nn.relu(L4)
    L4 = pool_layer(L4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


    W5 = weight_variable([3, 3, 128, 256])
    L5 = conv_layer(L4, W5, [1, 1, 1, 1], 'SAME')
    L5 = tf.nn.relu(L5)

    W6 = weight_variable([3, 3, 256, 256])
    L6 = conv_layer(L5, W6, [1, 1, 1, 1], 'SAME')
    L6 = tf.nn.relu(L6)

    W7 = weight_variable([3, 3, 256, 256])
    L7 = conv_layer(L6, W7, [1, 1, 1, 1], 'SAME')
    L7 = tf.nn.relu(L7)
    L7 = pool_layer(L7, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


    W8 = weight_variable([3, 3, 256, 512])
    L8 = conv_layer(L7, W8, [1, 1, 1, 1], 'SAME')
    L8 = tf.nn.relu(L8)

    W9 = weight_variable([3, 3, 512, 512])
    L9 = conv_layer(L8, W9, [1, 1, 1, 1], 'SAME')
    L9 = tf.nn.relu(L9)

    W10 = weight_variable([3, 3, 512, 512])
    L10 = conv_layer(L9, W10, [1, 1, 1, 1], 'SAME')
    L10 = tf.nn.relu(L10)
    L10 = pool_layer(L10, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


    W11 = weight_variable([3, 3, 512, 512])
    L11 = conv_layer(L10, W11, [1, 1, 1, 1], 'SAME')
    L11 = tf.nn.relu(L11)

    W12 = weight_variable([3, 3, 512, 512])
    L12 = conv_layer(L11, W12, [1, 1, 1, 1], 'SAME')
    L12 = tf.nn.relu(L12)

    W13 = weight_variable([3, 3, 512, 512])
    L13 = conv_layer(L12, W13, [1, 1, 1, 1], 'SAME')
    L13 = tf.nn.relu(L13)
    L13 = pool_layer(L13, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


    pool_shape = L13.get_shape().as_list()
    L13 = tf.reshape(L13, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]], name="Flat")


    W14 = weight_variable([pool_shape[1] * pool_shape[2] * pool_shape[3], 4096], name="WF1")
    L14 = tf.nn.relu(tf.matmul(L13, W14, name="FC1"), name="ReLu")
    L14 = tf.nn.dropout(L14, keep_prob=0.5, name="Dropout")


    W15 = weight_variable([4096, 4096], name="WF2")
    L15 = tf.nn.relu(tf.matmul(L14, W15, name="FC2"), name="ReLu")
    L15 = tf.nn.dropout(L15, keep_prob=0.5, name="Dropout")


    W16 = weight_variable([4096, n_category], name="WF3")
    L16 = tf.matmul(L15, W16, name="FC3")

    return L16



mdi_image = tf.placeholder(tf.float32, shape = (None, image_size, image_size, n_channel))
flare_class = tf.placeholder(tf.float32, shape = (None, n_category))
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool)
y_conv = VGG16(mdi_image, n_category, n_channel, keep_prob, phase_train)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = flare_class, logits = y_conv)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)
    
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(flare_class, 1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess :
    
    init = tf.global_variables_initializer()
    sess.run(init)

    for ep in range(epoch) :
        
        image_tensor_train = [] #input data, images
        label_tensor_train = [] #output data, ont-hot vectors
        
        _ = sess.run(train_step, feed_dict = {mdi_image : image_tensor_train, flare_class : label_tensor_train, keep_prob : 0.5, phase_train : True, learning_rate : learning_rate})
        accu_val, loss_val = sess.run([accuracy, loss], feed_dict = {mdi_image : image_tensor_train, flare_class : label_tensor_train, keep_prob : 1.0, phase_train : False, learning_rate : learning_rate})