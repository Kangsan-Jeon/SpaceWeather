import tensorflow as tf

epoch = 1000
batch_size_train = 32
batch_size_test = 32
image_size = 224
n_category = 4
n_channel = 1
learning_rate = 2e-4

training = tf.placeholder(tf.bool)



def flare_train(X_img, n_category, n_channel):
    conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                             padding="SAME", activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                    padding="SAME", strides=2)
    dropout1 = tf.layers.dropout(inputs=pool1,
                                 rate=0.3, training=training)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                             padding="SAME", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                    padding="SAME", strides=2)
    dropout2 = tf.layers.dropout(inputs=pool2,
                                 rate=0.3, training=training)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                             padding="SAME", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                    padding="SAME", strides=2)
    dropout3 = tf.layers.dropout(inputs=pool3,
                                 rate=0.3, training=self.training)

    # Dense Layer with Relu
    flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
    dense4 = tf.layers.dense(inputs=flat,
                             units=625, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(inputs=dense4,
                                 rate=0.5, training=self.training)

    # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
    logits = tf.layers.dense(inputs=dropout4, units=10)


    return logits


X_img = tf.placeholder(tf.float32, shape=(None, image_size, image_size, n_channel))
flare_class = tf.placeholder(tf.float32, shape=(None, n_category))
#keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
y_conv = flare_train(X_img, n_category, n_channel)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=flare_class, logits=y_conv)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(flare_class, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for ep in range(epoch):
        image_tensor_train = []  # input data, images
        label_tensor_train = []  # output data, ont-hot vectors

        _ = sess.run(train_step,
                     feed_dict={X_img: image_tensor_train, flare_class: label_tensor_train,
                                learning_rate: learning_rate})
        accu_val, loss_val = sess.run([accuracy, loss],
                                      feed_dict={X_img: image_tensor_train, flare_class: label_tensor_train,
                                                 learning_rate: learning_rate})
        print(accu_val, loss_val)