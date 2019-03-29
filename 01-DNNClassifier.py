import tensorflow as tf
N_EPOCHS = 5
BATCH_SIZE = 100
D = 784

with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (None, D))
    y = tf.placeholder(tf.float32, (None, ))
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("hidden_layer"):
    W = tf.Variable(tf.random_normal(d, n_hidden))
    b = tf.Variable(tf.random_normal(d, ))
    y_h = tf.nn.sigmoid(tf.matmul(x, W) + b)
    y_h = tf.nn.dropout(y_h, keep_prob)


with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal(n_hidden, ))
    b = tf.Variable(tf.random_normal(1, ))
    logits = tf.matmul(y_h, W) + b
    y_prob = tf.sigmoid(logits)
    y_pred = tf.round(y_prob)

with tf.name_scope("loss"):
    y_expand = tf.expand_dims(y, 1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_expand, logits=logits)
    loss = tf.reduce_sum(entropy)

with tf.name_scope('optim'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('trained_model\\01-DNNClassifier', tf.get_default_graph())

sess = tf.Session()
sess.run(tf.global_variables_initializer())
step = 0
for i in range(N_EPOCHS):
    pos = 0
    while pos < N:
        batch_X = train_X[pos:pos+BATCH_SIZE]
        batch_y = train_y[pos:pos+BATCH_SIZE]
        feed_dict = {x: batch_X, y: batch_y, keep_prob: dropout_prob}
        _, summary, loss = sess.run([train_op, merged, loss], feed_dict=feed_dict)
        print('epoch: {}, step: {}, loss: {}'.format(i, step, loss))
        train_writer.add_summary(summary, step)

        step += 1
        pos += batch_size



