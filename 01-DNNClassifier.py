import tensorflow as tf
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Read and normalize data
df = pd.read_csv('dataset\\01-data.csv')
df['Classification'] = df['Classification'].replace(2, 0)
for i in range(0, 9):
    df.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].min()) / (df.iloc[:, i].max() - df.iloc[:, i].min())
N = df.shape[0]
train_index = random.sample(range(N), 87)
valid_index = list(set(range(116)) - set(train_index))
train_X = df.iloc[train_index, :-1]
train_y = df.iloc[train_index, -1]
valid_x = df.iloc[valid_index, :-1]
valid_y = df.iloc[valid_index, -1]

# Baseline
baseline_model = RandomForestClassifier(n_estimators=50)
baseline_model.fit(train_X, train_y)
valid_y_predict = baseline_model.predict(valid_x)
baseline_acc = accuracy_score(valid_y, valid_y_predict)

# Used constant
N_EPOCHS = 50
BATCH_SIZE = 29
d = 9
n_hidden = 128
learning_rate = 0.01
dropout_prob = 0.4
tf.logging.set_verbosity(tf.logging.INFO)

# DNN Model
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (None, d))
    y = tf.placeholder(tf.float32, (None, ))
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("hidden_layer"):
    W = tf.Variable(tf.random_normal((d, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden, )))
    y_h = tf.nn.sigmoid(tf.matmul(x, W) + b)
    y_h = tf.nn.dropout(y_h, keep_prob)


with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden, 1)))
    b = tf.Variable(tf.random_normal((1, )))
    logits = tf.matmul(y_h, W) + b
    y_prob = tf.sigmoid(logits)
    y_pred = tf.round(y_prob)


with tf.name_scope("loss"):
    y_expand = tf.expand_dims(y, 1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_expand, logits=logits)
    loss = tf.reduce_sum(entropy)
    acc, acc_update = tf.metrics.accuracy(labels=y_expand, predictions=y_pred)

with tf.name_scope('optim'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('trained_model\\01-DNNClassifier', tf.get_default_graph())

# Create session and run train-valid
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
step = 0
for i in range(N_EPOCHS):
    pos = 0
    while pos < N:
        batch_X = train_X[pos:pos+BATCH_SIZE]
        batch_y = train_y[pos:pos+BATCH_SIZE]
        feed_dict = {x: batch_X, y: batch_y, keep_prob: dropout_prob}
        _,  summary, l, accuracy, _ = sess.run([train_op, merged, loss, acc, acc_update], feed_dict=feed_dict)

        print('epoch: {}, step: {}, loss: {}, accuracy: {}'.format(i, step, l, accuracy))
        train_writer.add_summary(summary, step)

        step += 1
        pos += BATCH_SIZE

y_predict = sess.run(y_pred, feed_dict={x: valid_x, keep_prob:1})
valid_acc = accuracy_score(y_true=valid_y, y_pred=y_predict)
tf.reset_default_graph()
