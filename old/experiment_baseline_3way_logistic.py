import random
import numpy as np
import tensorflow as tf

from dataset_reader import DataSet, DataSplit
from composite_dataset import CompositeDataset

np.random.seed(1)
tf.set_random_seed(1)

def onehot(vals):
    return (np.arange(num_classes)[:] == np.array(vals)[:, None]).astype(np.float32)


def bow(vals):
    vals = list(vals)
    arr = np.zeros((len(vals), num_feats))

    for idx, val in enumerate(vals):
        for key in val.keys():
            arr[idx, key] = val[key]

    return arr


racism = DataSet("racism_overlapfree")
sexism = DataSet("sexism_overlapfree")
neither = DataSet("neither_overlapfree")

data = CompositeDataset()
data.add_data('racism',racism)
data.add_data('sexism',sexism)
data.add_data('neither',neither)

train,dev,test,vocab = data.get_as_labelled()




expert = DataSet('amateur_expert')
e_racism, e_sexism, e_neither, e_both = [], [], [], []

for item in expert.data:
    if item['Annotation'].lower() == 'sexism':
        e_sexism.append(item)
    elif item['Annotation'].lower() == 'racism':
        e_racism.append(item)
    elif item['Annotation'].lower() == 'neither':
        e_neither.append(item)
    else:
        # TODO Consider merging this into 'racism' to boost counts there
        e_both.append(item)

c = CompositeDataset()
c.add_data("racism", DataSplit(e_racism,train=0.0,dev=0.0))
c.add_data("sexism", DataSplit(e_sexism,train=0.0,dev=0.0))
c.add_data("neither", DataSplit(e_neither,train=0.0,dev=0.0))

_,_,e_test,_ = c.get_as_labelled()

num_classes = len(data.labels)
num_feats = len(vocab.vocab)

X_train,y_train = map(list,zip(*train))
X_dev,y_dev = map(list,zip(*dev))
X_test,y_test = map(list,zip(*test))

X_external, y_external = map(list,zip(*e_test))

y_train = onehot(y_train)
y_dev = onehot(y_dev)
y_test = onehot(y_test)

y_external = onehot(y_external)

X_train = bow(map(vocab.lookup,X_train))
X_dev = bow(map(vocab.lookup,X_dev))
X_test = bow(map(vocab.lookup,X_test))

X_external = bow(map(vocab.lookup,X_external))


batch_size = 50
l2_lambda = 1e-5

graph = tf.Graph()

with graph.as_default():
    tf.set_random_seed(1)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.8)

    train_data = tf.placeholder(tf.float32, [batch_size, num_feats])
    train_labels = tf.placeholder(tf.float32, [batch_size, num_classes])

    dev_data = tf.constant(X_dev.astype(np.float32))
    test_data = tf.constant(X_test.astype(np.float32))
    external_data = tf.constant(X_external.astype(np.float32))

    weights1 = tf.Variable(tf.truncated_normal([num_feats, num_classes]))
    biases1 = tf.Variable(tf.zeros([num_classes]))

    logits = tf.add(tf.matmul(train_data, weights1), biases1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits)) + l2_lambda * (
    tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_prediction = tf.nn.softmax(logits)
    dev_prediction = tf.nn.softmax(tf.matmul(dev_data, weights1) + biases1)
    test_prediction = tf.nn.softmax(tf.matmul(test_data, weights1) + biases1)
    external_prediction = tf.nn.softmax(tf.matmul(external_data, weights1) + biases1)

num_epochs = 30
num_steps = y_train.shape[0] * num_epochs // batch_size
print(num_steps)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)

        batch_data = X_train[offset:(offset + batch_size)]
        batch_labels = y_train[offset:(offset + batch_size), :]

        feed_dict = {train_data: batch_data, train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 100 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

            print("VALIDATION accuracy: %.1f%%" % accuracy(dev_prediction.eval(), y_dev))
    print("FINAL TEST accuracy: %.1f%%" % accuracy(test_prediction.eval(), y_test))
    print("EXTERNAL TEST accuracy: %.1f%%" % accuracy(external_prediction.eval(), y_external))

    from sklearn.metrics import classification_report

    print(classification_report(y_test,onehot(np.argmax(test_prediction.eval(),1))))
    print(classification_report(y_external,onehot(np.argmax(external_prediction.eval(),1))))