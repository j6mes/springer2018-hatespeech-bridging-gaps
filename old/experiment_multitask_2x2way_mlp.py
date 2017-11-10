import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from dataset_reader import DataSet
from composite_dataset import CompositeDataset

np.random.seed(1)

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

_,_,_,vocab = data.get_as_labelled()


racism_data = CompositeDataset()
racism_data.add_data('racism',racism)
racism_data.add_data('neither',neither)

racism_train,racism_dev,racism_test,_ = racism_data.get_as_labelled()


sexism_data = CompositeDataset()
sexism_data.add_data('sexism',sexism)
sexism_data.add_data('neither',neither)

sexism_train,sexism_dev,sexism_test,_ = sexism_data.get_as_labelled()

num_classes = 2
num_feats = len(vocab.vocab)

X_racism_train,y_racism_train = map(list,zip(*racism_train))
y_racism_train = onehot(y_racism_train)

X_sexism_train,y_sexism_train = map(list,zip(*sexism_train))
y_sexism_train = onehot(y_sexism_train)

X_racism_train = bow(map(vocab.lookup,X_racism_train))
X_sexism_train = bow(map(vocab.lookup,X_sexism_train))

X_racism_dev,y_racism_dev = map(list,zip(*racism_dev))
y_racism_dev = onehot(y_racism_dev)

X_sexism_dev,y_sexism_dev = map(list,zip(*sexism_dev))
y_sexism_dev = onehot(y_sexism_dev)

X_racism_dev = bow(map(vocab.lookup,X_racism_dev))
X_sexism_dev = bow(map(vocab.lookup,X_sexism_dev))

X_racism_test,y_racism_test = map(list,zip(*racism_test))
y_racism_test = onehot(y_racism_test)

X_sexism_test,y_sexism_test = map(list,zip(*sexism_test))
y_sexism_test = onehot(y_sexism_test)

X_racism_test = bow(map(vocab.lookup,X_racism_test))
X_sexism_test = bow(map(vocab.lookup,X_sexism_test))

batch_size = 50
l2_lambda = 1e-3

hidden_size = 20

graph = tf.Graph()

with graph.as_default():
    tf.set_random_seed(1)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.8)

    train_data = tf.placeholder(tf.float32, [batch_size, num_feats])
    train_labels_racism = tf.placeholder(tf.float32, [batch_size, 2])
    train_labels_sexism = tf.placeholder(tf.float32, [batch_size, 2])
    # train_labels_neither = tf.placeholder(tf.float32, [batch_size, 2])

    dev_racism_data = tf.constant(X_racism_dev.astype(np.float32))
    test_racism_data = tf.constant(X_racism_test.astype(np.float32))

    dev_sexism_data = tf.constant(X_sexism_dev.astype(np.float32))
    test_sexism_data = tf.constant(X_sexism_test.astype(np.float32))

    weights1 = tf.Variable(tf.truncated_normal([num_feats, hidden_size]))
    biases1 = tf.Variable(tf.zeros([hidden_size]))

    weights_racism = tf.Variable(tf.truncated_normal([hidden_size, 2]))
    biases_racism = tf.Variable(tf.zeros([2]))

    weights_sexism = tf.Variable(tf.truncated_normal([hidden_size, 2]))
    biases_sexism = tf.Variable(tf.zeros([2]))

    weights_neither = tf.Variable(tf.truncated_normal([hidden_size, 2]))
    biases_neither = tf.Variable(tf.zeros([2]))

    layer1 = tf.nn.relu(tf.add(tf.matmul(train_data, weights1), biases1))

    racism_logits = tf.add(tf.matmul(layer1, weights_racism), biases_racism)
    racism_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_racism, logits=racism_logits)) + l2_lambda * (
    tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(weights_racism) + tf.nn.l2_loss(biases_racism))
    racism_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(racism_loss, global_step=global_step)

    sexism_logits = tf.add(tf.matmul(layer1, weights_racism), biases_racism)
    sexism_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_sexism, logits=sexism_logits)) + l2_lambda * (
    tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(weights_sexism) + tf.nn.l2_loss(biases_sexism))
    sexism_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(sexism_loss, global_step=global_step)

    # neither_logits = tf.add(tf.matmul(layer1,weights_neither),biases_neither)
    # neither_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_neither, logits=neither_logits)) + l2_lambda * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(weights_neither) + tf.nn.l2_loss(biases_neither))
    # neither_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(neither_loss,global_step=global_step)

    racism_prediction = tf.nn.softmax(racism_logits)
    sexism_prediction = tf.nn.softmax(sexism_logits)
    # neither_prediction = tf.nn.softmax(neither_logits)

    dev_racism_prediction = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(dev_racism_data, weights1), biases1)), weights_racism),
               biases_racism))
    test_racism_prediction = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(test_racism_data, weights1), biases1)), weights_racism),
               biases_racism))

    dev_sexism_prediction = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(dev_sexism_data, weights1), biases1)), weights_sexism),
               biases_sexism))
    test_sexism_prediction = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(test_sexism_data, weights1), biases1)), weights_sexism),
               biases_sexism))

    num_epochs = 30
    num_steps = (y_racism_train.shape[0] + y_sexism_train.shape[0]) * num_epochs // batch_size
    print(num_steps)


    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


    task = 0
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        racism_step = 0
        sexism_step = 0
        for step in range(num_steps):
            if task % 2 == 0:
                # Racism

                offset = (racism_step * batch_size) % (y_racism_train.shape[0] - batch_size)

                batch_data = X_racism_train[offset:(offset + batch_size)]
                batch_labels = y_racism_train[offset:(offset + batch_size), :]

                feed_dict = {train_data: batch_data, train_labels_racism: batch_labels}
                _, l, predictions = session.run([racism_optimizer, racism_loss, racism_prediction], feed_dict=feed_dict)

                if racism_step % 100 == 0:
                    print("Racism Minibatch loss at step %d: %f" % (step, l))
                    print("Racism Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

                    print("Racism VALIDATION accuracy: %.1f%%" % accuracy(dev_racism_prediction.eval(), y_racism_dev))

                racism_step += 1

            elif task % 2 == 1:
                # Sexism

                offset = (sexism_step * batch_size) % (y_sexism_train.shape[0] - batch_size)

                batch_data = X_sexism_train[offset:(offset + batch_size)]
                batch_labels = y_sexism_train[offset:(offset + batch_size), :]

                feed_dict = {train_data: batch_data, train_labels_sexism: batch_labels}
                _, l, predictions = session.run([sexism_optimizer, sexism_loss, sexism_prediction], feed_dict=feed_dict)

                if sexism_step % 100 == 0:
                    print("Sexism Minibatch loss at step %d: %f" % (step, l))
                    print("Sexism Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

                    print("Sexism VALIDATION accuracy: %.1f%%" % accuracy(dev_sexism_prediction.eval(), y_sexism_dev))

                sexism_step += 1

            task += 1

        print("FINAL Racism TEST accuracy: %.1f%%" % accuracy(test_racism_prediction.eval(), y_racism_test))
        print("FINAL Sexism TEST accuracy: %.1f%%" % accuracy(test_sexism_prediction.eval(), y_sexism_test))

        print(classification_report(y_racism_test, onehot(np.argmax(test_prediction.eval(), 1))))
        print(classification_report(y_external, onehot(np.argmax(external_prediction.eval(), 1))))