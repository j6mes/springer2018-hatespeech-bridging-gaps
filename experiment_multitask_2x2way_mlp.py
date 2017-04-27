import random
import numpy as np
import tensorflow as tf

from dataset_reader import DataSet
from composite_dataset import CompositeDataset
import util

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
patience = 3

graph = tf.Graph()


def create_model(args):
    return MTLModel(args)


class MTLModel:

    def __init__(self, args):
        self.train_ops = {}
        self.dev_predict_ops = {}
        self.dev_loss_ops = {}
        self.test_predict_ops = {}
        self.train_inputs = {}
        self.train_labels = {}
        self.train_loss_ops = {}
        self.train_predict_ops = {}

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

        ####################################
        # ####### OPS FOR TEST DATA ###### #
        ####################################
        test_sexism_prediction = tf.nn.softmax(
            tf.add(tf.matmul(
                tf.nn.relu(tf.add(tf.matmul(test_sexism_data, weights1), biases1)),
                weights_sexism),
                   biases_sexism))
        test_racism_prediction = tf.nn.softmax(
            tf.add(tf.matmul(
                tf.nn.relu(tf.add(tf.matmul(test_racism_data, weights1), biases1)),
                weights_racism),
                   biases_racism))

        ####################################
        # ####### OPS FOR DEV DATA ####### #
        ####################################
        dev_sexism_logits = tf.add(tf.matmul(
            tf.nn.relu(tf.add(tf.matmul(dev_sexism_data, weights1), biases1)),
            weights_sexism), biases_sexism)
        dev_sexism_prediction = tf.nn.softmax(dev_sexism_logits)
        dev_sexism_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=dev_sexism_logits, labels=y_sexism_dev
        ))
        dev_racism_logits = tf.add(tf.matmul(
            tf.nn.relu(tf.add(tf.matmul(dev_racism_data, weights1), biases1)),
            weights_racism), biases_racism)
        dev_racism_prediction = tf.nn.softmax(dev_racism_logits)
        dev_racism_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=dev_racism_logits, labels=y_racism_dev
        ))
        self.train_ops["racism"] = racism_optimizer
        self.train_ops["sexism"] = sexism_optimizer
        self.train_loss_ops["racism"] = racism_loss
        self.train_loss_ops["sexism"] = sexism_loss
        self.train_predict_ops["racism"] = racism_prediction
        self.train_predict_ops["sexism"] = sexism_prediction
        self.dev_predict_ops["racism"] = dev_racism_prediction
        self.dev_predict_ops["sexism"] = dev_sexism_prediction
        self.test_predict_ops["racism"] = test_racism_prediction
        self.test_predict_ops["sexism"] = test_sexism_prediction
        self.train_inputs["racism"] = racism_data
        self.train_inputs["sexism"] = sexism_data
        self.train_labels["racism"] = train_labels_racism
        self.train_labels["sexism"] = train_labels_sexism
        self.dev_loss_ops["racism"] = dev_racism_loss
        self.dev_loss_ops["sexism"] = dev_sexism_loss

    def step(self, session, task, inputs, y=None, mode="train"):
        if mode == "train":
            return session.run([self.train_ops[task], self.train_loss_ops[task],
                                self.train_predict_ops[task]],
                               feed_dict={self.train_inputs[task]: inputs,
                                          self.train_labels[task]: y})
        elif mode == "predict-dev":
            return session.run([self.dev_loss_ops[task],
                                self.dev_predict_ops[task]])
        elif mode == "predict-test":
            return session.run(self.test_predict_ops[task])
        else:
            print("Mode {} not permitted. Possible values are 'train', "
                  "'predict-dev' and 'predict-test'".format(mode))


####################################
# ###### TRAINING SETTINGS ####### #
####################################
num_epochs = 30
num_steps = (y_racism_train.shape[0] + y_sexism_train.shape[0]) // batch_size
print(num_steps)

####################################
# ########## TRAINING ############ #
####################################
task = 0
args = {}  # TODO provide hyperparams
model = create_model(args)
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    early_stopping_curve = []

    for epoch in range(1, num_epochs+1):
        if util.early_stopping(early_stopping_curve, patience,
                               lower_is_better=True):
            print("Ending training due to early stop criterion. Optimal "
                  "performance reached at epoch {}".format(epoch-patience))
            break
        print("<<< EPOCH {} >>>".format(epoch))
        racism_step = 0
        sexism_step = 0
        X_racism_train, y_racism_train = \
            util.shuffle_data(X_racism_train, y_racism_train)
        X_sexism_train, y_sexism_train = \
            util.shuffle_data(X_sexism_train, y_sexism_train)
        l = 0
        epoch_stop_criterion = []
        dev_loss = 0
        for step in range(num_steps):
            if task % 2 == 0:
                # Racism

                offset = (racism_step * batch_size) % (y_racism_train.shape[0] - batch_size)

                batch_data = X_racism_train[offset:(offset + batch_size)]
                batch_labels = y_racism_train[offset:(offset + batch_size), :]

                _, l, predictions = model.step(session, "racism", batch_data, batch_labels)
                dev_loss, dev_predictions = model.step(session, "racism", None, mode="predict-dev")

                if racism_step % 100 == 0:
                    print("Racism Minibatch loss at step %d: %f" % (step, l))
                    print("Racism Minibatch accuracy: %.1f%%" % util.accuracy(predictions, batch_labels))
                    print("Racism VALIDATION accuracy: %.1f%%" % util.accuracy(dev_predictions, y_racism_dev))

                racism_step += 1

            elif task % 2 == 1:
                # Sexism

                offset = (sexism_step * batch_size) % (y_sexism_train.shape[0] - batch_size)

                batch_data = X_sexism_train[offset:(offset + batch_size)]
                batch_labels = y_sexism_train[offset:(offset + batch_size), :]

                _, l, predictions = model.step(session, "sexism", batch_data,
                                               batch_labels)
                dev_loss, dev_predictions = model.step(session, "sexism", None,
                                             mode="predict-dev")

                if sexism_step % 100 == 0:
                    print("Sexism Minibatch loss at step %d: %f" % (step, l))
                    print("Sexism Minibatch accuracy: %.1f%%" % util.accuracy(predictions, batch_labels))

                    print("Sexism VALIDATION accuracy: %.1f%%" % util.accuracy(dev_predictions, y_sexism_dev))

                sexism_step += 1
            # END STEP
            epoch_stop_criterion.append(dev_loss)
            task += 1
        # END EPOCH
        early_stopping_curve.append(np.array(epoch_stop_criterion).mean())

    test_racism_prediction = model.step(session, "racism", None,
                                        mode="predict-test")
    test_sexism_prediction = model.step(session, "sexism", None,
                                        mode="predict-test")
    print("FINAL Racism TEST accuracy: %.1f%%" % util.accuracy(test_racism_prediction, y_racism_test))
    print("FINAL Sexism TEST accuracy: %.1f%%" % util.accuracy(test_sexism_prediction, y_sexism_test))
