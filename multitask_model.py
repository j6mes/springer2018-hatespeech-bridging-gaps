import tensorflow as tf
import numpy as np


def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    Taken from https://github.com/google/prettytensor/blob/
    a69f13998258165d6682a47a931108d974bab05e/prettytensor/layers.py
    :param n_inputs: The number of input nodes into each output.
    :param n_outputs: The number of output nodes for each input.
    :param uniform: If true use a uniform distribution, otherwise use a normal.
    :return: An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def classifier(inputs, y, n_classes):
    """
    Sequence classification, using last RNN output (if unidirectional) or last
    forward/backward outputs (if bidirectional).
    :param inputs:
    :param y:
    :param n_classes:
    :return:
    """
    input_dim = int(inputs.get_shape()[1])
    w = tf.get_variable("weights", [input_dim, n_classes],
                        initializer=xavier_init(input_dim, n_classes))
    b = tf.get_variable("biases", [n_classes],
                        initializer=xavier_init(1, n_classes))
    logits = tf.nn.xw_plus_b(inputs, w, b)
    preds = tf.nn.softmax(logits, name="preds")
    # y = tf.reshape(tf.one_hot(y, depth=n_classes), [-1, n_classes])
    y = tf.reshape(y, [-1, n_classes])
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(losses, name="batch_mean_loss")
    return preds, loss


def dense(inputs, hidden_size):
    input_dim = int(inputs.get_shape()[1])
    print(input_dim, hidden_size)
    w = tf.get_variable("weights", [input_dim, hidden_size],
                        initializer=xavier_init(input_dim, hidden_size))
    b = tf.get_variable("biases", [hidden_size],
                        initializer=xavier_init(1, hidden_size))
    logits = tf.nn.xw_plus_b(inputs, w, b)
    activations = tf.nn.tanh(logits)
    dropouts = tf.nn.dropout(activations, 0.8)
    return dropouts


class MTLModel:

    def __init__(self, args, tasks, input_length):
        self.predictions = {}
        self.losses = {}
        self.optimizers = {}
        self.labels = {}
        self.input_data = tf.placeholder(tf.float32, [None, input_length])

        shared_layers = args.get("shared_layers", 1)
        hidden_size = args.get("hidden_size", 50)

        tf.set_random_seed(1)

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.8)

        # SHARED LAYERS
        shared_layer = self.input_data
        for _ in range(shared_layers):
            shared_layer = dense(shared_layer, hidden_size)

        # TASK OPS
        for task in tasks:
            N_CLASSES = 2  # TODO make N_CLASSES dynamic
            with tf.variable_scope("output_"+task):
                self.labels[task] = tf.placeholder(tf.int32, [None, N_CLASSES])
                pred, loss = classifier(shared_layer, self.labels[task],
                                        N_CLASSES)
                self.predictions[task] = pred
                self.losses[task] = loss
                self.optimizers[task] = tf.train.GradientDescentOptimizer(
                    learning_rate).minimize(loss, global_step=global_step)

    def step(self, session, task, inputs, y=None, mode="train"):
        if mode == "train":
            return session.run([self.optimizers[task], self.losses[task],
                                self.predictions[task]],
                               feed_dict={self.input_data: inputs,
                                          self.labels[task]: y})
        elif mode == "predict-loss":
            return session.run([self.losses[task], self.predictions[task]],
                               feed_dict={self.input_data: inputs,
                                          self.labels[task]: y}
                               )
        elif mode == "predict":
            return session.run(self.predictions[task],
                               feed_dict={self.input_data: inputs})
        else:
            print("Mode {} not permitted. Possible values are 'train', "
                  "'predict-loss' and 'predict'".format(mode))
