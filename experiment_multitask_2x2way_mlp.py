import random
import numpy as np
import tensorflow as tf
import multitask_model
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


def create_model(args, tasks):
    # TODO provide possibility to load saved model or build a new one
    return multitask_model.MTLModel(args, tasks, num_feats)


####################################
# ###### TRAINING SETTINGS ####### #
####################################
num_epochs = 1
num_steps = (y_racism_train.shape[0] + y_sexism_train.shape[0]) // batch_size
print(num_steps)

####################################
# ########## TRAINING ############ #
####################################
task = 0
args = {}  # TODO provide hyperparams
model = create_model(args, tasks=["racism", "sexism"])
with tf.Session() as session:
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
                dev_loss, dev_predictions = model.step(session, "racism", X_racism_dev, y_racism_dev, mode="predict-loss")

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
                dev_loss, dev_predictions = model.step(session, "sexism", X_sexism_dev, y_sexism_dev,
                                             mode="predict-loss")

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

    test_racism_prediction = model.step(session, "racism", X_racism_test,
                                        mode="predict")
    test_sexism_prediction = model.step(session, "sexism", X_sexism_test,
                                        mode="predict")
    print("FINAL Racism TEST accuracy: %.1f%%" % util.accuracy(test_racism_prediction, y_racism_test))
    print("FINAL Sexism TEST accuracy: %.1f%%" % util.accuracy(test_sexism_prediction, y_sexism_test))
