import json
import random
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from common.training.batcher import Batcher, prepare, prepare_with_labels
from common.util.random import SimpleRandom


def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

def evaluate(model,data,labels,batch_size):
    predicted = predict(model,data,batch_size)
    return accuracy_score(labels,predicted.data.numpy().reshape(-1))


def evaluate_mt(model,data,labels,batch_size,id):
    predicted = predict_mt(model,data,batch_size,id)
    return accuracy_score(labels,predicted.data.numpy().reshape(-1))

def predict(model, data, batch_size):
    batcher = Batcher(data, batch_size)

    predicted = []
    for batch, size, start, end in batcher:
        d = prepare(batch)
        model.eval()
        logits = model(d).cpu()

        predicted.extend(torch.max(logits, 1)[1])
    return torch.stack(predicted)


def predict_mt(model, data, batch_size, id):
    batcher = Batcher(data, batch_size)

    predicted = []
    for batch, size, start, end in batcher:
        d = prepare(batch)
        model.eval()
        logits = model(d)[id].cpu()

        predicted.extend(torch.max(logits, 1)[1])
    return torch.stack(predicted)

def train(model, fs, batch_size, lr, epochs,dev=None, clip=None, early_stopping=None,l2=1e-5,lr_schedule=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    data, labels = fs
    if dev is not None:
        dev_data,dev_labels = dev

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_data = 0

        shuffle(data,labels)

        batcher = Batcher(data, batch_size)

        for batch, size, start, end in batcher:
            d,gold = prepare_with_labels(batch,labels[start:end])

            model.train()
            optimizer.zero_grad()
            logits = model(d)

            loss = F.cross_entropy(logits, gold)
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data += size

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            optimizer.step()

        if lr_schedule is not None:
            optimizer = lr_schedule(optimizer,epoch)

        print("Average epoch loss: {0}".format((epoch_loss/epoch_data).data.numpy()))

        #print("Epoch Train Accuracy {0}".format(evaluate(model, data, labels, batch_size)))
        if dev is not None:
            acc = evaluate(model,dev_data,dev_labels,batch_size)
            print("Epoch Dev Accuracy {0}".format(acc))

            if early_stopping is not None and early_stopping(model,acc):
                early_stopping.set_best_state(model)
                break

    if early_stopping is not None:
        early_stopping.set_best_state(model)



def train_mt(model, training_datasets, batch_size, lr, epochs,dev=None,
             clip=None, early_stopping=None, l2=1e-5, lr_schedule=None,
             batches_per_epoch=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    print(training_datasets)
    print(len(training_datasets))
    print([len(ds[0]) for ds in training_datasets])
    print(sum([len(ds[0]) for ds in training_datasets]))
    if dev is not None:
        dev_data,dev_labels = dev

    if batches_per_epoch is None:
        batches_per_epoch = sum([len(dataset[0]) for dataset
                                 in training_datasets]) // batch_size
    print("Batches per epoch:", batches_per_epoch)
    batches = []

    for training_dataset in training_datasets:
        data,labels = training_dataset
        shuffle(data,labels)
        batcher = Batcher(data, batch_size)
        batches.append(batcher)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_data = 0
        for b in range(batches_per_epoch):
            task_id = random.choice(range(len(training_datasets)))
            batcher = batches[task_id]
            # for idx,batcher in enumerate(zip(*batches)):
            dataset_id = task_id  # %len(training_datasets)
            data,labels = training_datasets[dataset_id]

            batch, size, start, end = batcher.next_loop()
            # for batch, size, start, end in batcher:
            d,gold = prepare_with_labels(batch,labels[start:end])

            model.train()
            optimizer.zero_grad()
            logits_list = model(d)

            logits = logits_list[dataset_id]
            loss = F.cross_entropy(logits, gold)
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data += size

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            optimizer.step()

        if lr_schedule is not None:
            optimizer = lr_schedule(optimizer,epoch)

        print("Average epoch loss: {0}".format((epoch_loss/epoch_data).data.numpy()))

        #print("Epoch Train Accuracy {0}".format(evaluate(model, data, labels, batch_size)))
        if dev is not None:
            acc = evaluate_mt(model,dev_data,dev_labels,batch_size,dataset_id)
            print("Epoch Dev Accuracy {0}".format(acc))

            if early_stopping is not None and early_stopping(model,acc):
                early_stopping.set_best_state(model)
                break

    if early_stopping is not None:
        early_stopping.set_best_state(model)




def print_evaluation(model,data,ls,log=None):
    features,actual = data
    predictions = predict(model, features, 500).data.numpy().reshape(-1).tolist()

    labels = [ls.idx[i] for i, _ in enumerate(ls.idx)]

    actual = [labels[i] for i in actual]
    predictions = [labels[i] for i in predictions]

    print(accuracy_score(actual, predictions))
    print(classification_report(actual, predictions))
    print(confusion_matrix(actual, predictions))

    data = zip(actual,predictions)
    if log is not None:
        f = open(log, "w+")
        for a,p in data:
            f.write(json.dumps({"actual": a, "predicted": p}) + "\n")
        f.close()
