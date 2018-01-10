import os

from common.dataset.data_set import DataSet, CompositeDataset
from common.dataset.reader import CSVReader, JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, print_evaluation, exp_lr_scheduler
from common.util.random import SimpleRandom

from hatemtl.features.label_schema import WaseemLabelSchema, WaseemHovyLabelSchema, DavidsonLabelSchema, \
    DavidsonToZLabelSchema
from hatemtl.features.formatter import TextAnnotationFormatter, DavidsonFormatter
from hatemtl.features.feature_function import UnigramFeatureFunction, BigramFeatureFunction, CharNGramFeatureFunction
from hatemtl.model.multi_layer import MLP

from torch import nn, autograd

import torch
from hatemtl.features.preprocessing import preprocess as pp


def model_exists(mname):
    if not os.path.exists("models"):
        os.mkdir("models")
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

if __name__ == "__main__":

    SimpleRandom.set_seeds()
    mname = "expt7"

    csvreader = CSVReader(encoding="ISO-8859-1")
    df = DavidsonFormatter(DavidsonToZLabelSchema(),preprocessing=pp)

    davidson_tr_dataset = DataSet(os.path.join("data", "davidson.tr.csv"), formatter=df, reader=csvreader, name="davidson_train")
    davidson_dv_dataset = DataSet(os.path.join("data", "davidson.dv.csv"), formatter=df, reader=csvreader, name="davidson_dev")
    davidson_te_dataset = DataSet(os.path.join("data", "davidson.te.csv"), formatter=df, reader=csvreader, name="davidson_test")

    davidson_tr_dataset.read()
    davidson_dv_dataset.read()
    davidson_te_dataset.read()

    features = Features([UnigramFeatureFunction(naming=mname),
                         BigramFeatureFunction(naming=mname),
                         CharNGramFeatureFunction(1,naming=mname),
                         CharNGramFeatureFunction(2,naming=mname),
                         CharNGramFeatureFunction(3,naming=mname)
                         ])

    train_fs, dev_fs, test_fs = features.load(davidson_tr_dataset, davidson_dv_dataset, davidson_te_dataset)

    print("Number of features: {0}".format(train_fs[0].shape[1]))
    model = MLP(train_fs[0].shape[1],20,3)

    if gpu():
        model.cuda()

    if model_exists(mname) and os.getenv("TRAIN").lower() not in ["y","1","t","yes"]:
        model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    else:
        train(model, train_fs, 50, 1e-3, 30, dev=dev_fs, early_stopping=EarlyStopping(mname),
              lr_schedule=lambda a, b: exp_lr_scheduler(a, b, 0.5, 5))
        torch.save(model.state_dict(), "models/{0}.model".format(mname))


    if not os.path.exists("logs/experiment7"):
        os.makedirs("logs/experiment7")

    print_evaluation(model,dev_fs, DavidsonLabelSchema(),log="logs/experiment7/dev.jsonl")
    print_evaluation(model,test_fs, DavidsonLabelSchema(),log="logs/experiment7/test.jsonl")