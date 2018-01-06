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

import torch


def model_exists(mname):
    if not os.path.exists("models"):
        os.mkdir("models")
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

if __name__ == "__main__":
    SimpleRandom.set_seeds()
    mname = "expt5"

    sexism_file_te = os.path.join("data","waseem_s.te.json")
    racism_file_te = os.path.join("data","waseem_r.te.json")
    neither_file_te = os.path.join("data","waseem_n.te.json")
    waseem_hovy_te = os.path.join("data","amateur_expert.te.json")

    sexism_file_dv = os.path.join("data","waseem_s.dv.json")
    racism_file_dv = os.path.join("data","waseem_r.dv.json")
    neither_file_dv = os.path.join("data","waseem_n.dv.json")
    waseem_hovy_dv = os.path.join("data","amateur_expert.dv.json")


    csvreader = CSVReader(encoding="ISO-8859-1")
    jlr = JSONLineReader()
    formatter = TextAnnotationFormatter(WaseemLabelSchema(),preprocessing=pp)
    formatter2 = TextAnnotationFormatter(WaseemHovyLabelSchema(),preprocessing=pp,mapping={0:0,1:1,2:2,3:0})
    df = DavidsonFormatter(DavidsonToZLabelSchema(),preprocessing=pp)


    datasets_dv = [
        DataSet(file=sexism_file_dv, reader=jlr, formatter=formatter),
        DataSet(file=racism_file_dv, reader=jlr, formatter=formatter),
        DataSet(file=neither_file_dv, reader=jlr, formatter=formatter),
        DataSet(file=waseem_hovy_dv, reader=jlr, formatter=formatter2)
    ]

    datasets_te = [
        DataSet(file=sexism_file_te, reader=jlr, formatter=formatter),
        DataSet(file=racism_file_te, reader=jlr, formatter=formatter),
        DataSet(file=neither_file_te, reader=jlr, formatter=formatter),
        DataSet(file=waseem_hovy_te, reader=jlr, formatter=formatter2)
    ]

    waseem_dv_composite = CompositeDataset()
    for dataset in datasets_dv:
        dataset.read()
        waseem_dv_composite.add(dataset)

    waseem_te_composite = CompositeDataset()
    for dataset in datasets_te:
        dataset.read()
        waseem_te_composite.add(dataset)

    davidson_tr = DataSet(os.path.join("data","davidson.tr.csv"),reader=csvreader,formatter=df)
    davidson_tr.read()

    davidson_dv = DataSet(os.path.join("data","davidson.dv.csv"),reader=csvreader,formatter=df)
    davidson_dv.read()


    features = Features([UnigramFeatureFunction(naming=mname),
                         BigramFeatureFunction(naming=mname),
                         CharNGramFeatureFunction(1,naming=mname),
                         CharNGramFeatureFunction(2,naming=mname),
                         CharNGramFeatureFunction(3,naming=mname)
                         ])

    train_fs, dev_fs, test_fs = features.load(davidson_tr, davidson_dv, waseem_te_composite)

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

    print_evaluation(model,dev_fs, WaseemLabelSchema())
    print_evaluation(model,test_fs, WaseemLabelSchema())