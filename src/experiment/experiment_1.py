import os

from common.dataset.data_set import DataSet, CompositeDataset
from common.dataset.reader import CSVReader, JSONLineReader
from common.features.feature_function import Features
from common.training.early_stopping import EarlyStopping
from common.training.options import gpu
from common.training.run import train, print_evaluation, exp_lr_scheduler, train_mt
from common.util.random import SimpleRandom

from hatemtl.features.label_schema import WaseemLabelSchema, WaseemHovyLabelSchema, DavidsonLabelSchema, \
    DavidsonToZLabelSchema
from hatemtl.features.formatter import TextAnnotationFormatter, DavidsonFormatter
from hatemtl.features.feature_function import UnigramFeatureFunction, BigramFeatureFunction, CharNGramFeatureFunction
from hatemtl.model.multi_layer import MLP, MTMLP

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
    mname = "expt1"

    sexism_file_tr = os.path.join("data","waseem_s.tr.json")
    racism_file_tr = os.path.join("data","waseem_r.tr.json")
    neither_file_tr = os.path.join("data","waseem_n.tr.json")
    waseem_hovy_tr = os.path.join("data","amateur_expert.tr.json")

    sexism_file_de = os.path.join("data","waseem_s.dv.json")
    racism_file_de = os.path.join("data","waseem_r.dv.json")
    neither_file_de = os.path.join("data","waseem_n.dv.json")
    waseem_hovy_de = os.path.join("data","amateur_expert.dv.json")


    csvreader = CSVReader(encoding="ISO-8859-1")
    jlr = JSONLineReader()
    formatter = TextAnnotationFormatter(WaseemLabelSchema(),preprocessing=pp)
    formatter2 = TextAnnotationFormatter(WaseemHovyLabelSchema(),preprocessing=pp,mapping={0:0,1:1,2:2,3:0})
    df = DavidsonFormatter(DavidsonToZLabelSchema(),preprocessing=pp,mapping={0:0,1:1,2:2})


    datasets_tr = [
        DataSet(file=sexism_file_tr, reader=jlr, formatter=formatter,name=None),
        DataSet(file=racism_file_tr, reader=jlr, formatter=formatter,name=None),
        DataSet(file=neither_file_tr, reader=jlr, formatter=formatter,name=None),
        DataSet(file=waseem_hovy_tr, reader=jlr, formatter=formatter2,name=None)
    ]

    datasets_de = [
        DataSet(file=sexism_file_de, reader=jlr, formatter=formatter,name=None),
        DataSet(file=racism_file_de, reader=jlr, formatter=formatter,name=None),
        DataSet(file=neither_file_de, reader=jlr, formatter=formatter,name=None),
        DataSet(file=waseem_hovy_de, reader=jlr, formatter=formatter2,name=None)
    ]

    waseem_tr_composite = CompositeDataset(name="waseem_composite_train")
    for dataset in datasets_tr:
        dataset.read()
        waseem_tr_composite.add(dataset)

    waseem_de_composite = CompositeDataset(name="waseem_composite_dev")
    for dataset in datasets_de:
        dataset.read()
        waseem_de_composite.add(dataset)


    davidson_tr = DataSet(os.path.join("data","davidson.tr.csv"),reader=csvreader,formatter=df,name="davidson_train")
    davidson_tr.read()

    davidson_dv = DataSet(os.path.join("data","davidson.dv.csv"),reader=csvreader,formatter=df,name="davidson_dev")
    davidson_dv.read()

    davidson_te = DataSet(os.path.join("data","davidson.te.csv"),reader=csvreader,formatter=df,name="davidson_test")
    davidson_te.read()

    features = Features([UnigramFeatureFunction(naming=mname),
                         BigramFeatureFunction(naming=mname),
                         CharNGramFeatureFunction(1,naming=mname),
                         CharNGramFeatureFunction(2,naming=mname),
                         CharNGramFeatureFunction(3,naming=mname)
                         ])

    primary_train_fs, aux_train_fs, dev_fs, test_fs = features.load(waseem_tr_composite, davidson_tr, waseem_de_composite, davidson_te)

    print("Number of features in primary: {0}".format(primary_train_fs[0].shape[1]))
    print("Number of features aux (=): {0}".format(aux_train_fs[0].shape[1]))

    model = MTMLP(primary_train_fs[0].shape[1],20,3,3)

    if gpu():
        model.cuda()

    if model_exists(mname) and os.getenv("TRAIN").lower() not in ["y","1","t","yes"]:
        model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    else:
        train_mt(model, (primary_train_fs,aux_train_fs), 50, 1e-3, 30, dev=dev_fs, early_stopping=EarlyStopping(mname),
              lr_schedule=lambda a, b: exp_lr_scheduler(a, b, 0.5, 5))
        torch.save(model.state_dict(), "models/{0}.model".format(mname))

    if not os.path.exists("logs/experiment1"):
        os.makedirs("logs/experiment1")

    print_evaluation(model,dev_fs, WaseemLabelSchema(),log="logs/experiment1/dev.jsonl")
    print_evaluation(model,test_fs, WaseemLabelSchema(),log="logs/experiment1/test.jsonl")