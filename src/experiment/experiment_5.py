import os

from common.dataset.data_set import DataSet, CompositeDataset
from common.dataset.reader import CSVReader, JSONLineReader
from common.features.feature_function import Features
from common.training.options import gpu
from common.training.run import train, print_evaluation

from hatemtl.features.label_schema import WaseemLabelSchema, WaseemHovyLabelSchema, DavidsonLabelSchema, \
    DavidsonToZLabelSchema
from hatemtl.features.formatter import TextAnnotationFormatter, DavidsonFormatter
from hatemtl.features.feature_function import UnigramFeatureFunction, BigramFeatureFunction, CharNGramFeatureFunction
from hatemtl.model.multi_layer import MLP

from torch import nn, autograd

import torch


def model_exists(mname):
    if not os.path.exists("models"):
        os.mkdir("models")
    return os.path.exists(os.path.join("models","{0}.model".format(mname)))

if __name__ == "__main__":

    mname = "expt5"



    sexism_file = os.path.join("data","sexism.json")
    racism_file = os.path.join("data","racism.json")
    neither_file = os.path.join("data","neither.json")
    waseem_hovy = os.path.join("data","amateur_expert.json")


    csvreader = CSVReader(encoding="ISO-8859-1")
    jlr = JSONLineReader()
    formatter = TextAnnotationFormatter(WaseemLabelSchema())
    formatter2 = TextAnnotationFormatter(WaseemHovyLabelSchema(),mapping={0:0,1:1,2:2,3:0})
    df = DavidsonFormatter(DavidsonToZLabelSchema(),mapping={0:0,1:1,2:2})


    datasets = [
        DataSet(file=sexism_file, reader=jlr, formatter=formatter),
        DataSet(file=racism_file, reader=jlr, formatter=formatter),
        DataSet(file=neither_file, reader=jlr, formatter=formatter),
        DataSet(file=waseem_hovy, reader=jlr, formatter=formatter2),
        ]

    waseem_composite = CompositeDataset()
    for dataset in datasets:
        dataset.read()
        waseem_composite.add(dataset)


    davidson = DataSet(os.path.join("data","davidson.clean.csv"),reader=csvreader,formatter=df)
    davidson.read()

    features = Features([UnigramFeatureFunction(naming=mname),
                         BigramFeatureFunction(naming=mname),
                         CharNGramFeatureFunction(1,naming=mname),
                         CharNGramFeatureFunction(2,naming=mname),
                         CharNGramFeatureFunction(3,naming=mname)
                         ])

    train_fs, _, test_fs = features.load(waseem_composite, None, davidson)

    print("Number of features: {0}".format(train_fs[0].shape[1]))
    model = MLP(train_fs[0].shape[1],100,davidson.num_classes())


    if gpu():
        model.cuda()

    if model_exists(mname) and os.getenv("TRAIN").lower() not in ["y","1","t","yes"]:
        model.load_state_dict(torch.load("models/{0}.model".format(mname)))
    else:
        train(model, train_fs, 2, 1e-3, 10)
        torch.save(model.state_dict(), "models/{0}.model".format(mname))

    print_evaluation(model,test_fs, DavidsonLabelSchema())