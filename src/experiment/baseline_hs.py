import os

import torch
import torch.nn.functional as F

from dataset.batcher import Batcher
from dataset.data_set import DataSet, CompositeDataset
from dataset.formatter import TextAnnotationFormatter
from dataset.label_schema import WaseemLabelSchema, WaseemHovyLabelSchema
from dataset.reader import CSVReader, JSONLineReader
from model.multi_layer import MLP


def train(model,data,batch_size,lr,epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = 0

    batcher = Batcher(data,batch_size)

    for epoch in range(epochs):
        for batch,size in batcher:

            data,labels = batch

            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits,labels)

            loss.backward()
            optimizer.step()




if __name__ == "__main__":
    sexism_file = os.path.join("data","sexism.json")
    racism_file = os.path.join("data","racism.json")
    neither_file = os.path.join("data","neither.json")
    waseem_hovy = os.path.join("data","amateur_expert.json")


    csvreader = CSVReader(encoding="ISO-8859-1")
    jlr = JSONLineReader()
    formatter = TextAnnotationFormatter(WaseemLabelSchema())
    formatter2 = TextAnnotationFormatter(WaseemHovyLabelSchema())

    datasets = [
        DataSet(file=sexism_file, reader=jlr, formatter=formatter),
        DataSet(file=racism_file, reader=jlr, formatter=formatter),
        DataSet(file=neither_file, reader=jlr, formatter=formatter),
        DataSet(file=waseem_hovy, reader=jlr, formatter=formatter2),
        ]

    composite = CompositeDataset()
    for dataset in datasets:
        dataset.read()
        composite.add(dataset)

    model = MLP(10,10,composite.num_classes())

    
    train(model,composite.data,200,1e-3,10)