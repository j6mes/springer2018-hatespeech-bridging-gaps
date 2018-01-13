import os

from common.util.bpe import BPETransformer
from hatemtl.features.feature_function import EmbeddingFeatureFunction, UnigramFeatureFunction, BigramFeatureFunction, \
    CharNGramFeatureFunction


def is_embedding_model():
    return os.getenv("EMBEDDING","").lower() in ["y","1",1,"yes","t","true"]

def is_huge_model():
    return os.getenv("HUGE","").lower() in ["y","1",1,"yes","t","true"]


def is_large_model():
    return os.getenv("LARGE","").lower() in ["y","1",1,"yes","t","true"]




def get_model_shape():
    if is_embedding_model() or is_large_model():
        return [20,50,20]
    else:
        return [20]



def get_feature_functions(mname, BASE_DIR="."):

    ffs = []
    if is_embedding_model():
        bpe_embeddings_vocab = BASE_DIR + "/res/en.wiki.bpe.op3000.d300.w2v.vocab"
        bpe_embeddings_file = BASE_DIR + "/res/en.wiki.bpe.op3000.d300.w2v.txt"
        bpe_transformer = BPETransformer(bpe_embeddings_file)

        ffs.extend([EmbeddingFeatureFunction(bpe_embeddings_file,
                                                  preprocessors=[bpe_transformer],
                                                  naming=mname)])
    elif not is_embedding_model() or is_huge_model():
        ffs.extend([UnigramFeatureFunction(naming=mname),
             BigramFeatureFunction(naming=mname),
             CharNGramFeatureFunction(1,naming=mname),
             CharNGramFeatureFunction(2,naming=mname),
             CharNGramFeatureFunction(3,naming=mname)
             ])
    return ffs


def create_log_dir(mname):
    if not os.path.exists("logs/{0}".format(mname)):
        os.makedirs("logs/{0}".format(mname))