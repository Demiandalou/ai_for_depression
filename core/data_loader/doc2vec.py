# --------------------------------------------------------
# Get text feature (doc2vec) 
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------
import os
from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors
import sys
sys.path.append(os.path.join('.','utils'))
from audio_to_text import audio_to_text
import jieba
import gensim

def doc2vec(doc_path):
    '''from video or from existing transcript'''
    Docu=[]
    txtfiles=os.listdir(doc_path)
    for f in txtfiles:
        if f[0]=='.':
            txtfiles.remove(f)
    for f in txtfiles:
        print(os.path.join(doc_path,f))
        txtfile=open(os.path.join(doc_path,f),'r')
        res=''.join([line.strip('\n') for line in txtfile.readlines()])

        # split words using jieba
        sentence_seg = jieba.cut(res)
        result = ' '.join(sentence_seg)

        # remove stopwords
        stopwords = [line.strip() for line in open(os.path.join('utils','chi_stopwords.txt'),encoding='UTF-8').readlines()]
        res=' '.join([word for word in result.split(' ') if word not in stopwords])
        Docu.append(res.split(' '))

    # prepare documents for training
    count = 0
    sentence=[]
    for d in Docu: 
        sentence.append(gensim.models.doc2vec.TaggedDocument(d, [str(count)]))
        count+=1

    # train and save doc2vec model
    model = Doc2Vec(sentence, dm=1, size=100, window=8, min_count=1, workers=4)
    model_path=os.path.join('core','data_loader','doc2vec.bin')
    model.save(model_path)
    model = Doc2Vec.load(model_path)

    return model

# if __name__ == "__main__":
    # d2vmodel=doc2vec(os.path.join('utils','transcript'))
    # d2vmodel.random.seed(0)
    # v1 = d2vmodel.infer_vector(['感觉'],steps=6,alpha=0.025)
    # print(v1)

