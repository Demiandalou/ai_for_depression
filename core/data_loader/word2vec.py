# --------------------------------------------------------
# Get text feature (word2vec) 
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

import os
from gensim.models import Word2Vec,KeyedVectors
from gensim.models import word2vec as w2v

import sys
sys.path.append(os.path.join('.','utils'))
from audio_to_text import audio_to_text
import jieba


def word2vec(video_path):
    '''from video or from existing transcript'''
    # res=audio_to_text(video_path) # to get transcript from video, uncomment this and comment the line below
    txtfile=open(video_path.split('/')[-1].split('.')[0]+'.txt','r')
    res=''.join([line.strip('\n') for line in txtfile.readlines()])

    # split words using jieba
    sentence_seg = jieba.cut(res)
    result = ' '.join(sentence_seg)

    # remove stopwords
    stopwords = [line.strip() for line in open(os.path.join('utils','chi_stopwords.txt'),encoding='UTF-8').readlines()]
    res=' '.join([word for word in result.split(' ') if word not in stopwords])

    with open('tmp.txt', 'w',encoding="utf-8") as f2:
        f2.write(result)
    sentence = w2v.LineSentence('tmp.txt')
    model= Word2Vec(sentence,min_count=1)
    if os.path.exists("tmp.txt"):
        os.remove("tmp.txt")

    model_path=os.path.join('core','data_loader','word2vec.bin')
    model.wv.save_word2vec_format(model_path)
    model = KeyedVectors.load_word2vec_format(model_path)

    # similar_words = model.most_similar('回来')
    # print(similar_words)	
    return model

# if __name__ == "__main__":
#     DATA='/Users/mac/Desktop/my/CI/Depression/BSdata'
#     word2vec(os.path.join(DATA,'2020-8-14-38.mov'))