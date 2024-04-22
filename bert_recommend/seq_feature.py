# -*- coding: utf-8 -*-

import pandas as pd
from gensim.models.word2vec import Word2Vec
from collections import defaultdict

'''读取用户、商品id对照json数据'''

def process_dataset(json_path, select_cols, csv_path):
    
    if json_path.endswith('gz'):
        df = pd.read_json(json_path, lines=True, compression='gzip')
    else:
        df = pd.read_json(json_path, lines=True)
    df = df[select_cols]
    df.columns = ['userID', 'itemID', 'review', 'rating']  
    df.to_csv(csv_path)
    print('Json数据读取结束...')
    
'''商品序列数据读取，返回二维数组[[‘商品1’，‘商品2’，...], []]'''

def read_item_data():
    path = "./predata/music.csv"
    data = pd.read_csv(path)
    item_dict = defaultdict(list)  # 遍历数据集，将每个用户购买记录添加到一个list
    
    for i in range(len(data)):
        item_dict[data.iloc[i]['userID']].append(data.iloc[i]['itemID'])
    
    item_list = list(item_dict.values())
    print('******item序列加载结束！******')
    print('用户数量为:{}'.format(len(item_dict.keys())))
    return item_list


'''输入二维矩阵数据训练（[[], []]）'''

def train(item_seq, model_path):
    
    # 加载一个空模型
    model = Word2Vec(vector_size=64, min_count = 2)
    model.build_vocab(item_seq)                                                # 加载词表
    model.train(item_seq, total_examples=model.corpus_count, epochs=10)
    print('模型参数为：{}'.format(model))
    model.save(model_path)                                                     # 训练结果保存路径
    print('训练完成！')

if __name__ == '__main__':
    model_path = './item2vec_pretrain/item2vec.model'  # item转向量值训练模型保存路径
    
    '''训练部分'''
    
    item_seq = read_item_data()  # 读取全部商品序列数据
    print('*****开始训练模型......*****')
    train(item_seq, model_path)  # 训练模型并保存
















