# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from ast import literal_eval# list格式字符串转list
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
from tqdm import tqdm

'''item转为id，再转tensor'''

def i2v_model_load(item2vec_path):
    i2v_model = Word2Vec.load(item2vec_path)
    item2id = i2v_model.wv.key_to_index                                        # 获取item-id表
    return item2id, i2v_model

def get_index(item2id, sentence):                                              # 将样本中的词替换为id
    sequence = []
    for word in sentence:
        try:
            sequence.append(item2id[word]+2)                                   # +2 是因为加了pading和unknow
        except KeyError:
            sequence.append(1)                                                 # 匹配不到的 给id=1 
    return sequence

'''基于word2vec的属性向量化'''

def get_weight(w2v_model):    #获取初始化embedding层的表
    id2vec = w2v_model.wv.vectors       #获取id-embedding表
    id2vec = np.insert(id2vec, 0, values=np.zeros(id2vec.shape[1]), axis=0) # pading掉的id为0，向量为 全0
    id2vec = np.insert(id2vec, 1, values=np.ones(id2vec.shape[1]), axis=0)  # 匹配不到的id为1 ， 向量为 全1
    weight = torch.from_numpy(id2vec)
    return weight

'''将文本数据处理为bert可接受形态'''

PAD, CLS = '[PAD]', '[CLS]'                                                    # padding符号, bert中综合信息符号
def build_dataset(config, data_len):
    
    '''文本通过bert预训练模型转为id'''
    
    def token_to_bert(sentence, pad_size):
        #sentence = re.split(' ', sentence)
        token = config.tokenizer.tokenize(sentence)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:                                                           # 确定文本长度，多的切片，少了补零
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        return (token_ids, seq_len, mask)                                      # 返回一个元组形式的转换结果
    
    '''数据读取处理，形成基于预训练模型的编码，用户转为tensor的输入'''
    
    def load_dataset(pad_size):
        train, dev, test = [], [], []
        data_user = pd.read_csv('./predata/users.csv') 
        data_item = pd.read_csv('./predata/items.csv') 
        data_music = pd.read_csv('./predata/music.csv') 
        
        itme_content = {}   # 商品id-文本                                                   
        for i in range(len(data_item)):
            itme_content[data_item.iloc[i]['items_id']] = data_item.iloc[i]['comments_i']
            
        user_content = {}   # 用户id-（文本，购买商品id）                                                   
        for i in range(len(data_user)):
            item_list_single = literal_eval(data_user.iloc[i]['item_id_list'])  # 当前用户购买记录
            user_content[data_user.iloc[i]['users_id']] = (data_user.iloc[i]['comments_u'], item_list_single)
        
        # 遍历大数据集，构建训练格式 len(data_music)
        train_sum = 0
        for i in tqdm(range(data_len)):
            userID = data_music.iloc[i]['userID']  # 当前用户id
            itemID = data_music.iloc[i]['itemID']  # 当前商品id
            content_user = user_content[userID][0] # 用户评论文本
            
            if content_user:
                content_user_bert = token_to_bert(content_user, pad_size)  # 用户文本转为bert词典的编号
                content_item = itme_content[itemID]  #商品评论文本
                if content_item:
                    content_item_bert = token_to_bert(content_item, pad_size)  # 商品文本转为bert词典的编号
                    id_items_i2v = get_index(config.item2id, user_content[userID][1])  # 商品id形成用户的item2vec
                    id_item_i2v = get_index(config.item2id, itemID)  # 单个商品id形成商品的item2vec
                    label = data_music.iloc[i]['rating']  # 商品评分
                    
                    
                    '''混合数据为测试集合'''
                    
                    if train_sum < 0.8 * data_len:
                        train.append((content_user_bert, content_item_bert, id_items_i2v, id_item_i2v, label))
                        train_sum += 1
                    elif len(dev) < 0.1 * data_len:
                        dev.append((content_user_bert, content_item_bert, id_items_i2v, id_item_i2v, label))
                    else:
                        test.append((content_user_bert, content_item_bert, id_items_i2v, id_item_i2v, label))
        return train, dev, test
    
    train, dev, test = load_dataset(config.pad_size)
    return train, dev, test

'''迭代器，将数据分成batchsize大小不断输入模型'''

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False                                                   # 记录batch数量是否为整数
        print(len(batches))
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        #输入build_dataset(config)函数转化为id后的数据，并转为tensor，datas格式：(content_user_bert, content_item_bert, id_items_i2v, id_item_i2v, label)
        x1 = torch.LongTensor([_[0][0] for _ in datas]).to(self.device)        # user文本数据
        x2 = torch.LongTensor([_[1][0] for _ in datas]).to(self.device)        # item文本数据
        
        seq_len1 = torch.LongTensor([_[0][1] for _ in datas]).to(self.device)
        mask1 = torch.LongTensor([_[0][2] for _ in datas]).to(self.device)
        
        seq_len2 = torch.LongTensor([_[1][1] for _ in datas]).to(self.device)
        mask2 = torch.LongTensor([_[1][2] for _ in datas]).to(self.device)
        
        items_i2v = torch.LongTensor([_[2][0] for _ in datas]).to(self.device) # 用户数据
        item_i2v = torch.LongTensor([_[3][0] for _ in datas]).to(self.device)  # 商品数据
        
        y = torch.LongTensor([_[4] for _ in datas]).to(self.device)            # 标签
        return ((x1, seq_len1, mask1, items_i2v), (x2, seq_len2, mask2, item_i2v)), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]# 每次迭代，数据切分出batch_size大小
            self.index += 1# 切分索引变大一个，方便下次在正确位置切分
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

'''将用户id、评论集合、购买集合写出到users表，商品id、商品评论集合写出到items表'''

def write_comment():
    path = "./predata/music.csv"
    data = pd.read_csv(path)
    id_dict = defaultdict(list)  # 每个用户购买的商品集合
    content_dict_u = defaultdict(list)  # 每个用户的评论集合
    content_dict_i = defaultdict(list)  # 每个商品的评论集合
    
    for i in range(len(data)):
        id_dict[data.iloc[i]['userID']].append(data.iloc[i]['itemID'])
        content_dict_u[data.iloc[i]['userID']].append(str(data.iloc[i]['review']))
        content_dict_i[data.iloc[i]['itemID']].append(str(data.iloc[i]['review']))        

    print("数据切分结束！")
    
    #将数据写出到csv
    users, comments_u, item_id_list = [], [], []                                              
    items, comments_i = [], []                                                 
    
    # 用户id、用户评论集
    for id_u, content_u in content_dict_u.items():
        if len(content_u) > 5:
            comments_u.append(''.join(content_u[:5]))
        else:
            comments_u.append(''.join(content_u))
        users.append(id_u)
        item_id_list.append(str(id_dict[id_u]))
          
    print("用户表融合结束，准备写出！")  
    df1 = pd.DataFrame()
    df1['users_id'] = users  # 用户id
    df1['comments_u'] = comments_u  # 用户评论集合
    df1['item_id_list'] = item_id_list  # 商品集合
    df1.to_csv('./predata/users.csv')

    # 商品id、商品评论集
    for k, v in content_dict_i.items():
        items.append(k)
        if len(v) > 5:
            comments_i.append(''.join(v[:5]))
        else:
            comments_i.append(''.join(v))
            
    print("商品表融合结束，准备写出！")
    df2 = pd.DataFrame()
    df2['items_id'] = items
    df2['comments_i'] = comments_i
    df2.to_csv('./predata/items.csv')
        
if __name__ == '__main__':
    
    write_comment()
    
    
    
    
    
    
    
    
