# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
from utils import i2v_model_load, get_weight

class Config(object):

    """配置参数"""
    def __init__(self):
        
        self.save_path = './saved_dict/bert_and_graph_disam' + '.ckpt'           # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # 设备

        self.require_improvement = 1000                                        # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 5                                                    # epoch数
        self.batch_size = 4                                                    # mini-batch大小
        self.pad_size = 256                                                    # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                              # 学习率
        self.bert_path = './bert_pretrain_english'  # bert预训练模型路径
        self.item2vec_path = './item2vec_pretrain/item2vec.model'              # item2vec预训练路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.item2id, self.i2v_model = i2v_model_load(self.item2vec_path)
        self.item2vec = get_weight(self.i2v_model)
        self.hidden_size = 832   # 融合模型
                                           


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.pro_embeds = nn.Embedding.from_pretrained(config.item2vec, freeze=True)
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # self.fc = nn.Linear(16, 1)  # 点乘使用
        self.fc = nn.Linear(config.hidden_size, 1)  # 拼接使用
        print('dropout = 0.9')
        self.dp = nn.Dropout(0.9)  # 随机丢弃
        self.ru = nn.ReLU()  # 激活函数
        
    def forward(self, sentence):
        
        '''嵌入部分'''
        
        sentence1 = sentence[0]
        sentence2 = sentence[1]
        context1 = sentence1[0]  # 输入的句子
        mask1 = sentence1[2]     # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 0, 0]
        context2 = sentence2[0]                                                  
        mask2 = sentence2[2]                                                     
        _, pooled1 = self.bert(context1, attention_mask=mask1, output_all_encoded_layers=False)
        _, pooled2 = self.bert(context2, attention_mask=mask2, output_all_encoded_layers=False)
        
        item2vec1 = sentence1[3]
        item2vec2 = sentence2[3]
        i2v_tensor1 = self.pro_embeds(item2vec1)
        i2v_tensor2 = self.pro_embeds(item2vec2)
        
        # Bert+item2vec
        pooled1 = torch.cat((pooled1, i2v_tensor1), 1)
        pooled2 = torch.cat((pooled2, i2v_tensor2), 1)
        
        out = torch.mul(pooled1, pooled2)
        out = self.dp(out)
        out = self.ru(out)
        out = self.fc(out)
        out = out.squeeze(-1)
        return out

if __name__ == '__main__':
    
    print('ok')
    

    
    
     
    
    
    
     
    
    
    
    
    
    