# coding: UTF-8

import torch
import torch.nn as nn
from pytorch_pretrained.optimization import BertAdam
from utils import build_dataset, build_iterator
from model import Config, Model

'''模型训练函数'''

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def train(config, model, train_iter, dev_iter, test_iter):
    model.train()
    param_optimizer = list(model.named_parameters())                           # 返回各层中参数名称和数据
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    
    model.train()
    best_loss = 1000                                                           # 设定一个较大损失值，观察模型训练情况。
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (x, labels) in enumerate(train_iter):
            # optimizer.zero_grad()
            outputs = model(x)
            model.zero_grad()
            # loss = nn.functional.mse_loss(outputs, labels.to(torch.float)) # mse loss
            loss = mean_absolute_error(outputs, labels.to(torch.float))
            if loss < best_loss:
                best_loss = loss
                print('当前最小损失为：{}'.format(best_loss))
            loss.backward()
            optimizer.step()   
    print('训练结束，开始测试！')
    evaluate(config, model, dev_iter)  # 平均损失测试

def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    evaluate(config, model, test_iter)
    
def evaluate(config, model, data_iter):
    # criterion  = nn.L1Loss(reduction='mean')      
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # loss = nn.functional.mse_loss(outputs, labels) # mse error
            loss = mean_absolute_error(outputs, labels) # mae
            loss_total += loss
    print('测试损失为：{}'.format(loss_total / len(data_iter)))
    return loss_total / len(data_iter)

if __name__ == '__main__':
    
    config = Config()
    data_len = 200  # 数据量
    print("data loading......")
    train_data, dev_data, test_data = build_dataset(config, data_len)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = ''
    print("training......")
    
    model = Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

















