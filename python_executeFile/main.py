from sklearn.metrics import f1_score
from get_Data import Dataset
import pandas as pd
import torch
import pickle
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from bilstm_crf import BiLSTM_CRF, NerDataset, NerDatasetTest
import os
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



train_file_path = '../dataset/cluener_public/train.json'  #训练数据json
test_file_path ='../dataset/cluener_public/test.json'    #测试数据json
WORD2INDEX_PATH = '../data/word2index.pkl'
VOCAB_PATH = '../data/vocab.txt'                         #词典路径
MODEL_PATH = '../model/model.pkl'                   #模型储存路径


MAX_LEN =50
BATCH_SIZE = 8
EMBEDDING_DIM = 120
HIDDEN_DIM = 12
EPOCH = 5


#BIO实体定义
tag2index = {
    "O": 0,  # 其他
    "B-address": 1, "I-address": 2,  # 地址实体
    "B-book": 3, "I-book": 4,  #书实体
    "B-company": 5, "I-company": 6,  #公司
    "B-government": 7, "I-government": 8,  # 政府
    "B-movie": 9, "I-movie": 10,  # 电影
    "B-name": 11, "I-name": 12,  #人名
    "B-organization": 13, "I-organization": 14,  # 组织机构
    "B-position": 15, "I-position": 16,  # 职称
    "B-scene": 17, "I-scene": 18,  #场景
    "B-game": 19, "I-game": 20,  #游戏

}


def get_f1_score(tags, masks, predictions):
    final_tags = []
    final_predictions = []
    tags = tags.to('cpu').data.numpy().tolist()
    masks = masks.to('cpu').data.numpy().tolist()
    for index in range(BATCH_SIZE):
        length = masks[index].count(1)  # 未被mask的长度
        final_tags += tags[index][1:length - 1]  # 去掉头和尾，有效tag，最大max_len-2
        final_predictions += predictions[index][1:length - 1]  # 去掉头和尾，有效mask，最大max_len-2

    f1 = f1_score(final_tags, final_predictions, average='micro')  # 取微平均
    return f1


# 生成word2index词典
def create_word2index():
    if not os.path.exists(WORD2INDEX_PATH):  # 如果文件不存在，则生成word2index
        word2index = dict()
        with open(VOCAB_PATH, 'r', encoding='utf8') as f:
            for word in f.readlines():
                word2index[word.strip()] = len(word2index) + 1
        with open(WORD2INDEX_PATH, 'wb') as f:
            pickle.dump(word2index, f)

def train(train_dataloader, model, optimizer, epoch):
    for i, batch_data in tqdm(enumerate(train_dataloader)):
        texts = batch_data['texts'].to(DEVICE)
        tags = batch_data['tags'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)

        loss, predictions = model(texts, tags, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            micro_f1 = get_f1_score(tags, masks, predictions)
            print(f'Epoch:{epoch} | i:{i} | loss:{loss.item()} | Micro_F1:{micro_f1}')

def train_execute(): #训练执行函数
    text, tag = Data.get_text_and_tag(data_list)  # 获取text文本和tags时间步长标签
    create_word2index()
    with open(WORD2INDEX_PATH, 'rb') as f:
        word2index = pickle.load(f)
    texts, tags, masks = Data.text_tag_to_index(text, tag, word2index)

    # 数据集装载
    train_dataset = NerDataset(texts, tags, masks)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # 构建模型
    model = BiLSTM_CRF(vocab_size=len(word2index), tag2index=tag2index, embedding_dim=EMBEDDING_DIM,
                       hidden_size=HIDDEN_DIM, padding_idx=1, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    print(f"GPU_NAME:{torch.cuda.get_device_name()} | Memory_Allocated:{torch.cuda.memory_allocated()}")
    # 模型训练
    for i in range(EPOCH):
        train(train_dataloader, model, optimizer, i)

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)

# 测试集预测实体标签
def test(test_dataset, word2index):
    texts, masks = Data.text_to_index(test_dataset, word2index)
    print(texts[0])
    print(masks[0])
    # 装载测试集
    dataset_test = NerDatasetTest(texts, masks)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    # 构建模型
    model = BiLSTM_CRF(vocab_size=len(word2index), tag2index=tag2index, embedding_dim=EMBEDDING_DIM,
                       hidden_size=HIDDEN_DIM, padding_idx=1, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    # 模型预测
    model.eval()
    predictions_list = []
    for i, batch_data in enumerate(test_dataloader):
        texts = batch_data['texts'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        predictions = model(texts, None, masks)
        # print(len(texts), len(predictions))
        predictions_list.extend(predictions)
    print(predictions_list)
    # 将预测结果转换为文本格式
    entity_tag_list = []
    index2tag = {v: k for k, v in tag2index.items()}  # 反转字典
    for i, (text, predictions) in enumerate(zip(test_dataset, predictions_list)):
        # 删除首位和最后一位
        predictions.pop()
        predictions.pop(0)
        text_entity_tag = []
        for c, t in zip(text, predictions):
            if t != 0:
                text_entity_tag.append(c + index2tag[t])
        entity_tag_list.append(" ".join(text_entity_tag))  # 合并为str并加入列表中

    print(len(entity_tag_list))
    result_df = pd.DataFrame(data=entity_tag_list, columns=['result'])
    result_df.to_csv('../data/result.csv')




if __name__ == '__main__':
    # Data = Dataset(train_file_path, test_file_path, MAX_LEN,tag2index)
    # 预测环节
    # data_list = Data.read_json(1) #1是获取测试集
    # test_text = []
    # # 加载word2index文件
    # with open(WORD2INDEX_PATH, 'rb') as f:
    #     word2index = pickle.load(f)
    # # 文本转化为索引
    # for i in data_list:
    #     test_text.append(i.get('text'))
    # print(test_text)
    # test(test_text, word2index)
    predict = pd.read_csv('../data/result.csv', encoding='utf8')
    for i in predict.get('result'):
        if isinstance(i,str):
            a = i.split(' ')
            print(a)