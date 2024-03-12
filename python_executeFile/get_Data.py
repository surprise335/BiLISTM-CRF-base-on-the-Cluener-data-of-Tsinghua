import json
from tqdm import tqdm
import torch

# 预设标签
unk_flag = '[UNK]'  # 未知
pad_flag = '[PAD]'  # 填充
start_flag = '[STA]'  # 开始
end_flag = '[END]'  # 结束

class Dataset():
    def __init__(self,train_file_json, test_file_json, MAX_LEN, tag2index):
        self.train_file_json = train_file_json
        self.test_file_json = test_file_json
        self.MAX_LEN = MAX_LEN
        self.tag2index = tag2index



    def read_json(self,train):
        """
        参数train =1则是读取text_json
        :param train:
        :return:
        """
        data_list = []
        if train ==1:
            file_path = self.test_file_json
        else:
            file_path = self.train_file_json
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_data = json.loads(line)
                    data_list.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

        return data_list

    def get_text_and_tag(self,data_list):
        """
        本函数是对cluener数据集的处理，将其转化为BIO标注类型
        传入read_json返回的值，可以获取到 text文本列表 和 tag实体BIO标注列表
        返回已经处理好的text序列化文本和tag，与有效位判断掩码mask
        :param data_list:
        :return:
        """
        text = []
        tag = []
        time = []
        for i in data_list:
            text.append(i.get('text'))  # 获取文本内容
            tag_ = []
            for yy in range(100):
                tag_.append('O')
            for entity, value in i.get('label').items():
                for key, time_len in value.items():
                    for j in range(len(time_len)):
                        sum = 1
                        tag_[time_len[j][0]] = 'B-' + entity
                        for i in range(time_len[j][1] - time_len[j][0]):
                            tag_[time_len[j][0] + sum] = 'I-' + entity
                            sum += 1

            tag.append(tag_)

        return text, tag

    def text_tag_to_index(self,text_data, time_batch, word2index):
        """
        传入
        :param text: text文本数据 一维列表形式
        :param time_batch: tagBIO标记(时间步) 二维列表
        :param word2index: 数据集序列化方法函数word2index
        :return:
        """
        # 预设索引
        unk_index = word2index.get(unk_flag)
        pad_index = word2index.get(pad_flag)
        start_index = word2index.get(start_flag, 2)
        end_index = word2index.get(end_flag, 3)

        texts, tags, masks = [], [], []

        n_rows = len(text_data)  # 行数
        for row in tqdm(range(n_rows)):  # 文本对应的索引
            text = text_data[row]
            tag = time_batch[row]
            text_index = [start_index] + [word2index.get(w, unk_index) for w in text] + [end_index]
            # 标签对应的索引
            tag_index = [0] + [self.tag2index.get(t) for t in tag] + [0]
            # 填充或截断句子至标准长度
            if len(text_index) < self.MAX_LEN or len(tag_index) < self.MAX_LEN:  # 句子短，填充
                pad_len = self.MAX_LEN - len(text_index)
                text_index += pad_len * [pad_index]
                tag_index += pad_len * [0]
            if len(text_index) > self.MAX_LEN:  # 句子长，截断
                text_index = text_index[:self.MAX_LEN - 1] + [end_index]
            if len(tag_index) > self.MAX_LEN:
                tag_index = tag_index[:self.MAX_LEN - 1] + [0]

            # 转换为mask
            def _pad2mask(t):
                return 0 if t == pad_index else 1

            # mask
            mask = [_pad2mask(t) for t in text_index]

            # 加入列表中
            texts.append(text_index)
            tags.append(tag_index)
            masks.append(mask)

        # 转换为tensor
        texts = torch.LongTensor(texts)
        tags = torch.LongTensor(tags)
        masks = torch.tensor(masks, dtype=torch.uint8)
        return texts, tags, masks

    def text_to_index(self,test_data, word2index):
        # 预设索引
        unk_index = word2index.get(unk_flag)
        pad_index = word2index.get(pad_flag)
        start_index = word2index.get(start_flag, 2)
        end_index = word2index.get(end_flag, 3)

        texts, masks = [], []

        n_rows = len(test_data)  # 行数
        for row in tqdm(range(n_rows)):  # 文本对应的索引
            text = test_data[row]
            try:
                text_index = [start_index] + [word2index.get(w, unk_index) for w in text] + [end_index]
                # 填充或截断句子至标准长度
                if len(text_index) < self.MAX_LEN:  # 句子短，填充
                    pad_len = self.MAX_LEN - len(text_index)
                    text_index += pad_len * [pad_index]
                if len(text_index) > self.MAX_LEN:  # 句子长，截断
                    text_index = text_index[:self.MAX_LEN - 1] + [end_index]

                # 转换为mask
                def _pad2mask(t):
                    return 0 if t == pad_index else 1

                # mask
                mask = [_pad2mask(t) for t in text_index]

                # 加入列表中
                texts.append(text_index)
                masks.append(mask)

            except Exception as e:
                print(text)
                print("this type is {}".format(type(text)))
                continue
        # 转换为tensor
        texts = torch.LongTensor(texts)
        masks = torch.tensor(masks, dtype=torch.uint8)
        return texts, masks