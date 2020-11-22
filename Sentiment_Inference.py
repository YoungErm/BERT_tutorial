from dataset.inference_dataloader import preprocessing
from models.bert_sentiment_analysis import *
import numpy as np
import configparser
import os
import json
import warnings
import time
import metrics

all_preds = []

class Sentiment_Analysis:
    def __init__(self, max_seq_len,
                 batch_size,
                 with_cuda=True, # 是否使用GPU, 如未找到GPU, 则自动切换CPU
                 ):
        config_ = configparser.ConfigParser()
        config_.read("./config/sentiment_model_config.ini")
        self.config = config_["DEFAULT"]
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size
        # 加载字典
        with open(self.config["word2idx_path"], "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # 允许的最大序列长度
        self.max_seq_len = max_seq_len
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        # 初始化BERT情感分析模型
        self.bert_model = Bert_Sentiment_Analysis(config=bertconfig)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 开去evaluation模型, 关闭模型内部的dropout层
        self.bert_model.eval()

        # 初始化位置编码
        self.hidden_dim = bertconfig.hidden_size
        self.positional_enc = self.init_positional_encoding()
        # 扩展位置编码的维度, 留出batch维度,
        # 即positional_enc: [batch_size, embedding_dimension]
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        # 初始化预处理器
        self.process_batch = preprocessing(hidden_dim=bertconfig.hidden_size,
                                           max_positions=max_seq_len,
                                           word2idx=self.word2idx)
        # 加载BERT预训练模型
        self.load_model(self.bert_model, dir_path=self.config["state_dict_dir"])


    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        # 归一化
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc


    def load_model(self, model, dir_path="../output"):

        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        # checkpoint_dir = './bert_state_dict/'
        checkpoint = torch.load(checkpoint_dir)
        # 情感分析模型刚开始训练的时候, 需要载入预训练的BERT,
        # 这是我们不载入模型原本用于训练Next Sentence的pooler
        # 而是重新初始化了一个
        # model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        model.load_state_dict(checkpoint["model_state_dict"],False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded!".format(checkpoint_dir))

    def __call__(self, text_list, batch_size=1, threshold=.52):
        """
        :param text_list:
        :param batch_size: 为了注意力矩阵的可视化, batch_size只能为1, 即单句
        :return:
        """
        # 异常判断
        if isinstance(text_list, str):
            text_list = [text_list, ]
        len_ = len(text_list)
        text_list = [i for i in text_list if len(i) != 0]
        if len(text_list) == 0:
            raise NotImplementedError("输入的文本全部为空, 长度为0!")
        if len(text_list) < len_:
            warnings.warn("输入的文本中有长度为0的句子, 它们将被忽略掉!")

        # max_seq_len=self.max_seq_len+2 因为要留出cls和sep的位置
        max_seq_len = max([len(i) for i in text_list])
        # 预处理, 获取batch
        texts_tokens, positional_enc = \
            self.process_batch(text_list, max_seq_len=max_seq_len)
        # 准备positional encoding
        positional_enc = torch.unsqueeze(positional_enc, dim=0).to(self.device)

        # 正向
        n_batches = math.ceil(len(texts_tokens) / batch_size)

        # 数据按mini batch切片过正向, 这里为了可视化所以吧batch size设为1
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            # 切片
            texts_tokens_ = texts_tokens[start: end].to(self.device)


            predictions = self.bert_model.forward(text_input=texts_tokens_,
                                                  positional_enc=positional_enc,
                                                  )
            predictions = np.ravel(predictions.detach().cpu().numpy()).tolist()
            print(predictions)
            print("122",type(predictions))
            for text, pred in zip(text_list[start: end], predictions):
                self.sentiment_print_func(text, pred, threshold)

    def sentiment_print_func(self, text, pred, threshold):
        print(text)
        all_preds.append(pred)
        if pred >= threshold:
            print("正样本, 输出值{:.2f}".format(pred))
        else:
            print("负样本, 输出值{:.2f}".format(pred))
        print("----------")


    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

def read_predict(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = f.readlines()
        #lines = []
        test_list = []
        for line in reader[:10]:
            split_line = line.replace('\n', '').split('\t')
            #lines.append(split_line)
            test_list.append(split_line[1])
        return test_list


def write(write_list):
    filename = "{}.txt".format(time.strftime("%m-%d_%H-%M_preds", time.localtime()))
    output_pred_file = os.path.join('./outputs', filename)
    with open(output_pred_file, "w") as writer:
        writer.write('SessionId,Probability\n')
        for id, element in enumerate(write_list):
            writer.write(str(id) + "," + str(element) + '\n')


if __name__ == '__main__':
    metrics.seed_everything(42)
    model = Sentiment_Analysis(max_seq_len=300, batch_size=1)

    test_list = read_predict('./TianMadata/predict.txt')

    model(test_list)
    write(all_preds)