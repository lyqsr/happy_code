'''
    Encoder+Decoder+Attention is implemented from:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, channel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
                      nn.Conv2d(channel_size, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)),
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)),
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))

    def forward(self, input):
        # [n, channel_size, 32, 280] -> [n, 512, 1, 71]
        conv = self.cnn(input)
        return conv


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, output_size]
        output = output.view(T, b, -1)
        return output


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size  # 256
        self.output_size = output_size  # 5991
        self.dropout_p = dropout_p
        self.max_length = max_length  # 71

        self.embedding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.hidden_size)  # 5991 ---> 256
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  # 512 | 71
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)  # 512 | 256
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size)  # 256 256
        self.out = nn.Linear(self.hidden_size, self.output_size)  # 256 | 5991


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # 把 标签input 转化为 标签向量embedded # [1] ---> [1, 256]
        embedded = self.dropout(embedded)  # very good !!! # [1, 256]

        tmp_mtr = torch.cat((embedded, hidden[0]), 1)  # [1, 256] | [1, 256] ---> [1, 512]
        tmp = self.attn(tmp_mtr)  # 得到attention分布 # [1, 512] ---> [1, 71]
        attn_weights = F.softmax(tmp, dim=1)  # [1, 71]
        # attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)

        t_a = attn_weights.unsqueeze(1)  # [1, 71] ---> [1, 1, 71] # 升维 # 维度对齐
        t_b = encoder_outputs.permute(1, 0, 2)  # [71, 1, 256] ---> [1, 71, 256]
        attn_applied = torch.bmm(t_a, t_b)  # [1, 1, 71] vs [1, 71, 256] ---> [1, 1, 256]
        # attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))

        tmp = attn_applied.squeeze(1)  # [1, 1, 256] ---> [1, 256]
        output = torch.cat((embedded, tmp), 1)  # [1, 256] vs [1, 256] ---> [1, 512]
        # output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)  # [1, 512] ---> [1, 256] ---> [1, 1, 256]

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)  # Inputs: input, h_0 # Outputs: output, h_n # [1, 1, 256] vs [1, 1, 256] ---> [1, 1, 256] vs [1, 1, 256]

        tmp = self.out(output[0])  # [1, 256] ---> [1, 5991]
        output = F.log_softmax(tmp, dim=1)  # [1, 5991]
        # output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Encoder(nn.Module):
    def __init__(self, channel_size, hidden_size):
        super(Encoder, self).__init__()
        self.cnn = CNN(channel_size)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"

        # rnn feature
        conv = conv.squeeze(2)        # [b, c, 1, w] -> [b, c, w]
        conv = conv.permute(2, 0, 1)  # [b, c, w] -> [w, b, c]
        output = self.rnn(conv)
        return output


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = AttnDecoderRNN(hidden_size, output_size, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))  # 1 means one time step
        return result

