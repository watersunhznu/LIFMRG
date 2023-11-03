from models import MLC, EncoderCNN, SentenceLSTM, WordLSTM, Tag_fuse
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--int_stop_dim', type=int, default=64,
                    help='intermediate state dimension of stop vector network')
parser.add_argument('--sent_hidden_dim', type=int, default=512, help='hidden state dimension of sentence LSTM')
parser.add_argument('--sent_input_dim', type=int, default=1024, help='dimension of input to sentence LSTM')
parser.add_argument('--word_hidden_dim', type=int, default=512, help='hidden state dimension of word LSTM')
parser.add_argument('--word_input_dim', type=int, default=1024, help='dimension of input to word LSTM')
parser.add_argument('--att_dim', type=int, default=64,
                    help='dimension of intermediate state in co-attention network')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in word LSTM')

parser.add_argument('--lambda_sent', type=int, default=1,
                    help='weight for cross-entropy loss of stop vectors from sentence LSTM')
parser.add_argument('--lambda_word', type=int, default=1,
                    help='weight for cross-entropy loss of words predicted from word LSTM with target words')
parser.add_argument('--lambda_tag', type=int, default=10,
                    help='weight for cross-entropy loss of tag')
parser.add_argument('--lambda_l1', type=int, default=5e-7,
                    help='weight for l1 regularization')

parser.add_argument('--fl_alpha', type=float, default=0.2, help='')
parser.add_argument('--fl_gamma', type=int, default=3, help='')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batch')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the instances in dataset')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloader')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train the model')
parser.add_argument('--learning_rate_cnn', type=int, default=1e-5, help='learning rate for CNN Encoder')
parser.add_argument('--learning_rate_mlc', type=int, default=1e-5, help='learning rate for MLC Encoder')
parser.add_argument('--learning_rate_lstm', type=int, default=1e-5, help='learning rate for LSTM Decoder')

args = parser.parse_args()
tags_l = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
          'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
          'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
args.device = torch.device('cpu')
encoder = EncoderCNN()
mlc = MLC(tags_l, 256, 768, 4146, args.device)
sentLSTM = SentenceLSTM(768, 512, 64, 1024, 512, 64)
wordLSTM = WordLSTM(512, 512, 4146, 1)
# model = Tag_fuse(tags_l, 11322, args)

total = []
total.append(sum([param.nelement() for param in encoder.parameters()]))
total.append(sum([param.nelement() for param in mlc.parameters()]))
total.append(sum([param.nelement() for param in sentLSTM.parameters()]))
total.append(sum([param.nelement() for param in wordLSTM.parameters()]))
# topic=torch.randn([16,512])
# captions=torch.randn([16,44]).long()+3
# wordLSTM.sample(topic,captions)
# print(sum([param.nelement() for param in model.parameters()]))
print([i / 1e6 for i in total], sum(total) / 1e6)
