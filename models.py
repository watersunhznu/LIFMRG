import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT
from einops import repeat
import timm


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        # cnn = models.resnet34(pretrained=True)
        # modules = list(cnn.children())[:-2]
        # self.cnn = nn.Sequential(*modules)
        # self.enc_dim = list(cnn.features.children())[-3].weight.shape[0]
        # self.vit = Vito(image_size=224, patch_size=32, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=2048, dropout=0.1,
        # emb_dropout=0.1)
        self.vit = timm.create_model('vit_base_patch32_224', pretrained=True)
        # self.batch_size = 16
        self.enc_dim = 768
        # self.linear = nn.Linear(768, 768)

    def forward(self, x):
        # x = self.cnn(x)  # (batch_size, enc_dim, enc_img_size, enc_img_size)
        # x = x.permute(0, 2, 3, 1).view(x.shape[0], -1, 512)
        x = self.vit(x)
        # x = self.linear(x)
        return x


class AttentionVisual(nn.Module):
    def __init__(self, vis_enc_dim, sent_hidden_dim, att_dim):
        super(AttentionVisual, self).__init__()

        self.enc_att = nn.Linear(vis_enc_dim, att_dim)
        self.dec_att = nn.Linear(sent_hidden_dim, att_dim)
        self.tanh = nn.Tanh()
        self.full_att = nn.Linear(att_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, vis_enc_output, dec_hidden_state):
        vis_enc_att = self.enc_att(vis_enc_output)  # (batch_size, num_pixels, att_dim)
        dec_output = self.dec_att(dec_hidden_state)  # (batch_size, att_dim)

        join_output = self.tanh(vis_enc_att + dec_output.unsqueeze(1))  # (batch_size, num_pixels, att_dim)

        join_output = self.full_att(join_output).squeeze(2)  # (batch_size, num_pixels)

        att_scores = self.softmax(join_output)  # (batch_size, num_pixels)

        att_output = torch.sum(att_scores.unsqueeze(2) * vis_enc_output, dim=1)

        return att_output, att_scores


class AttentionSemantic(nn.Module):
    def __init__(self, sem_enc_dim, sent_hidden_dim, att_dim):
        super(AttentionSemantic, self).__init__()

        self.enc_att = nn.Linear(sem_enc_dim, att_dim)
        self.dec_att = nn.Linear(sent_hidden_dim, att_dim)
        self.tanh = nn.Tanh()
        self.full_att = nn.Linear(att_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sem_enc_output, dec_hidden_state):
        sem_enc_output = self.enc_att(sem_enc_output)  # (batch_size, no_of_tags, att_dim)
        dec_output = self.dec_att(dec_hidden_state)  # (batch_size, att_dim)

        join_output = self.tanh(sem_enc_output + dec_output.unsqueeze(1))  # (batch_size, no_of_tags, att_dim)

        join_output = self.full_att(join_output).squeeze(2)  # (batch_size, no_of_tags)

        att_scores = self.softmax(join_output)  # (batch_size, no_of_tags)

        att_output = torch.sum(att_scores.unsqueeze(2) * sem_enc_output, dim=1)

        return att_output, att_scores


class SentenceLSTM(nn.Module):
    def __init__(self, vis_embed_dim, sent_hidden_dim, att_dim, sent_input_dim, word_input_dim, int_stop_dim):
        super(SentenceLSTM, self).__init__()

        # self.vis_att = AttentionVisual(vis_embed_dim, sent_hidden_dim, att_dim)
        # self.sem_att = AttentionSemantic(sem_embed_dim, sent_hidden_dim, att_dim)

        self.CoAtt = CoAttention()
        self.contextLayer = nn.Linear(vis_embed_dim, sent_input_dim)
        self.lstm = nn.LSTMCell(sent_input_dim, sent_hidden_dim, bias=True)

        self.sent_hidden_dim = sent_hidden_dim
        self.word_input_dim = word_input_dim

        self.topic_hid_layer = nn.Linear(sent_hidden_dim, word_input_dim)
        self.topic_context_layer = nn.Linear(sent_input_dim, word_input_dim)
        self.tanh1 = nn.Tanh()

        self.stop_prev_hid = nn.Linear(sent_hidden_dim, int_stop_dim)
        self.stop_cur_hid = nn.Linear(sent_hidden_dim, int_stop_dim)
        self.tanh2 = nn.Tanh()
        self.final_stop_layer = nn.Linear(int_stop_dim, 2)

    def forward(self, vis_enc_output, sem_enc_output, captions, device):
        """
        Forward propagation.

        :param vis_enc_output: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param captions: captions, a tensor of dimension (batch_size, max_no_of_sent, max_sent_len)
        :return: topic vector for word LSTM (batch_size, max_no_of_sent, word_input_dim), stop vector for each time step (batch_size, max_no_of_sent, 2)
        """
        batch_size = vis_enc_output.shape[0]
        vis_enc_dim = vis_enc_output.shape[-1]

        vis_enc_output = vis_enc_output.view(batch_size, -1, vis_enc_dim)  # (batch_size, num_pixels, vis_enc_dim)

        h = torch.zeros((batch_size, self.sent_hidden_dim)).to(device)
        c = torch.zeros((batch_size, self.sent_hidden_dim)).to(device)

        topics = torch.zeros((batch_size, captions.shape[1], self.word_input_dim)).to(device)
        ps = torch.zeros((batch_size, captions.shape[1], 2)).to(device)

        for t in range(captions.shape[1]):
            context_output, _, _ = self.CoAtt(vis_enc_output, sem_enc_output, h)  # (batch_size, vis_enc_dim),
            # (batch_size, num_pixels)

            # can concat with the semantic attention module output
            # context_output = self.contextLayer(vis_att_output)  # (batch_size, sent_input_dim)

            h_prev = h.clone()

            h, c = self.lstm(context_output, (h, c))  # (batch_size, sent_hidden_dim), (batch_size, sent_hidden_dim)

            topic = self.tanh1(
                self.topic_hid_layer(h) + self.topic_context_layer(context_output))  # (batch_size, word_input_dim)

            p = self.tanh2(self.stop_prev_hid(h_prev) + self.stop_cur_hid(h))  # (batch_size, int_stop_dim)
            p = self.final_stop_layer(p)  # (batch_size, 2)

            topics[:, t, :] = topic
            ps[:, t, :] = p

        return topics, ps


class WordLSTM(nn.Module):
    def __init__(self, word_input_dim, word_hidden_dim, vocab_size, num_layers=1):
        super(WordLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, word_input_dim)
        self.lstm = nn.LSTM(word_input_dim, word_hidden_dim, num_layers, batch_first=True, bidirectional=False)
        # self.fc1 = nn.Linear(word_input_dim, word_input_dim)
        # self.fc = nn.Linear(word_hidden_dim*2, vocab_size)
        self.fc = nn.Sequential(
            nn.Linear(word_hidden_dim, word_hidden_dim*4),
            nn.GELU(),
            nn.Linear(word_hidden_dim*4, vocab_size)
        )
        self.__init_weights()

    def __init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.fc.weight.data.uniform_(-0.1, 0.1)
        # self.fc.bias.data.fill_(0)
        # self.fc1.weight.data.uniform_(-0.1, 0.1)
        # self.fc.bias.data.fill_(0)

    def forward(self, topic, captions):
        embeddings = self.embedding(captions)  # (batch_size, max_sent_len, word_input_dim)

        outputs, _ = self.lstm(
            torch.cat((topic.unsqueeze(1), embeddings), 1))  # (batch_size, max_sent_len + 1, word_hidden_dim)

        outputs = self.fc(outputs)  # (batch_size, max_sent_len + 1, vocab_size)

        outputs = outputs[:, :-1, :]  # (batch_size, max_sent_len, vocab_size)

        return outputs

    def sample(self, topic, start_token, shape, args):
        pred_sent = torch.zeros(shape[0], shape[2]).to(args.device)
        # words_out = torch.zeros([shape[0], shape[1], self.vocab_size]).to(args.device)
        # p_state = None
        # p_out, state = self.lstm(topic.unsqueeze(1))
        # p_out = self.fc(p_out)
        # predicted = torch.max(p_out, -1)[1].squeeze()
        # p_state = state
        predicted = start_token.unsqueeze(1)
        embeddings = topic.unsqueeze(1)
        for i in range(1, shape[2]):
            predicted = self.embedding(predicted)
            embeddings = torch.cat((embeddings, predicted), 1)  # (b,i,d),((2,16,d),(2,16,d))
            hidden_states, _ = self.lstm(embeddings)
            output = self.fc(hidden_states[:, -1, :])  # (b,1,v)
            # words_out[:, i, :] = output  # (b,l,v)
            predicted = torch.max(output, 1)[1]
            pred_sent[:, i] = predicted
            predicted = predicted.unsqueeze(1)
        return self.fc(hidden_states), pred_sent


class Vito(ViT):
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def cosine_similarity(x, y, axis=-1):
    x = normalize(x)
    y = normalize(y)
    cos = 1 - torch.bmm(x, y.permute(0, 2, 1))
    return cos


def drop_tokens(embeddings, p_drop):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - p_drop)
    embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    return embeddings.long(), mask


class CoAttention(nn.Module):
    def __init__(self,
                 sem_size=256,
                 hidden_size=512,
                 visual_size=768,
                 vis_dim=49,
                 sem_dim=10,
                 momentum=0.1,
                 sent_dim=1024):
        super(CoAttention, self).__init__()
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=vis_dim, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)
        self.bn_cs = nn.BatchNorm1d(num_features=vis_dim, momentum=momentum)
        # self.cs = torch.cosine_similarity()

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=vis_dim, momentum=momentum)

        self.W_a = nn.Linear(in_features=sem_size, out_features=sem_size)
        self.bn_a = nn.BatchNorm1d(num_features=sem_dim, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=sem_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=sem_size, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=sem_size, out_features=sem_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=sem_dim, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + sem_size, out_features=sent_dim)
        self.bn_fc = nn.BatchNorm1d(num_features=sent_dim, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.version = 2
        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, vis_features, semantic_features, h_sent):
        batch_size = vis_features.shape[0]
        if self.version == 1:
            W_v = self.bn_v(self.W_v(vis_features))
            W_v_h = self.bn_v_h(self.W_v_h(h_sent))

            alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h.unsqueeze(1)))))
            v_att = torch.mul(alpha_v, vis_features).mean(1)

            W_a_h = self.bn_a_h(self.W_a_h(h_sent))
            W_a = self.bn_a(self.W_a(semantic_features))
            alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(W_a_h.unsqueeze(1) + W_a))))
            a_att = torch.mul(alpha_a, semantic_features).mean(1)

            ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

            return ctx, alpha_v, alpha_a

        elif self.version == 2:
            W_v = self.bn_v(self.W_v(vis_features))  # (b,49,d1)
            W_v_h = self.bn_v_h(self.W_v_h(h_sent))  # (b,d1)
            S_v = cosine_similarity(W_v, W_v_h.unsqueeze(1)).squeeze()  # (b,49)
            alpha_v = self.softmax(0.3 * normalize(torch.maximum(S_v, torch.zeros(S_v.shape).cuda())))  # (b,49)
            v_att = torch.bmm(alpha_v.unsqueeze(1), vis_features).squeeze()  # (b,d1)

            W_a = self.bn_a(self.W_a(semantic_features))
            W_a_h = self.bn_a_h(self.W_a_h(h_sent))
            S_a = cosine_similarity(W_a, W_a_h.unsqueeze(1)).squeeze()
            alpha_a = self.softmax(0.3 * normalize(torch.maximum(S_a, torch.zeros(S_a.shape).cuda())))
            a_att = torch.bmm(alpha_a.unsqueeze(1), semantic_features).squeeze()

            ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

            return ctx, alpha_v, alpha_a


class MLC(nn.Module):
    def __init__(self, tags, emb_dim, vis_dim, vocab_dim, device):
        super(MLC, self).__init__()
        self.tags = tags
        self.vis_dim = vis_dim
        self.emb_dim = emb_dim
        # self.linear = nn.Linear(vis_dim, emb_dim)
        self.subnetworks = {}
        self.cls = {}
        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        # self.mlp = MixerBlock(len(self.tags), 2, 128, 128).to(device)
        # self.mlp = nn.Sequential(
        #     *[MixerBlock(len(self.tags), self.vis_dim, 768, 768) for _ in range(2)]).to(device)
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.Linear(768, 768))
        self.init_weights()
        self.version = 1
        if self.version == 1:
            for tag in self.tags:
                self.subnetworks[tag] = nn.Sequential(
                    nn.Linear(self.vis_dim, self.vis_dim),
                ).to(device)
                self.cls[tag] = nn.Sequential(
                    nn.LayerNorm(self.vis_dim),
                    nn.Linear(self.vis_dim, 2),
                    nn.Sigmoid()
                ).to(device)
        else:
            for tag in self.tags:
                self.subnetworks[tag] = nn.Sequential(
                    nn.Linear(self.vis_dim, self.vis_dim),
                ).to(device)
            self.cls = nn.Sequential(nn.Linear(len(self.tags), len(self.tags)), nn.Sigmoid()).to(device)

    def tag_class(self, x, device):
        ret1 = torch.zeros([x.shape[0], len(self.tags), self.vis_dim], requires_grad=True).to(device)
        for i, tag1 in enumerate(self.subnetworks.keys()):
            ret1[:, i, :] = self.subnetworks[tag1](x)
        return ret1

    def classify(self, x, device):
        ret2 = torch.zeros([x.shape[0], len(self.tags), 2], requires_grad=True).to(device)
        for i, tag2 in enumerate(self.cls.keys()):
            ret2[:, i, :] = self.cls[tag2](x[:, i, :])
        return ret2

    def forward(self, x, device):
        # x = self.linear(x)
        x = self.tag_class(x.view(-1, self.vis_dim), device)
        x = self.mlp(x)
        if self.version == 1:
            x = self.classify(x, device)
            _, indices = torch.sort(x[:, :, 1], dim=1, descending=True)
            tags = torch.zeros([x.shape[0], x.shape[1]]).long().to(device)
            for i, b1 in enumerate(x):
                for j, b2 in enumerate(b1):
                    tags[i, j] = 1 if b2[0] < b2[1] else 0
            indices = self.embedding(indices[:, : 10])
            return x, indices, tags
        else:
            x = self.cls(x.mean(dim=2))
            _, indices = torch.sort(x[:, :], dim=1, descending=True)
            tags = torch.zeros([x.shape[0], x.shape[1]]).long().to(device)
            for i in range(x.shape[0]):
                for j in range(3):
                    if _[i, j] >= 0.7:
                        tags[i, indices[i, j]] = 1
            indices = self.embedding(indices[:, : 10])
            return x, indices, tags

    def init_weights(self):
        for tag in self.subnetworks.keys():
            for i in range(len(self.subnetworks[tag])):
                if self.subnetworks[tag][i]._get_name() == 'Linear':
                    self.subnetworks[tag][i].bias.data.fill_(0)
                    self.subnetworks[tag][i].weight.data.uniform_(-0.1, 0.1)
        for tag in self.cls.keys():
            for i in range(len(self.cls[tag])):
                if self.cls[tag][i]._get_name() == 'Linear':
                    self.cls[tag][i].bias.data.fill_(0)
                    self.cls[tag][i].weight.data.uniform_(-0.1, 0.1)

    # def for_bak(self, x, device):
    #     x = self.linear(x)
    #     x = self.linear(x)
    #     x = self.softmax(x)
    #     topk2 = torch.topk(x, 2)
    #     topk10 = torch.tokk(x, 10)
    #     indices = self.embedding(topk10.indices)
    #     tags = torch.zeros([x.shape[0], x.shape[1]]).long().to(device)
    #     for i in range(topk2.indices.shape[0]):
    #         for j in range(topk2.indices.shape[1]):
    #             tags[i, topk2.indices[i, j]] = 1
    #     return x, indices,


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)  # 先进行一次token-mixing MLP
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)  # 再进行一次channel-mixing MLP
        # x = self.dropout(x)
        return x


class Tag_fuse(nn.Module):
    def __init__(self, tags_l, vocab_size, args):
        super(Tag_fuse, self).__init__()
        self.vocab_size = vocab_size
        self.encoderCNN = EncoderCNN()

        self.mlc = MLC(tags_l, 256, self.encoderCNN.enc_dim, vocab_size, args.device)

        self.sentLSTM = SentenceLSTM(self.encoderCNN.enc_dim, args.sent_hidden_dim, args.att_dim,
                                     args.sent_input_dim,
                                     args.word_input_dim, args.int_stop_dim)

        self.wordLSTM = WordLSTM(args.word_input_dim, args.word_hidden_dim, vocab_size, args.num_layers)
        # self.params_cnn = list(self.encoderCNN.parameters())
        # self.params_mlc = list(self.mlc.parameters())
        # self.params_lstm = list(self.sentLSTM.parameters()) + list(self.wordLSTM.parameters())
        # self.optim_cnn = torch.optim.Adam(params=self.params_cnn, lr=args.learning_rate_cnn)  # , weight_decay=1e-4)
        # self.optim_mlc = torch.optim.Adam(params=self.params_mlc, lr=args.learning_rate_mlc)  # L2 regularization
        # self.optim_lstm = torch.optim.Adam(params=self.params_lstm, lr=args.learning_rate_lstm)
        self.word_softmax = nn.Softmax(dim=3)

    def forward(self, x, y, args, state):
        vis_enc_output = self.encoderCNN(x)
        sem_enc_output = vis_enc_output[:, :1, :]
        vis_enc_output = vis_enc_output[:, 1:, :]
        # sem_enc_output = vis_enc_output.mean(1).unsqueeze(1)
        tag_score, sem_enc_output, predict_tags = self.mlc(sem_enc_output, args.device)
        topics, ps = self.sentLSTM(vis_enc_output, sem_enc_output, y, args.device)
        word_outputs = torch.zeros((y.shape[0], y.shape[1], y.shape[2], self.vocab_size)).to(args.device)
        pred_words = torch.zeros((y.shape[0], y.shape[1], y.shape[2]))
        # masked_caption, _ = drop_tokens(y, 0.85)
        if state == 'train':
            for i in range(y.shape[1]):
                word_outputs[:, i] = self.wordLSTM(topics[:, i, :], y[:, i, :])

                _, words = torch.max(word_outputs[:, i], 2)

                pred_words[:, i, :] = words

            return tag_score, predict_tags, ps, word_outputs, pred_words
        elif state == 'val':
            # start_token = torch.ones((y.shape[0], y.shape[1], y.shape[2])) * 4
            # start_token[:, :, 0] = 1
            # start_token = start_token.long().to(args.device)
            # for i in range(y.shape[1]):
            #     word_outputs[i], pred_words[:, i, :] = self.wordLSTM.sample(topics[:, i, :], y[:, i, :], args)
            # return tag_score, predict_tags, ps, word_outputs, pred_words
            for i in range(y.shape[1]):
                word_outputs[:, i] = self.wordLSTM(topics[:, i, :], y[:, i, :])

                _, words = torch.max(word_outputs[:, i], 2)

                pred_words[:, i, :] = words

            return tag_score, predict_tags, ps, word_outputs, pred_words

        elif state == 'one_step_train':
            start_token = torch.ones((y.shape[0]), requires_grad=False)
            # pred[:, :, 0] = 1
            start_token = start_token.long().to(args.device)
            for i in range(y.shape[1]):
                word_outputs[:, i], pred_words[:, i] = self.wordLSTM.sample(topics[:, i, :], start_token, y.shape, args)
            return tag_score, predict_tags, ps, word_outputs, pred_words

        elif state == 'one_step_val':
            start_token = torch.ones((y.shape[0]), requires_grad=False)
            # pred[:, :, 0] = 1
            start_token = start_token.long().to(args.device)
            for i in range(y.shape[1]):
                word_outputs[:, i], pred_words[:, i] = self.wordLSTM.sample(topics[:, i, :], start_token, y.shape, args)
            return tag_score, predict_tags, ps, word_outputs, pred_words
        else:
            raise ValueError('invalid state')











