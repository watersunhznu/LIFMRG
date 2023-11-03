# coding=utf-8
import os
import torch
import argparse
from models import EncoderCNN, SentenceLSTM, WordLSTM, MLC
from dataloader import get_loader
from score import evalscores
from torchvision import transforms
from torch import nn
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import pandas
import visdom
import metrics
from focal_loss import focal_loss


def script(args):
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    tags_l = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                       'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    train_loader, vocab = get_loader(args.root_dir,
                                     args.train_tsv_path,
                                     args.image_path,
                                     transform,
                                     args.batch_size,
                                     args.shuffle,
                                     args.num_workers,
                                     args.dataset)
    for tag in tags_l:
        vocab.add_word(tag)
    vocab_size = len(vocab)
    print("vocab_size: ", vocab_size)

    val_loader, _ = get_loader(args.root_dir,
                               args.val_tsv_path,
                               args.image_path,
                               transform,
                               args.batch_size,
                               args.shuffle,
                               args.num_workers,
                               args.dataset,
                               vocab)

    title = args.title
    loss_t = visdom.Visdom(env=title)
    scores_t = visdom.Visdom(env=title)

    loss_t.line([[0., 0.]], [0.], win='loss', opts=dict(title='loss', legend=['train', 'val']))
    scores_t.line([[0., 0., 0., 0., 0., 0., 0.]], [0.], win='scores',
                  opts=dict(title='scores', legend=['blue1', 'blue2', 'blue3', 'blue4', 'meteor', 'rouge', 'cider']))

    encoderCNN = EncoderCNN().to(args.device)

    mlc = MLC(tags_l, 111, encoderCNN.enc_dim, vocab_size, args.device).to(args.device)

    sentLSTM = SentenceLSTM(encoderCNN.enc_dim, args.sent_hidden_dim, args.att_dim, args.sent_input_dim,
                            args.word_input_dim, args.int_stop_dim).to(args.device)

    wordLSTM = WordLSTM(args.word_input_dim, args.word_hidden_dim, vocab_size, args.num_layers).to(args.device)

    # criterion_tags = nn.CrossEntropyLoss().to(args.device)
    criterion_stop = nn.CrossEntropyLoss().to(args.device)
    criterion_words = nn.CrossEntropyLoss().to(args.device)
    criterion_fl = focal_loss(alpha=args.fl_alpha, gamma=args.fl_gamma).to(args.device)
    # criterion_mse = nn.MSELoss().to(args.device)

    params_cnn = list(encoderCNN.parameters())
    params_mlc = list(mlc.parameters())
    params_lstm = list(sentLSTM.parameters()) + list(wordLSTM.parameters())

    optim_cnn = torch.optim.Adam(params=params_cnn, lr=args.learning_rate_cnn, weight_decay=1e-4)
    optim_mlc = torch.optim.Adam(params=params_mlc, lr=args.learning_rate_mlc)  # L2 regularization
    optim_lstm = torch.optim.Adam(params=params_lstm, lr=args.learning_rate_lstm)

    total_step = len(train_loader)

    evaluate(args, val_loader, encoderCNN, mlc, sentLSTM, wordLSTM, vocab, 0, '', tags_l)

    for epoch in range(args.num_epochs):
        encoderCNN.train()
        mlc.train()
        sentLSTM.train()
        wordLSTM.train()
        met_t = np.zeros([len(tags_l), 4])

        for i, (images, captions, prob, tags) in enumerate(train_loader):
            optim_cnn.zero_grad()
            optim_mlc.zero_grad()
            optim_lstm.zero_grad()
            # l1_regularization = torch.tensor([0], dtype=torch.float32).to(args.device)

            batch_size = images.shape[0]
            images = images.to(args.device)
            captions = captions.to(args.device)
            prob = prob.to(args.device)
            tags = tags.to(args.device)

            vis_enc_output = encoderCNN(images)
            sem_enc_output = vis_enc_output[:, :1, :]
            vis_enc_output = vis_enc_output[:, 1:, :]
            # sem_enc_output = vis_enc_output.mean(1).unsqueeze(1)
            tag_score, sem_enc_output, _1 = mlc(sem_enc_output, args.device)
            topics, ps = sentLSTM(vis_enc_output, sem_enc_output, captions, args.device)
            met_t = metrics.cala(tags, _1, met_t)

            # for parm in mlc.parameters():
            #     l1_regularization += torch.sum(abs(parm))  # calculation L1 regularization

            # tags_loss = criterion_tags(tag_score.view(-1, 2), tags.view(-1))
            tags_loss = criterion_fl(tag_score.view(-1, 2), tags.view(-1))
            # tags_loss = criterion_mse(tag_score.view(-1), tags.view(-1).float())
            loss_sent = criterion_stop(ps.view(-1, 2), prob.view(-1))

            loss_word = torch.tensor([0.0]).to(args.device)
            for j in range(captions.shape[1]):
                word_outputs = wordLSTM(topics[:, j, :], captions[:, j, :])

                loss_word += criterion_words(word_outputs.contiguous().view(-1, vocab_size),
                                             captions[:, j, :].contiguous().view(-1))

            # loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word + tags_loss * args.lambda_tag \
            #        + args.lambda_l1 * l1_regularization
            loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word + tags_loss * args.lambda_tag
            # loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word

            loss.backward()
            optim_cnn.step()
            optim_mlc.step()
            optim_lstm.step()

            # Print log info
            if i % args.log_step == 0:
                print('train： Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} >{:.2f}< >{:.2f}< >{:.5f}<'
                    .format(
                    epoch, args.num_epochs, i, total_step, loss.item(), loss_sent.item() * args.lambda_sent,
                                                                        loss_word.item() * args.lambda_word,
                                                                        tags_loss.item() * args.lambda_tag))
        ## Save the model checkpoints
        # if (i+1) % args.save_step == 0:
        #     torch.save(decoder.state_dict(), os.path.join(
        #         args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        #     torch.save(encoder.state_dict(), os.path.join(
        #         args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

        scores = evaluate(args, val_loader, encoderCNN, mlc, sentLSTM, wordLSTM, vocab, epoch, title, tags_l)
        loss_t.line([[loss.cpu().item(), scores[0].cpu().item()]], [epoch], win='loss', update='append')
        scores_t.line([scores[1:]], [epoch], win='scores', update='append')

        print('train metrics', np.sum(met_t, axis=0), 'Acc Precision Recall F1---------------->>', metrics.acc_f1(met_t))
    for i in range(len(tags_l)):
        print(' '*8, tags_l[i], ' '*(40-len(tags_l[i])), metrics.acc_f1(met_t[i]), met_t[i])



def evaluate(args, val_loader, encoderCNN, mlc, sentLSTM, wordLSTM, vocab, epoch, title, tags_l):
    encoderCNN.eval()
    mlc.eval()
    sentLSTM.eval()
    wordLSTM.eval()
    met_e = np.zeros([51, 4])
    vocab_size = len(vocab)
    total_step = len(val_loader)

    # criterion_tags_val = nn.CrossEntropyLoss().to(args.device)
    criterion_stop_val = nn.CrossEntropyLoss().to(args.device)
    criterion_words_val = nn.CrossEntropyLoss().to(args.device)
    criterion_fl_val = focal_loss(alpha=args.fl_alpha, gamma=args.fl_gamma).to(args.device)
    # criterion_mse = nn.MSELoss().to(args.device)

    references = list()
    hypotheses = list()

    for i, (images, captions, prob, tags) in enumerate(val_loader):
        images = images.to(args.device)
        captions = captions.to(args.device)
        prob = prob.to(args.device)
        tags = tags.to(args.device)
        # l1_regularization_eval = torch.tensor([0], dtype=torch.float32).to(args.device)

        # test_caption = list()
        # print(prob[0])

        # for x in range(captions.shape[1]):
        # 	print(prob[0][x])
        # 	print("diff")
        # 	print(prob[0, x])
        # 	if prob[0, x] == 1:
        # 		test_words = captions[0, x, :].tolist()
        # 		test_caption.extend([w for w in test_words if w not in {vocab.word2idx['<pad>']}])

        # print([vocab.idx2word[k] for k in test_caption])

        vis_enc_out = encoderCNN(images)  # (batch_size, , vis_dim)

        sem_enc_out = vis_enc_out[:, :1, :]
        vis_enc_out = vis_enc_out[:, 1:, :]
        # sem_enc_out = vis_enc_out.mean(1).unsqueeze(1)

        tag_score, sem_enc_out, _2 = mlc(sem_enc_out, args.device)
        topics, ps = sentLSTM(vis_enc_out, sem_enc_out, captions, args.device)
        # topics, ps = sentLSTM(vis_enc_out, sem_enc_out, captions, args.device)
        met_e = metrics.cala(tags, _2, met_e)

        # for parm in mlc.parameters():
        #     l1_regularization_eval += torch.sum(abs(parm))  # calculation L1 regularization

        # tags_loss = criterion_tags_val(tag_score.view(-1, 2), tags.view(-1))
        tags_loss = criterion_fl_val(tag_score.view(-1, 2), tags.view(-1))
        # tags_loss = criterion_mse(tag_score.view(-1), tags.view(-1).float())
        loss_sent = criterion_stop_val(ps.view(-1, 2), prob.view(-1))

        loss_word = torch.tensor([0.0]).to(args.device)

        pred_words = torch.zeros((captions.shape[0], captions.shape[1], captions.shape[2]))

        for j in range(captions.shape[1]):
            word_outputs = wordLSTM(topics[:, j, :], captions[:, j, :])

            loss_word += criterion_words_val(word_outputs.contiguous().view(-1, vocab_size),
                                             captions[:, j, :].contiguous().view(-1))

            _, words = torch.max(word_outputs, 2)

            pred_words[:, j, :] = words

        # loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word + tags_loss * args.lambda_tag \
        #        + args.lambda_l1 * l1_regularization_eval
        loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word + tags_loss * args.lambda_tag
        # loss = args.lambda_sent * loss_sent + args.lambda_word * loss_word

        for j in range(captions.shape[0]):
            pred_caption = []
            target_caption = []
            for k in range(captions.shape[1]):
                if ps[j, k, 1] > 0.5:
                    words_x = pred_words[j, k, :].tolist()

                    # pred_caption.extend([w for w in words_x if w not in {vocab.word2idx['<pad>'], vocab.word2idx[
                    # '<start>']}])

                    pred_caption.append(" ".join([vocab.idx2word[w] for w in words_x if
                                                  w not in {vocab.word2idx['<pad>'], vocab.word2idx['<start>'],
                                                            vocab.word2idx['<end>']}]) + ".")

                if prob[j, k] == 1:
                    words_y = captions[j, k, :].tolist()
                    # target_caption.extend([w for w in words_y if w not in {vocab.word2idx['<pad>'], vocab.word2idx[
                    # '<start>']}])
                    target_caption.append(" ".join([vocab.idx2word[w] for w in words_y if
                                                    w not in {vocab.word2idx['<pad>'], vocab.word2idx['<start>'],
                                                              vocab.word2idx['<end>']}]) + ".")

            hypotheses.append(pred_caption)
            references.append(target_caption)
        if i % args.log_step == 0:
            print('val: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} >{:.2f}< >{:.2f}< >{:.5f}<'
                .format(
                epoch, args.num_epochs, i, total_step, loss.item(), loss_sent.item() * args.lambda_sent,
                                                                    loss_word.item() * args.lambda_word,
                                                                    tags_loss.item() * args.lambda_tag
            ))
    assert len(references) == len(hypotheses)
    print('eval metrics', np.sum(met_e, axis=0), 'Acc Precision Recall F1-->>', metrics.acc_f1(met_e))
    for i in range(len(tags_l)):
        print(' '*8, tags_l[i], ' '*(40-len(tags_l[i])), metrics.acc_f1(met_e[i]), met_e[i])
    # print(references[0])
    # print(hypotheses[0])

    if epoch == args.num_epochs - 1:
        datas = pandas.DataFrame(columns=['groud_truth', 'generation'])
        for i in range(len(references)):
            datas.loc[i] = [references[i], hypotheses[i]]
        datas.to_csv(
            '/home/mzjs/bio_image_caption/On-the-Automatic-Generation-of-Medical-Imaging-Reports/result/' + title + '.csv')
        print('Save to file: ', title, '.csv')

    blue1, blue2, blue3, blue4, meteor, rouge, cider = evalscores(hypotheses, references)
    return loss, blue1, blue2, blue3, blue4, meteor, rouge, cider


# print([vocab.idx2word[references[0][0][k]] for k in range(len(references[0][0]))])
# print([vocab.idx2word[hypotheses[0][k]] for k in range(len(hypotheses[0]))])
# bleu4 = corpus_bleu(references, hypotheses)
# bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
# bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
# bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))


# print(
#     '\n * BLEU-1 - {bleu1}, BLEU-2 - {bleu2}, BLEU-3 - {bleu3}, BLEU-4 - {bleu4}\n'.format(
#         bleu1=bleu1,
#         bleu2=bleu2,
#         bleu3=bleu3,
#         bleu4=bleu4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='../SiVL19/iu_xray', help='path for root dir')
    parser.add_argument('--dataset', type=str, default='iuxray', help='path for root dir')
    parser.add_argument('--train_tsv_path', type=str, default='train1.json', help='path of the training tsv file')
    parser.add_argument('--val_tsv_path', type=str, default='val1.json', help='path of the validating tsv file')
    parser.add_argument('--image_path', type=str, default='iu_xray_images', help='path of the images file')
    parser.add_argument('--img_size', type=int, default=224, help='size to which image is to be resized')
    parser.add_argument('--crop_size', type=int, default=224, help='size to which the image is to be cropped')
    parser.add_argument('--device_number', type=str, default='1', help='which GPU to run experiment on')
    parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--title', type=str, default='default', help='title')

    parser.add_argument('--int_stop_dim', type=int, default=64,
                        help='intermediate state dimension of stop vector network')
    parser.add_argument('--sent_hidden_dim', type=int, default=512, help='hidden state dimension of sentence LSTM')
    parser.add_argument('--sent_input_dim', type=int, default=1024, help='dimension of input to sentence LSTM')
    parser.add_argument('--word_hidden_dim', type=int, default=512, help='hidden state dimension of word LSTM')
    parser.add_argument('--word_input_dim', type=int, default=512, help='dimension of input to word LSTM')
    parser.add_argument('--att_dim', type=int, default=64,
                        help='dimension of intermediate state in co-attention network')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in word LSTM')

    parser.add_argument('--lambda_sent', type=int, default=1,
                        help='weight for cross-entropy loss of stop vectors from sentence LSTM')
    parser.add_argument('--lambda_word', type=int, default=1,
                        help='weight for cross-entropy loss of words predicted from word LSTM with target words')
    parser.add_argument('--lambda_tag', type=int, default=10,
                        help='weight for cross-entropy loss of tag')
    parser.add_argument('--lambda_l1', type=int, default=1e-6,
                        help='weight for l1 regularization')

    parser.add_argument('--fl_alpha', type=float, default=0.15, help='')
    parser.add_argument('--fl_gamma', type=int, default=2, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='size of the batch')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the instances in dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train the model')
    parser.add_argument('--learning_rate_cnn', type=int, default=1e-5, help='learning rate for CNN Encoder')
    parser.add_argument('--learning_rate_mlc', type=int, default=1e-5, help='learning rate for MLC Encoder')
    parser.add_argument('--learning_rate_lstm', type=int, default=1e-5, help='learning rate for LSTM Decoder')

    parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    args = parser.parse_args()
    # torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    print('cuda version', torch.version.cuda)

    script(args)
