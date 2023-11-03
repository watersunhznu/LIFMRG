import jieba
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, re
import nltk
from collections import Counter
from build_vocab import Vocabulary, build_vocab
import json


def create_captions(filepath, dataset):

    ## the captions have the impression and findings concatenated to form one big caption
    ## i.e. caption = impression + " " + findings
    ## WARNING: in addition to the XXXX in the captions, there are <unk> tokens


    # clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{},0-9]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()

    captions = []
    '''
    with open(filepath, "r") as file:

        for line in file:
            line = line.replace("\n", "").split("\t")
            
            sentence_tokens = []
            
            for sentence in line[1].split("."):
                tokens = bioclean(sentence)
                if len(tokens) == 0:
                    continue
                caption = " ".join(tokens)
                sentence_tokens.append(caption)
            
            captions.append(sentence_tokens)
    '''
    file = json.load(open(filepath, 'r'))
    if dataset == 'bra':
        for item in file:
            sentence_tokens = []

            for sentence in item['paragraph'].split('。'):
                tokens = bioclean(sentence)
                if len(tokens) == 0:
                    continue
                caption = " ".join(tokens)
                sentence_tokens.append(caption)

            captions.append(sentence_tokens)
    else:
        for item in file:
            sentence_tokens = []

            for sentence in item['paragraph'].split('.'):
                tokens = bioclean(sentence)
                if len(tokens) == 0:
                    continue
                caption = " ".join(tokens)
                sentence_tokens.append(caption)

            captions.append(sentence_tokens)
    return captions

class iuxray(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab = None, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))
        
        tsv_file = os.path.join(self.root_dir, self.tsv_path)
        
        self.captions = create_captions(tsv_file, 'iuxray')
        if vocab is None:
            self.vocab = build_vocab(self.captions, 1)
        else:
            self.vocab = vocab
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.data_file = json.load(open(tsv_file, 'r'))
        self.transform = transform
        self.tags_l = ['normal', 'opacity', 'degenerative change', 'atelectases', 'atelectasis', 'scarring',
          'cardiomegaly', 'calcified granuloma', 'granuloma', 'pneumonia', 'pleural effusion', 'sternotomy',
          'pleural effusions', 'pulmonary emphysema', 'infiltrates', 'emphysemas', 'granulomatous disease',
          'nodule', 'pulmonary edema', 'diaphragm', 'emphysema', 'deformity', 'thoracic aorta',
          'osteophytes', 'hiatal hernia', 'thoracic vertebrae', 'fracture', 'tortuous aorta',
          'bilateral pleural effusion', 'rib fracture', 'aorta', 'edemas', 'calcinosis', 'scar', 'edema',
          'copd', 'pulmonary disease, chronic obstructive', 'pneumothorax', 'effusion', 'pleural thickening',
          'pulmonary atelectasis', 'congestion', 'eventration', 'rib fractures', 'hyperinflation',
          'arthritic changes', 'ribs', 'cabg', 'catheterization, central venous', 'infection', 'others']

        # self.tags_l = ['normal', 'opacity', 'degenerative change', 'atelectases', 'atelectasis', 'scarring',
        #                'cardiomegaly', 'calcified granuloma', 'granuloma', 'pneumonia',  'others']


    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        # print(idx)
        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['file_name'])
        image = Image.open(img_name)
        
        if self.transform is not None:
            image = self.transform(image)
        
        caption = self.captions[idx]

        # print("---")
        # print(caption)
        # print("----")

        sentences = []
        if len(caption):
            for i in range(len(caption)):
                tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())
                sentence = [self.vocab('<start>')]
                sentence.extend([self.vocab(token) for token in tokens])
                sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentences.append(sentence)

        # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]]) 
            
        max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])
        
        for i in range(len(sentences)):
            if len(sentences[i]) < max_sent_len:
                sentences[i] = sentences[i] + (max_sent_len - len(sentences[i])) * [self.vocab('<pad>')]
                
        target = torch.Tensor(sentences)
        tags = []
        tags_yn = []
        for i in self.data_file[idx]['tag']:
            if i in self.tags_l:
                tags.append(i)
            else:
                tags.append('others')
        for j in self.tags_l:
            if j in tags:
                tags_yn.append(1)
            else:
                tags_yn.append(0)

        return image, target, len(sentences), max_sent_len, tags_yn


class mimic(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab=None, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))

        tsv_file = os.path.join(self.root_dir, self.tsv_path)

        self.captions = create_captions(tsv_file, 'mimic')
        if vocab is None:
            self.vocab = build_vocab(self.captions, 10)
        else:
            self.vocab = vocab
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.data_file = json.load(open(tsv_file, 'r'))
        self.transform = transform
        self.tags_l = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                       'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

        # self.tags_l = ['normal', 'opacity', 'degenerative change', 'atelectases', 'atelectasis', 'scarring',
        #                'cardiomegaly', 'calcified granuloma', 'granuloma', 'pneumonia',  'others']

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        # print(idx)
        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['file_name'])
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions[idx]

        # print("---")
        # print(caption)
        # print("----")

        sentences = []
        if len(caption):
            for i in range(len(caption)):
                tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())
                sentence = [self.vocab('<start>')]
                sentence.extend([self.vocab(token) for token in tokens])
                sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentences.append(sentence)

        # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])

        max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])

        for i in range(len(sentences)):
            if len(sentences[i]) < max_sent_len:
                sentences[i] = sentences[i] + (max_sent_len - len(sentences[i])) * [self.vocab('<pad>')]

        target = torch.Tensor(sentences)
        tags = self.data_file[idx]['tag']
        tags_yn = []
        # for i in self.data_file[idx]['tag']:
        #     if i in self.tags_l:
        #         tags.append(i)
        #     else:
        #         pass
        #         # tags.append('others')
        for j in self.tags_l:
            if j in tags:
                tags_yn.append(1)
            else:
                tags_yn.append(0)

        return image, target, len(sentences), max_sent_len, tags_yn


class bra(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, vocab=None, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        # self.tags = json.load(open('/home/mzjs/bio_image_caption/SiVL19/iu_xray/iu_xray_auto_tags.json'))

        tsv_file = os.path.join(self.root_dir, self.tsv_path)

        self.captions = create_captions(tsv_file, 'bra')
        if vocab is None:
            self.vocab = build_vocab(self.captions, 1)
        else:
            self.vocab = vocab
        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.data_file = json.load(open(tsv_file, 'r'))
        self.transform = transform
        self.tags_l = ['淋巴结增多', '混合型', '片絮状影', '结节影', '腺体影', '腺体致密', '致密灶', '轻度增大', '钙化灶', '高密度影']
        jieba.add_word('高密度影')
        jieba.add_word('片絮状影')

        # self.tags_l = ['normal', 'opacity', 'degenerative change', 'atelectases', 'atelectasis', 'scarring',
        #                'cardiomegaly', 'calcified granuloma', 'granuloma', 'pneumonia',  'others']

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        # print(idx)
        img_name = os.path.join(self.root_dir, self.image_path, self.data_file[idx]['file_name'])
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions[idx]

        # print("---")
        # print(caption)
        # print("----")

        sentences = []
        if len(caption):
            for i in range(len(caption)):
                tokens = jieba.cut(caption[i])
                sentence = [self.vocab('<start>')]
                sentence.extend([self.vocab(token) for token in tokens])
                sentence.append(self.vocab('<end>'))
                # print([self.vocab.idx2word[k] for k in sentence])
                sentences.append(sentence)
        else:
            sentence = [self.vocab('<start>'), self.vocab('<end>')]
            sentences.append(sentence)

        # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])

        max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])

        for i in range(len(sentences)):
            if len(sentences[i]) < max_sent_len:
                sentences[i] = sentences[i] + (max_sent_len - len(sentences[i])) * [self.vocab('<pad>')]

        target = torch.Tensor(sentences)
        tags = []
        tags_yn = []
        for i in self.data_file[idx]['tag']:
            if i in self.tags_l:
                tags.append(i)
            else:
                tags.append('others')
        for j in self.tags_l:
            if j in tags:
                tags_yn.append(1)
            else:
                tags_yn.append(0)

        return image, target, len(sentences), max_sent_len, tags_yn


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, no_of_sent, max_sent_len).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption, no_of_sent, max_sena list indicating valid length for each caption.t_len).
            - image: torch tensor of shape (3, crop_size, crop_size).
            - caption: torch tensor of shape (no_of_sent, max_sent_len); variable length.
            - no_of_sent: number of sentences in the caption
            - max_sent_len: maximum length of a sentence in the caption

    Returns:
        images: torch tensor of shape (batch_size, 3, crop_size, crop_size).
        targets: torch tensor of shape (batch_size, max_no_of_sent, padded_max_sent_len).
        prob: torch tensor of shape (batch_size, max_no_of_sent)
    """
    # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: len(x[1]), reverse=True)
    
    images, captions, len_sentences, max_sent_len, tags_yn = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    tags = torch.LongTensor(tags_yn)
    targets = torch.zeros(len(captions), max(len_sentences), max(max_sent_len)).long()
    prob = torch.zeros(len(captions), max(len_sentences)).long()
    
    for i, cap in enumerate(captions):
        for j, sent in enumerate(cap):
            targets[i, j, :len(sent)] = sent[:] 
            prob[i, j] = 1
        # stop after the last sentence
        # prob[i, j] = 0
      
    return images, targets, prob, tags

def get_loader(root_dir, tsv_path, image_path, transform, batch_size, shuffle, num_workers, dataset, vocab = None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # dataset
    if dataset == 'iuxray':
        data = iuxray(root_dir = root_dir,
                 tsv_path = tsv_path,
                 image_path = image_path,
                 vocab = vocab,
                 transform = transform)
    elif dataset == 'mimic':
        data = mimic(root_dir = root_dir,
                 tsv_path = tsv_path,
                 image_path = image_path,
                 vocab = vocab,
                 transform = transform)
    elif dataset == 'bra':
        data = bra(root_dir=root_dir,
                     tsv_path=tsv_path,
                     image_path=image_path,
                     vocab=vocab,
                     transform=transform)


    
    # Data loader for dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, resize_length, resize_width).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset = data, 
                                              batch_size = batch_size,
                                              shuffle = shuffle,
                                              num_workers = num_workers,
                                              collate_fn = collate_fn)

    return data_loader, data.vocab

if __name__ == '__main__':
    root_dir = '/home/mzjs/data'
    tsv_path = 'data.json'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    image_path = 'images'
    train_loader, _ = get_loader(root_dir, tsv_path, image_path, transform, 1, True, 1, 'bra')
    for i, (images, captions, prob, tags) in enumerate(train_loader):
        print(i)
