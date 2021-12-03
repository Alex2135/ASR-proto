import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import string


class LangHandling():
    def __init__(self, tokens):
        self.token_to_index = {c: i for i, c in enumerate(tokens)}
        self.index_to_token = {i: c for i, c in enumerate(tokens)}
        self.num_classes = len(tokens)


class LangCharHandling(LangHandling):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.stop_signs = string.punctuation + "–—»«…“”’"

    def remove_stop_signs(self, sentence):
        for sign in self.stop_signs:
            sentence = sentence.replace(sign, "")
        return sentence.strip()

    def sentence_to_indeces(self, sentence):
        sent = self.remove_stop_signs(sentence)
        sent = sent.lower()
        sent = sent.split()
        result = []
        for word in sent:
            for c in word:
                char = self.token_to_index.get(c, self.token_to_index["<unk>"])
                result.append(char)
            result.append(self.token_to_index[" "])
        result = result[:-1]
        result = [self.token_to_index["<sos>"]] + result + [self.token_to_index["<eos>"]]
        return result

    def sentences_to_indeces(self, sentences):
        result = [torch.Tensor(self.sentence_to_indeces(sent)).long()
                   for sent in sentences]
        result = pad_sequence(result, batch_first=True, padding_value=self.token_to_index["<eos>"])
        #result = F.pad(result, (0, sent_max_len - result.shape[-1]), value=self.token_to_index['<eos>'])
        return result

    def sentence_to_one_hots(self, sent):
        """
        Convert "sent" to one hots matrix by characters of language with indeces.

        Example:
        sent = "Московитам дозволено створити свою державу а татарам чеченцям – ні Але це – расизм"
        obj = LangCharHandling(tokens)
        sent_to_idxs = obj.sentence_to_indeces(sent, char_to_index)

        :param sent:
        :return:
        """
        sent_to_idxs = self.sentence_to_indeces(sent)
        sent_to_idxs = torch.Tensor(sent_to_idxs).long()
        one_hots = F.one_hot(sent_to_idxs, num_classes=self.num_classes)
        return one_hots

    def sentences_to_one_hots(self, sents, sent_max_len):
        sents_one_hots = [self.sentence_to_one_hots(sent) for sent in sents]
        sents_one_hots = [F.pad(one_hot, (0, 0, 0, sent_max_len - one_hot.shape[0]), value=0)
                          for one_hot in sents_one_hots]
        result = torch.stack(sents_one_hots)
        b, l, d = result.shape
        result = result.view(b, 1, d, l)
        return result

    def one_hots_to_sentences(self, one_hots):
        b, _, h, w = one_hots.shape
        one_hots = one_hots.view(b, w, h)
        result = []
        idxs_batch = []
        for one_hot in one_hots:
            idxs_batch.append(self.onehot_matrix_to_idxs(one_hot))
        for idxs in idxs_batch:
            sent = ""
            for index in idxs:
                sent += self.index_to_token[int(index)]
            result.append(sent)
        return result

    def onehot_matrix_to_idxs(self, one_hots):
        result = []
        for i in range(one_hots.shape[0]):
            one_hot = one_hots[i, :]
            number = np.argmax(one_hot)
            result.append(number)
        return result


def CER(indeces1, indeces2):
    assert len(indeces1) == len(indeces2)

    n = len(indeces1)
    char_sum = torch.sum(indeces1 != indeces2)
    return char_sum / n


extra_tokens = ["<sos>", "<eos>", "<unk>", "<blank>", " "]
tokens = extra_tokens + ['а', 'б', 'в', 'г', 'д',
        'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м',
        'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
        'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я',
        'є', 'і', 'ї', 'ґ']

ukr_lang_chars_handle = LangCharHandling(tokens)
