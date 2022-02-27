import os
from loguru import logger
import torch

class CnerProcessor():
    """Processor for the chinese ner data set."""
    def __init__(self, train_path, dev_path, test_path, tokenizer, max_len, segment_a_id=0):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.segment_a_id = segment_a_id
        self.label2id = {label: i for i, label in enumerate(self.get_labels())}  # 每种标签对应的id
        self.id2label = {i: label for i, label in enumerate(self.get_labels())}  # 每种标签对应的id

    def get_train_examples(self):
        examples = self.create_examples(self.read_text(self.train_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of train data:{}'.format(len(features)))
        return features

    def get_dev_examples(self):
        examples = self.create_examples(self.read_text(self.dev_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of dev data:{}'.format(len(features)))
        return features

    def get_test_examples(self):
        examples = self.create_examples(self.read_text(self.test_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of test data:{}'.format(len(features)))
        return features

    def convert_examples_to_inputs(self, examples):
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        o_label_id = self.label2id['O']
        features = []
        for words, labels in examples:
            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(words) + [sep_token_id]
            label_ids = [o_label_id] + [self.label2id[x] for x in labels] + [o_label_id]
            if len(input_ids) > self.max_len:
                input_ids = input_ids[: self.max_len]
                label_ids = label_ids[: self.max_len]
            input_mask = [1] * len(input_ids)
            token_type_ids = [self.segment_a_id] * len(input_ids)
            assert len(input_ids) == len(label_ids)

            # 对输入进行padding
            padding_length = self.max_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length
            input_mask += [pad_token_id] * padding_length
            token_type_ids += [pad_token_id] * padding_length
            label_ids += [pad_token_id] * padding_length
            text = ''.join(words)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            token_type_ids = torch.LongTensor(token_type_ids)
            feature = {'text': text, 'input_ids': input_ids, 'label_ids': label_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
            features.append(feature)
        return features

    def read_text(self, file):
        """
        读取文件，将每条记录读取为words:['我','在','天','津']，labels:['O','O','B-ORG','I-ORG']
        :param file:
        :return:
        """
        lines = []
        with open(file,'r') as f:
            words = []
            labels = []
            for line in f:
                if line == "" or line == "\n":
                    # 读取完一条记录
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    def get_labels(self):
        """See base class."""
        # return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
        #         'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
        #         'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]
        return ['B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'S-NAME', 'S-ORG', 'S-RACE', 'O']

    def create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            words = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append((words, labels))
        return examples