import pandas as pd
import torch
from torch.utils.data import (DataLoader,TensorDataset)
import logging
from tqdm import tqdm
import os
from config import Config
from fastNLP import cache_results
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, **kwargs):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = kwargs.get('_id',)
        self.text_a = kwargs.get('text_a')
        self.text_b = kwargs.get('text_b','')
        self.label = kwargs.get('label',None)


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = {
                'input_ids': choices_features[1],
                'attention_mask': choices_features[2],
                'token_type_ids': choices_features[3]
            }
        self.label = label


class DataSet(object):
    def __init__(self, tokenizer, verbose=1,use_tqdm=True):
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.use_tqdm = use_tqdm


    def load_dataset(self,file_path):
        dataset_tmp = []
        import json
        f=open(file_path,'r',encoding='utf8')
        for line in f:
            t = json.loads(line)
            dataset_tmp.append(t)
        f.close()
        return dataset_tmp
    
    def load_iterator(self,file_path):
        dataset_tmp = self.load_dataset(file_path)
        for line in dataset_tmp:
            yield {'text_a': line['query'], 'text_b': line['title'], 'label':line['label']}
            
    def load_iterator1(self,file_path):
        df = pd.read_csv(file_path)
        #df=df.sample(400)
        if 'sentiment' not in df.columns:
            df['sentiment'] = 0
        lst = df[['_id', 'content', 'sentiment']].rename(columns={'content':'text_a','sentiment':'label'}).to_dict('records')
        for line in lst:
            yield line
    
    def _convert_iterator_to_example(self,iterator):
        examples = []
        for val in iterator:
            examples.append(InputExample(**val))
        return examples
    def convert_examples_to_features(self, examples, max_seq_length):
        '''Loads a data file into a list of `InputBatch`s.
        Args:
            examples      : [List] 输入样本，包括guid,text_a,text_b, label
            max_seq_length: [int] 文本最大长度
            tokenizer     : [Method] 分词方法
        Returns:
            features:
                input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
                attention_mask : [ListOfInt] 真实字符对应1，补全字符对应0
                token_type_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
        '''
        features = []
        if self.use_tqdm:
            converting_bars = tqdm(enumerate(examples), total=len(examples),
                                   desc='converting_examples_to_features')
        else:
            converting_bars = enumerate(examples)

        for example_index, example in converting_bars:

            text_a = self.tokenizer.tokenize(example.text_a)
            # self.tokenizer.encode_plus(example.text_a)
            text_b = self.tokenizer.tokenize(example.text_b)

            if len(text_b) == 0:
                end_token = []
            else:
                end_token = ["[SEP]"]

            self._truncate_seq_pair(text_a, text_b, max_seq_length - 2 - len(end_token))
            tokens = ["[CLS]"] + text_a + ["[SEP]"] + text_b + end_token
            token_type_ids = [0] * (len(text_a) + 2) + [1] * (len(text_b) + len(end_token))
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            attention_mask += ([0] * padding_length)
            token_type_ids += ([0] * padding_length)

            label = example.label
            if example_index < 1 and self.verbose == 1:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

            features.append(
                InputFeatures(
                    example_id = example.guid,
                    choices_features = (tokens, input_ids, attention_mask, token_type_ids),
                    label = label
                )
            )
        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _select_field(self, features, field):
        return [feature.choices_features[field]
            for feature in features]


    def prepare_dataloader_from_iterator(self, iterator, batch_size, max_seq_length, sampler=None):

        examples = self._convert_iterator_to_example(iterator)
        features = self.convert_examples_to_features(examples, max_seq_length)

        all_input_ids = torch.tensor(self._select_field(features, 'input_ids'), dtype=torch.long)
        all_attention_mask = torch.tensor(self._select_field(features, 'attention_mask'), dtype=torch.long)
        all_token_type_ids = torch.tensor(self._select_field(features, 'token_type_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)

        sampler_func = sampler(dataset) if sampler is not None else None
        dataloader = DataLoader(dataset, sampler=sampler_func, batch_size=batch_size)
        return dataloader

    @cache_results(_cache_fp='', _refresh=False)
    def prepare_dataloader(self, file_path, batch_size,max_seq_length,sampler=None):
        iterator = self.load_iterator(file_path)

        return self.prepare_dataloader_from_iterator(iterator,batch_size,max_seq_length,sampler)


if __name__ == '__main__':
    config = Config()
    args = config.get_default_cofig()
    from transformers import BertTokenizer
    from torch.utils.data import RandomSampler

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    dt=SentimentData(tokenizer = tokenizer)
    dt.prepare_dataloader(file_path=os.path.join(args.data_dir, 'train.csv'),sampler=RandomSampler,
                          batch_size = args.per_gpu_train_batch_size,max_seq_length=args.max_seq_length)