from torch.utils.data import RandomSampler,SequentialSampler
from transformers import BertConfig,BertTokenizer,BertForSequenceClassification,BertModel

from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm
from pipeline import Trainer,Tester
from dataset import DataSet
from model import PointwiseMatching
import torch
import sys
from torchmetrics import F1,Recall,Precision,Accuracy
from config import Config
from fastNLP import logger
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter

debug = True
colab = False
if colab:
    sys.argv =['name',
            '--model_name_or_path', '../pretrained_model/chinese_roberta_wwm_ext_pytorch',
            '--data_dir', '../data/tianchi',
            #'--do_test',
            '--do_train',
            '--save_logs',
            '--n_epochs', '3',
            '--num_classification', '3',
            '--max_seq_length', '80',
            '--train_batch_size', '16',
            '--eval_batch_size', '16',
            '--task', '0', '1',
            '--early_stop', '6',
            '--update_every', '2',
            '--validate_every', '1000',
            '--print_every', '4',
            '--warmup_steps', '0',
            '--warmup_proportion', '0.1',
            '--learning_rate', '2e-5',
            '--adam_epsilon', '1e-6',
            '--weight_decay', '1e-4']

    parser = Config.get_parser()
    args = parser.parse_args()
else:
    args = Config.get_default_cofig()

if debug:
    args.save_runs = True
    args.save_logs = True
    args.validate_every = 50

cache_dir = os.path.join(args.data_dir,'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

output_dir = os.path.join(args.data_dir,'model')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.save_logs:
    # logs_dir
    logs_dir = os.path.join(output_dir,'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger.add_file(os.path.join(logs_dir,'{}'.format(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))),level='info')

if args.save_runs:
    # runs_dir
    runs_dir = os.path.join(output_dir,'runs')
    writer = SummaryWriter(log_dir = os.path.join(runs_dir,'{}'.format(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))))
else:
    writer = None

# prepare model
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_classification)
# model = PointwiseMatching.from_pretrained(args.model_name_or_path,config = config)
model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

# prepare train_dataloader
dt = DataSet(tokenizer=tokenizer)
train_name='train.txt'
dev_name = 'dev.txt'
train_cache_name = os.path.join(cache_dir,'train_batchsize{}'.format(args.train_batch_size))
train_dataloader = dt.prepare_dataloader(file_path=os.path.join(args.data_dir, train_name),
                                            batch_size=args.train_batch_size,
                                            max_seq_length=args.max_seq_length,
                                            sampler=RandomSampler,
                                            _cache_fp=train_cache_name, _refresh=False)
# need to revised for specific model
# if args.save_runs:
#     writer.add_graph(model.to('cpu'),[i[0].unsqueeze(0) if i.shape else i for i in (*next(iter(train_dataloader))[:3],torch.tensor(True))])

# prepare scheduler
if args.warmup_proportion>0:
    n_steps = len(train_dataloader)*args.n_epochs/args.update_every
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*n_steps),
                                num_training_steps=n_steps)
else:
    scheduler = None

# prepare dev_dataloader
dev_cache_name = os.path.join(cache_dir,'dev_batchsize{}'.format(args.eval_batch_size))
dev_dataloader = dt.prepare_dataloader(file_path=os.path.join(args.data_dir, dev_name),
                                        batch_size=args.eval_batch_size,
                                        max_seq_length=args.max_seq_length,
                                        sampler=SequentialSampler,
                                       _cache_fp=dev_cache_name, _refresh=False)

# prepare metrics
f1 = F1(average='macro',num_classes=args.num_classification)
recall = Recall(average='macro',num_classes=args.num_classification)
precision = Precision(average='macro',num_classes=args.num_classification)
acc = Accuracy(num_classes=args.num_classification)
metrics = {'f1':f1,'recall':recall,'precision':precision,'acc':acc}

def generate_submit(predictor,read_filename='Xeon3NLP_round1_test_20210524.txt',write_filename='submit_addr_match_runid.txt'):
    # prepare test_dataloder
    #if os.path.isfile(os.path.join(args.data_dir,write_filename)):
    #    raise FileExistsError('write_file has existed')

    fo = open(os.path.join(args.data_dir,write_filename), "w",encoding='utf8')
    import json
    f = open(os.path.join(args.data_dir,read_filename), 'r',encoding='utf8')
    for line in tqdm(f.readlines(),total=len(f.readlines()),desc='generate submit file'):
        t = json.loads(line)
        for j in range(len(t['candidate'])):
            l = dict()
            l['text_a'] = t['query']
            l['text_b'] = t['candidate'][j]['text']
            l['label'] = 0
            dataloader = dt.prepare_dataloader_from_iterator([l],
                                                             args.eval_batch_size,
                                                             args.max_seq_length,
                                                             sampler=SequentialSampler)
            result = predictor.predict(dataloader)
            idx = result['infer_labels']
            if idx[0] == 0:
                t['candidate'][j]['label'] = '不匹配'
            elif idx[0] == 1:
                t['candidate'][j]['label'] = '部分匹配'
            else:
                t['candidate'][j]['label'] = '完全匹配'
        fo.write(json.dumps(t, ensure_ascii=False))
        fo.write('\n')
    f.close()
    fo.close()

args.do_train = True
dt = DataSet(tokenizer=tokenizer,verbose=0,use_tqdm=False)
if args.do_train:
    load_model = False
    if load_model:
        model_path = os.path.join(output_dir, 'best_BertForSequenceClassification_2021-06-18-17-09-32_')
        if not os.path.exists(model_path):
            raise FileNotFoundError("folder `{}` does not exist. Please make sure model are there.".format(model_path))
        states = torch.load(model_path).state_dict()
        model.load_state_dict(states)

    # trainer
    trainer = Trainer(train_dataloader, model, optimizer, scheduler=scheduler,
                update_every=args.update_every,n_epochs=args.n_epochs,
                 print_every=args.print_every,early_stop=args.early_stop,metrics=metrics,
                 dev_dataloader=dev_dataloader, validate_every=args.validate_every,
                save_path=output_dir,customize_model_name = None,seed=args.seed,debug=debug,
                writer=writer)

    trainer.train()
    # dev
    predictor = Tester(model)
    # result = predictor.predict(dev_dataloader)
    generate_submit(predictor,'Xeon3NLP_round1_train_20210524.txt','submit_dev.txt')
    generate_submit(predictor, read_filename='Xeon3NLP_round1_test_20210524.txt',
                    write_filename='submit_addr_match_runid.txt')

#args.do_test = True
if args.do_test:
    # prepare predictor
    model_path = os.path.join(output_dir, 'best_BertForSequenceClassification_2021-06-21-12-08-52_')
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    predictor = Tester(model)
    generate_submit(predictor,read_filename='Xeon3NLP_round1_test_20210524.txt',write_filename='submit_addr_match_runid.txt')


    # if args.predict_text:
    #     def prepare_data(text):
    #         return [{'_id':'uts1299034-124', 'text_a':text, 'label':0}]
    #
    #     dataloader = dt.prepare_dataloader_from_iterator(prepare_data(args.predict_text),
    #                                                      args.eval_batch_size,
    #                                                      args.max_seq_length,
    #                                                      sampler=SequentialSampler)
    # if args.predict_filename:
    #     dataloader = dt.prepare_dataloader(os.path.join(args.data_dir, args.predict_filename),
    #                                         args.eval_batch_size,
    #                                         args.max_seq_length,
    #                                         sampler=SequentialSampler)
    #
    # result = predictor.predict(dataloader)
    # logits = result['infer_logits']
    #
    # df=pd.read_csv(os.path.join(args.data_dir, args.predict_filename))
    # for i in range(args.num_classification):
    #     df['label_{}'.format(str(i))]=logits[:,i]
    # df[['_id']+['label_{}'.format(str(i)) for i in range(args.num_classification)]].to_csv(
    #     os.path.join(args.output_dir, "sub.csv"),index=False)
    









