import random
import numpy as np
import torch
import os

from tqdm import tqdm
import torch.nn as nn
import time
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import classification_report,f1_score, accuracy_score,precision_score,recall_score
import torchmetrics
from fastNLP import logger
from datetime import datetime,timedelta


class Trainer(object):

    def __init__(self,train_dataloader, model, optimizer, scheduler,
                update_every=1,n_epochs=10, print_every=5,early_stop=5,
                 dev_dataloader=None, validate_every=-1, save_path=None,metrics=None,
                 customize_model_name = None,seed = 125,debug=False,save_last=True,writer=None):

        self.device = model.device
        self.model = model
        self.save_last = save_last
        self.train_data = train_dataloader
        self.dev_data = DataLoader(TensorDataset(*next(iter(dev_dataloader))),batch_size=dev_dataloader.batch_size)  if debug and dev_dataloader is not None else dev_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._set_seed(seed)

        self.n_epochs = int(n_epochs)
        self.print_every = int(print_every)
        self.validate_every = int(validate_every) if validate_every != 0 else -1
        self.step = 0
        self.update_every = int(update_every)
        self._update = self._update_with_warmup if self.scheduler else self._update_without_warmup
        self.n_steps = len(self.train_data) * self.n_epochs

        self.flag = 0  # count(eval_data performance is better self.best_score)
        self.early_stop = early_stop if early_stop else self.n_steps
        self.best_dev_epoch = 0
        self.best_dev_step = 0
        self.best_dev_perf = 0
        if metrics and isinstance(metrics,dict):
            self.metrics = metrics
            self.indicator = list(self.metrics.keys())[0]
        else:
            self.metrics = None

        self.writer = writer
        self.save_path = save_path if not debug else None
        self.customize = customize_model_name if customize_model_name else ''
        if self.dev_data is not None and self.metrics is not None:
            self.tester = Tester(self.model,self.metrics)

    def _grad_backward(self, loss):
        """Compute gradient with link rules.

        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        if (self.step-1) % self.update_every == 0:
            self.model.zero_grad()
        loss.backward()


    def train(self, load_best_model=True):
        """
        使用该函数使Trainer开始训练。

        :param bool load_best_model: 该参数只有在初始化提供了dev_data的情况下有效，如果True, trainer将在返回之前重新加载dev表现
                最好的模型参数。
        :return dict: 返回一个字典类型的数据,
                内含以下内容::
                    seconds: float, 表示训练时长
                    以下三个内容只有在提供了dev_data的情况下会有。
                    best_eval: Dict of Dict, 表示evaluation的结果。第一层的key为Metric的名称，
                                第二层的key为具体的Metric
                    best_epoch: int，在第几个epoch取得的最佳值
                    best_step: int, 在第几个step(batch)更新取得的最佳值
        """
        results = {}
        if self.n_epochs <= 0:
            logger.info(f"training epoch is {self.n_epochs}, nothing was done.")
            results['seconds'] = 0.
            return results

        self.start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        start_time = time.time()
        logger.info("training epochs started " + self.start_time)
        self.model_name = "_".join(['best', self.model.__class__.__name__, self.start_time, self.customize])

        # 模型训练
        self.model.train()
        self._train()
        self.end_run()

        # 模型保存和加载
        if self.save_last:
            self._save_model(self.model,"_".join(['last', self.model.__class__.__name__, self.start_time, self.customize]))

        if self.dev_data is not None and self.best_dev_perf is not None:
            logger.info(
                "\nIn Epoch:{}/Step:{}, got best dev performance:{}".format(self.best_dev_epoch, self.best_dev_step,self.best_dev_perf))
            results['best_eval'] = self.best_dev_perf
            results['best_epoch'] = self.best_dev_epoch
            results['best_step'] = self.best_dev_step

            if load_best_model:
                load_succeed = self._load_model(self.model, self.model_name)
                if load_succeed:
                    logger.info("Reloaded the best model.")
                else:
                    logger.info("Fail to reload best model.")

        results['minutes'] = round((time.time() - start_time)/60, 2)

        return results

    def _train(self):
        self.step = 0
        self.epoch = 0
        avg_loss = 0

        start_time = time.time()
        pbar = tqdm(total=self.n_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True)
        for epoch in range(1,self.n_epochs+1):
            self.epoch = epoch
            pbar.set_description_str(desc='Epoch {}/{}'.format(self.epoch,self.n_epochs))

            for batch in self.train_data:
                self.step += 1
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch
                model_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,labels=labels)
                if self.writer:
                    # 指标计算
                    logits = model_output.get('logits')
                    batch_res = evaluate(self.metrics,logits.softmax(dim=-1).to('cpu'),labels.to('cpu'))

                # 损失计算
                loss = model_output.get('loss')
                loss = loss.mean()
                avg_loss += loss.item()

                # 反向传播
                loss = loss / self.update_every
                self._grad_backward(loss)

                # 梯度更新
                self._update()

                if self.step % self.print_every == 0:
                    avg_loss = float(avg_loss) / self.print_every
                    pbar.update(self.print_every)

                    end_time = time.time()
                    diff = timedelta(seconds=round(end_time - start_time))
                    print_output = "[epoch: {:>3} step: {:>4}] train_batch_loss: {:>4.6} time: {}".format(
                        epoch, self.step, avg_loss,diff)
                    pbar.set_postfix_str(print_output)

                    if self.writer:
                        self.writer.add_scalar('train_batch_loss', avg_loss, self.step)

                        train_res = compute_metrics(self.metrics)
                        reset_metrics(self.metrics)
                        for key in train_res:
                            self.writer.add_scalar('train_'+key, train_res[key], self.step)

                    avg_loss = 0

                # evaluation
                if ((self.validate_every > 0 and self.step % self.validate_every == 0) or
                    (self.validate_every < 0 and self.step % len(self.train_data) == 0)) \
                        and self.dev_data is not None and self.metrics is not None:

                    eval_batch_loss, eval_res = self._do_validation(self.dev_data)
                    eval_str = "Evaluation on dev at Epoch {}/{}. Step {}/{} ".format(epoch, self.n_epochs, self.step,
                                                                                self.n_steps)
                    logger.info(eval_str)
                    logger.info('eval_batch_loss:{:>4.6}, {}'.format(eval_batch_loss,_format_eval_results(eval_res)) + '\n')

                    if self.writer:
                        self.writer.add_scalar('eval_batch_loss', eval_batch_loss, self.step)
                        for key in eval_res:
                            self.writer.add_scalar('eval_'+key, eval_res[key], self.step)

                    self.model.train()
                    if self.flag > self.early_stop:
                        logger.info('Earlystop at Epoch {}/{} Step {}/{}'.format(self.epoch,self.n_epochs,self.step,self.n_steps))
                        return

    def end_run(self):
        if self.writer:
            self.writer.close()

    def _update_with_warmup(self):
        """Perform weight update on a model.
        """
        if (self.step) % self.update_every == 0:
            self.optimizer.step()
            self.scheduler.step()

    def _update_without_warmup(self):
        """Perform weight update on a model.
        """
        if (self.step) % self.update_every == 0:
            self.optimizer.step()


    def _do_validation(self,dev_dataloader):
        # eval_data result
        batch_loss,eval_res = self.tester.test(dev_dataloader)
        eval_score = eval_res.get(self.indicator)

        if eval_score > self.best_dev_perf:
            self.flag = 0
            self.best_dev_perf = eval_score
            self.best_dev_epoch = self.epoch
            self.best_dev_step = self.step
            print("****** Best {} ({}) update ****".format(self.indicator,eval_score))
            # Save a trained model
            if self.save_path is not None:
                self._save_model(self.model, self.model_name)
        else:
            self.flag += 1

        return batch_loss,eval_res


    def _save_model(self, model, model_name, only_param=False):
        """ 存储不含有显卡信息的state_dict或model
        :param model:
        :param model_name:
        :param only_param:
        :return:
        """
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
            if _model_contains_inner_module(model):
                model = model.module
            if only_param:
                state_dict = model.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].cpu()
                torch.save(state_dict, model_path)
            else:
                model.cpu()
                torch.save(model, model_path)
                model.to(self.device)

    def _load_model(self, model, model_name, only_param=False):
        # 返回bool值指示是否成功reload模型
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError("file `{}` does not exist. Please make sure model are there.".format(model_name))

            if only_param:
                states = torch.load(model_path)
            else:
                states = torch.load(model_path).state_dict()
            if _model_contains_inner_module(model):
                model.module.load_state_dict(states)
            else:
                model.load_state_dict(states)
        else:
            return False
        return True

    def _set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == 'gpu':
            torch.cuda.manual_seed_all(seed)


class Tester(object):
    def __init__(self,model,metrics = None):
        self.model = model
        self.device = model.device
        self.metrics = metrics

    def predict(self, dataloader):
        '''
        return:
        result = {
                'infer_logits': inference_logits,
                'infer_labels': inference_labels,
                'gold_labels': gold_labels
                }
        '''
        if self.model is None:
            raise FileNotFoundError("model not been loaded.")
        self.model.eval()

        inference_labels = []
        gold_labels = []
        inference_logits = []
        for batch in dataloader:
            batch = tuple(i.to(self.device) for i in batch)
            input_ids, attention_mask, token_type_ids, labels = batch

            with torch.no_grad():
                model_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,
                                            labels=None)
                logits =model_output.get('logits')

            inference_labels.append(np.argmax(logits.to('cpu').numpy(), axis=1))
            gold_labels.append(labels.to('cpu').numpy())
            inference_logits.append(logits.to('cpu').numpy())

        inference_labels = np.concatenate(inference_labels, 0)
        gold_labels = np.concatenate(gold_labels, 0)
        inference_logits = np.concatenate(inference_logits, 0)

        result = {
                'infer_logits': inference_logits,
                'infer_labels': inference_labels,
                'gold_labels': gold_labels
                }
        return result

    def test(self, dataloader):
        '''
        return:
        batch_loss,eval_result
        '''
        self.model.eval()
        reset_metrics(self.metrics)

        loss,nb_steps = 0, 0
        for batch in dataloader:
            batch = tuple(i.to(self.device) for i in batch)
            input_ids, attention_mask, token_type_ids, labels = batch

            with torch.no_grad():
                model_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,
                                            labels=labels)
                tmp_loss, logits =model_output.get('loss'),model_output.get('logits')
                batch_res = evaluate(self.metrics,logits.softmax(dim=-1).to('cpu'),labels.to('cpu'))
                # 记录损失和步次
            loss += tmp_loss.mean().item()
            nb_steps += 1

        eval_batch_loss = loss / nb_steps
        eval_res = compute_metrics(self.metrics)
        reset_metrics(self.metrics)
        return eval_batch_loss,eval_res

def _model_contains_inner_module(model):
    """

    :param nn.Module model: 模型文件，判断是否内部包含model.module, 多用于check模型是否是nn.DataParallel,
        nn.parallel.DistributedDataParallel。主要是在做形参匹配的时候需要使用最内部的model的function。
    :return: bool
    """
    if isinstance(model, nn.Module):
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return True
    return False

def reset_metrics(metrics):
    for key in metrics:
        metrics[key].reset()

def compute_metrics(metrics):
    res = {}
    for key in metrics:
        res[key] = metrics[key].compute().item()
    return res

def evaluate(metrics, y_pred,y_true):
    # report = classification_report(y_true, y_pred, output_dict=True)
    # f1 = f1_score(y_true,y_pred,average='macro')
    # acc = accuracy_score(y_true,y_pred)
    # precision = precision_score(y_true, y_pred, average='macro')
    # recall = recall_score(y_true, y_pred, average='macro')
    # eval_res={'f1':f1,'acc':acc,'precision':precision,'recall':recall}

    eval_res={}
    for key in metrics:
        eval_res[key] = metrics[key](y_pred,y_true).item()
    return eval_res

def _format_eval_results(results):
    """Override this method to support more print formats.
    """
    _str = ", ".join([str(key) + ":" + '{:.4f}'.format(value) for key, value in results.items()])
    return _str

