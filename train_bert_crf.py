import torch
# from models.ner_model import BertSoftmaxForNer, BertSoftmaxForNer_
from models.ner_model import BertCrfForNer
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
import os
import numpy as np
from os.path import join
from loguru import logger
import time
from transformers import BertTokenizer, BertConfig
from torch.utils.data import Dataset, DataLoader
from processors.processor import CnerProcessor
from dataset import NerDataset
import json
from tqdm import tqdm
from metrics.ner_metrics import SeqEntityScore
import transformers


def set_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument("--output_path", type=str, default='output/bert_softmax_crf', help='模型与预处理数据的存放位置')

    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'], help='损失函数类型')
    parser.add_argument("--lr", type=float, default=5e-5, help='Bert的学习率')
    parser.add_argument("--crf_lr", default=5e-3, type=float,  help="The initial learning rate for crf and linear layer.")
    parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--eval_step", type=int, default=25, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=150, help="max length of input")
    parser.add_argument("--data_path", type=str, default="datasets/cner/", help='数据集存放路径')
    # parser.add_argument("--train_file", type=str, default="datasets/cner/train.txt")
    # parser.add_argument("--dev_file", type=str, default="datasets/cner/dev.txt")
    # parser.add_argument("--test_file", type=str, default="datasets/cner/test.txt")
    parser.add_argument("--dataset_name", type=str, choices=['cner', "cluener"], default='cner', help='数据集名称')
    # parser.add_argument("--pretrain_model_path", type=str,
    #                     default="pretrain_model/bert-base-chinese")
    parser.add_argument("--pretrain_model_path", type=str, default="pretrain_model/bert-base-chinese")
    # parser.add_argument("--load_model_weights", action='store_true', default=False, help='是否加载预训练模型权重')
    # parser.add_argument("--overwrite_cache", action='store_true', default=True, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)

    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'], help='数据集的标注方式')
    parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.')
    args = parser.parse_args()
    return args


def seed_everything(seed=42):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, dev_loader, optimizer, scheduler, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)
            loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids)
            loss = loss.mean()  # 对多卡的loss取平均

            # 梯度累积
            loss = loss / args.grad_acc_step
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if step % args.grad_acc_step == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            # 评测验证集上的指标
            if step % args.eval_step == 0:
                result = evaluate(args, model, dev_loader)
                precision = result['acc']
                recall = result['recall']
                f1 = result['f1']
                loss = result['loss']
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('f1', f1, step)
                writer.add_scalar('precision', precision, step)
                writer.add_scalar('recall', recall, step)

                model.train()
                if best < f1:
                    best = f1
                    torch.save(model.state_dict(), join(args.output_path, 'ner_model.pt'))
                    logger.info('higher f1: {} in step {} epoch {}, save model'.format(best, step, epoch))


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :param args:
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()
    device = args.device
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    # Eval!
    logger.info("***** Running evaluation %s *****")
    logger.info("  Num examples = {}".format(len(dataloader)))
    logger.info("  Batch size = {}".format(args.batch_size_eval))
    eval_loss = 0.0  #
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)
            loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids)
            # loss, logits = model(input_ids, token_type_ids, label_ids, attention_mask)
            loss = loss.mean()  # 对多卡的loss取平均
            eval_loss += loss

            input_lens = (torch.sum(input_ids != 0, dim=-1) - 2).tolist()   # 减去padding的[CLS]与[SEP]
            # preds = torch.argmax(logits, dim=2)[:, 1:].tolist()  # 减去padding的[CLS]
            preds = model.crf.decode(logits, attention_mask).squeeze(0)
            preds = preds[:, 1:].tolist()  # 减去padding的[CLS]
            label_ids = label_ids[:, 1:].tolist()   # 减去padding的[CLS]
            # preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            # label_ids = label_ids.cpu().numpy().tolist()
            for i in range(len(label_ids)):
                input_len = input_lens[i]
                pred = preds[i][:input_len]
                label = label_ids[i][:input_len]
                metric.update(pred_paths=[pred], label_paths=[label])

    logger.info("\n")
    eval_loss = eval_loss / len(dataloader)
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****")
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results


def main(args):
    # 加载模型
    config = BertConfig.from_pretrained(args.pretrain_model_path, num_labels=20)
    config.loss_type = args.loss_type
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=True)
    model = BertCrfForNer.from_pretrained(args.pretrain_model_path, config=config).to(args.device)
    # 加载数据集
    if args.dataset_name == 'cner':
        processor = CnerProcessor(args.train_file, args.dev_file, args.test_file, tokenizer, args.max_len)
    elif args.dataset_name == 'cluener':
        # todo
        pass
    args.id2label = processor.id2label
    args.label2id = processor.label2id

    # 训练
    if args.do_train:
        # 加载数据集
        train_data = processor.get_train_examples()
        train_dataset = NerDataset(train_data)
        train_dataset = train_dataset[:8]
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_data = processor.get_dev_examples()
        dev_dataset = NerDataset(dev_data)
        dev_dataset = dev_dataset[:8]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.lr},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_lr},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_lr},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_lr},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_lr}
        ]
        t_total = len(train_dataloader) // args.grad_acc_step * args.epochs
        warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        train(model, train_dataloader, dev_dataloader, optimizer, scheduler, args)

    # 测试集上的指标
    if args.do_eval:
        test_data = processor.get_test_examples()
        test_dataset = NerDataset(test_data)
        test_dataset = test_dataset[:8]
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        path = join(args.output_path, 'ner_model.pt')
        logger.info(path)
        # model.load_state_dict(torch.load(path))
        model.eval()
        result = evaluate(args, model, test_dataloader)
        precision = result['acc']
        recall = result['recall']
        f1 = result['f1']
        loss = result['loss']
        logger.info('testset precision:{}, recall:{}, f1:{}, loss:{}'.format(precision, recall, f1, loss.item()))


if __name__ == '__main__':
    args = set_train_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    args.output_path = join(args.output_path, args.dataset_name, 'bsz-{}-lr-{}'.format(args.batch_size_train, args.lr))
    args.train_file = join(args.data_path, 'train.txt')
    args.dev_file = join(args.data_path, 'dev.txt')
    args.test_file = join(args.data_path, 'test.txt')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        logger.info(args)
        writer = SummaryWriter(args.output_path)
    main(args)
