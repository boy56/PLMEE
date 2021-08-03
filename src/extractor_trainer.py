# allennlp 0.9
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR # 等间隔调整学习率

from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import _PyTorchLearningRateSchedulerWrapper
from extractor_model import TriggerExtractor, ArgumentExtractor
from cfg import args
from dueereader import DataMeta, TriggerReader, RoleReader, ETRoleReader, ETTextReader


def argument_dataset_statistics(train_dataset, val_dataset, dataset_meta):
    import matplotlib.pyplot as plt
    import numpy as np

    def get_dataset_et_num(dataset, et_num, use_ratio=True):
        for instance in dataset:
            et = instance['event_type'].label
            et_num[et] += 1
        if use_ratio:
            et_num /= np.sum(et_num)

    def get_dataset_role_num(dataset, r_num, use_ratio=True):
        for instance in dataset:
            for role in instance['roles']:
                label = role.span_label
                if label == -1:
                    continue
                r_num[label] += 1
        if use_ratio:
            r_num /= np.sum(r_num)

    event_type_num = dataset_meta.get_et_num()
    role_type_num = dataset_meta.get_role_num()
    event_num = np.zeros(shape=(2, event_type_num))
    role_num = np.zeros(shape=(2, role_type_num))

    get_dataset_et_num(train_dataset, event_num[0])
    get_dataset_et_num(val_dataset, event_num[1])
    get_dataset_role_num(train_dataset, role_num[0])
    get_dataset_role_num(val_dataset, role_num[1])

    et_x = np.arange(event_type_num)
    et_y = np.transpose(event_num)
    r_x = np.arange(role_type_num)
    r_y = np.transpose(role_num)

    fig1 = plt.subplot(1, 2, 1)
    fig1.plot(et_x, et_y)
    fig1.set_title("event type")
    fig1.legend(labels=['train', 'val'], loc='best')

    fig2 = plt.subplot(1, 2, 2)
    fig2.plot(r_x, r_y)
    fig2.set_title('role type')
    fig2.legend(labels=['train', 'val'], loc='best')

    plt.show()
    raise NotImplementedError


def train_argument_extractor(data_meta, vocab, iterator, train_dataset, val_dataset):
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model=args.pretrained_bert,
        requires_grad=True,
        top_layer_only=True)

    argument_extractor = ArgumentExtractor(
        vocab=vocab,
        embedder=pretrained_bert,
        role_num=data_meta.get_role_num(),
        event_roles=data_meta.event_roles,
        prob_threshold=args.extractor_argument_prob_threshold,
        af=data_meta.af,
        ief=data_meta.ief,
        use_loss_weight=args.use_loss_weight).cuda(args.extractor_cuda_device)

    optimizer = optim.Adam([
        {'params': argument_extractor.embedder.parameters(), 'lr': args.extractor_embedder_lr},
        {'params': argument_extractor.start_tagger.parameters(), 'lr': args.extractor_tagger_lr},
        {'params': argument_extractor.end_tagger.parameters(), 'lr': args.extractor_tagger_lr}])
    scheduler = StepLR(
        optimizer,
        step_size=args.extractor_lr_schduler_step,
        gamma=args.extractor_lr_schduler_gamma)
    learning_rate_scheduler = _PyTorchLearningRateSchedulerWrapper(scheduler)
    # learning_rate_scheduler = scheduler
    if args.train_argument_with_generation:
        serialization_dir = os.path.join(args.extractor_generated_role_dir, str(args.extractor_generated_mask_rate))
        serialization_dir = os.path.join(serialization_dir, "x"+str(args.extractor_generated_timex))
        if args.extractor_sorted:
            serialization_dir = serialization_dir + '-sorted'
    elif args.train_argument_only_with_generation:
        serialization_dir = os.path.join(args.extractor_generated_role_dir, str(args.extractor_generated_mask_rate))
        serialization_dir = os.path.join(serialization_dir, "x"+str(args.extractor_generated_timex))
        if args.extractor_sorted:
            serialization_dir = serialization_dir + '-sorted'
        serialization_dir = serialization_dir + '-onlyed'
    else:
        serialization_dir = args.extractor_origin_role_dir

    trainer = Trainer(
        model=argument_extractor,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        num_epochs=args.extractor_epoc,
        serialization_dir=serialization_dir,
        num_serialized_models_to_keep=1,
        validation_metric='+r_c_f', # 需要与metric中的字段一致, +/- 参照trianer中的定义
        learning_rate_scheduler=learning_rate_scheduler,
        cuda_device=args.extractor_cuda_device)
    trainer.train()
    return argument_extractor


def train_trigger_extractor(data_meta, vocab, iterator, train_dataset, val_dataset):
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model=args.pretrained_bert,
        requires_grad=True,
        top_layer_only=True)

    trigger_extractor = TriggerExtractor(
        vocab=vocab,
        embedder=pretrained_bert,
        et_num=data_meta.get_et_num()).cuda(args.extractor_cuda_device)

    optimizer = optim.Adam([
        {'params': trigger_extractor.embedder.parameters(), 'lr': args.extractor_embedder_lr},
        {'params': trigger_extractor.taggers.parameters(), 'lr': args.extractor_tagger_lr}])
    scheduler = StepLR(
        optimizer,
        step_size=args.extractor_lr_schduler_step,
        gamma=args.extractor_lr_schduler_gamma)
    learning_rate_scheduler = _PyTorchLearningRateSchedulerWrapper(scheduler)
    # learning_rate_scheduler = scheduler
    trainer = Trainer(
        model=trigger_extractor,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        num_epochs=args.extractor_epoc,
        serialization_dir=args.extractor_origin_trigger_dir,
        num_serialized_models_to_keep=3,
        validation_metric='+t_c_f', # 需要与metric中的返回字段一致, +/- 参照trianer中的定义
        learning_rate_scheduler=learning_rate_scheduler,
        cuda_device=args.extractor_cuda_device)
    trainer.train()


if __name__ == '__main__':
    # ==== indexer and reader =====
    bert_indexer = {'tokens': PretrainedBertIndexer(
        pretrained_model=args.bert_vocab,
        use_starting_offsets=True,
        do_lowercase=False)}
    
    data_meta = DataMeta(event_id_file=args.data_meta_dir + "/events.id", role_id_file=args.data_meta_dir + "/roles.id")
    # print(args.istrigger)
    # print(args.isETid)
    if args.istrigger: # 原版PLMEE方式
        trigger_reader = TriggerReader(data_meta=data_meta, token_indexer=bert_indexer)
        role_reader = RoleReader(data_meta=data_meta, token_indexer=bert_indexer)

        # trigger数据加载
        trigger_train_dataset = trigger_reader.read(args.extractor_train_file)
        trigger_val_dataset = trigger_reader.read(args.extractor_val_file)

        if args.do_train_trigger: # trigger 训练
            train_trigger_extractor(data_meta, vocab, iterator, trigger_train_dataset, trigger_val_dataset)
    else:
        role_reader = ETRoleReader(data_meta=data_meta, token_indexer=bert_indexer, isETid=args.isETid)

    # role 数据加载
    role_train_dataset = role_reader.read(args.extractor_train_file)
    role_val_dataset = role_reader.read(args.extractor_val_file)
    data_meta.compute_AF_IEF(role_train_dataset)

    # ==== iterator =====
    vocab = Vocabulary()
    iterator = BucketIterator(
        sorting_keys=[('sentence', 'num_tokens')],
        batch_size=args.extractor_batch_size)
    iterator.index_with(vocab)
    # argument_dataset_statistics(role_train_dataset, role_val_dataset, data_meta)

    if args.do_train_argument:
        assert not (args.train_argument_with_generation and args.train_argument_only_with_generation)

        if args.train_argument_with_generation:
            generated_file = args.extractor_generated_file+"-"+str(args.extractor_generated_mask_rate)
            generated_file = generated_file + "x" + str(args.extractor_generated_timex)
            if args.extractor_sorted:
                generated_file = generated_file + '-sorted'
            generated_dataset = role_reader.read(generated_file)
            role_train_dataset = role_train_dataset + generated_dataset
        if args.train_argument_only_with_generation:
            generated_file = args.extractor_generated_file+"-"+str(args.extractor_generated_mask_rate)
            generated_file = generated_file + "x" + str(args.extractor_generated_timex)
            if args.extractor_sorted:
                generated_file = generated_file + '-sorted'
            generated_dataset = role_reader.read(generated_file)
            role_train_dataset = generated_dataset
        role_extractor = train_argument_extractor(data_meta, vocab, iterator, role_train_dataset, role_val_dataset)
