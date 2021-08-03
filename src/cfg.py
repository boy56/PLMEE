import argparse


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print(f'Unparsed args: {unparsed}')
    return args, unparsed


arg_lists = []
parser = argparse.ArgumentParser()

extractor = parser.add_argument_group('extractor')
extractor.add_argument('--pretrained_bert', type=str, default='BertPara/bert-base-cased')
extractor.add_argument('--bert_vocab', type=str, default='BertPara/bert-base-cased/vocab.txt')
extractor.add_argument('--extractor_origin_trigger_dir', type=str, default='./save/origin/trigger')
extractor.add_argument('--extractor_origin_role_dir', type=str, default='./save/origin/role')
extractor.add_argument('--data_meta_dir', type=str, default='./data/dataset_meta')
extractor.add_argument('--extractor_train_file', type=str, default='./data/events.train')
extractor.add_argument('--extractor_val_file', type=str, default='./data/events.val')
extractor.add_argument('--extractor_cuda_device', type=int, default=0)

# 训练和应用版本
extractor.add_argument('--istrigger', action="store_true") # True: trigger + argument方式; False: ET + text -> argument方式
extractor.add_argument('--isETid', action="store_true") # 在上述为False时, True: ETid + text, False: ETText + text


extractor.add_argument('--extractor_embedder_lr', type=float, default=1e-5)
extractor.add_argument('--extractor_tagger_lr', type=float, default=3e-4)
extractor.add_argument('--extractor_batch_size', type=int, default=28)
extractor.add_argument('--extractor_epoc', type=int, default=40)
extractor.add_argument('--use_loss_weight', action="store_true")
extractor.add_argument('--extractor_lr_schduler_step', type=int, default=15)
extractor.add_argument('--extractor_lr_schduler_gamma', type=float, default=0.3)
extractor.add_argument('--extractor_argument_prob_threshold', type=float, default=0.5)
# extractor.add_argument('--do_test', type=bool, default=False)
extractor.add_argument('--do_train_trigger', action="store_true")
extractor.add_argument('--do_train_argument', action="store_true")

# PLMEE 论文中采用额外生成的数据进行训练的参数设置
extractor.add_argument('--train_argument_with_generation', action="store_true")
extractor.add_argument('--train_argument_only_with_generation', action="store_true")
extractor.add_argument('--extractor_generated_mask_rate', type=float, default=0.)
extractor.add_argument('--extractor_generated_timex', type=float, default=1.)
extractor.add_argument('--extractor_sorted', action="store_true")
extractor.add_argument('--extractor_generated_file', type=str, default='./data/events.generated')
extractor.add_argument('--extractor_generated_role_dir', type=str, default='./save/generated/')

args, _ = get_args()
