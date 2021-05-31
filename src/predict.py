# input: test json file   {"sentence": foo, "id": }
# output: test result json file 比赛结果格式
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import ListField, LabelField, ArrayField
from allennlp.nn import util
from allennlp.data.instance import Instance

import argparse
import numpy as np
import os
import re
import torch
import pickle as pkl
from tqdm import tqdm
import codecs
import json

from extractor_model import TriggerExtractor, ArgumentExtractor
from dueereader import CustomSpanField, DataMeta, TriggerReader, RoleReader, TextReader, TextReaderPro
from allennlp.data.iterators import BucketIterator
from extractormetric import ExtractorMetric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
device_num = -1
if torch.cuda.is_available():
    device_num = torch.cuda.current_device()

print(device)

parser = argparse.ArgumentParser(description='Predict DuEE or DuEE-Fin')
parser.add_argument('--pretrained_bert', type=str, default='BertPara/pytorch/chinese_roberta_wwm_ext')
parser.add_argument('--bert_vocab', type=str, default='BertPara/pytorch/chinese_roberta_wwm_ext/vocab.txt')

parser.add_argument('--save_trigger_dir', type=str, default='./save/DuEE/bert_large/trigger/model_state_epoch_27.th')
parser.add_argument('--save_role_dir', type=str, default='./save/DuEE/bert_large/role/model_state_epoch_29.th')

parser.add_argument('--data_meta_dir', type=str, default='./data/DuEE')
parser.add_argument('--extractor_train_file', type=str, default='./data/DuEE/train.json')
parser.add_argument('--extractor_val_file', type=str, default='./data/DuEE/dev.json')
parser.add_argument('--extractor_test_file', type=str)
parser.add_argument('--extractor_batch_size', type=int, default=28)
parser.add_argument('--extractor_argument_prob_threshold', type=float, default=0.5)

parser.add_argument('--mode', type=str, default='DuEE', help="DuEE or DuEE-Fin") # DuEE or DuEE-Fin
parser.add_argument('--bert_mode', type=str, default='bert_base', help='bert_base or bert_large') # bert_base or bert_large
parser.add_argument('--et_mode', action="store_true", help='if set it, mode is et + ner')

args = parser.parse_args()



# trigger 提取步骤
def trigger_extractor_deal(pre_dataset, iterator, trigger_model_path, dataset_meta):
    def get_instance(sentence_data, t_list):
        instances = []
        for trigger in t_list:
            trigger_span_start = trigger[0]
            trigger_span_end = trigger[1]
            et_id = trigger[2]

            sentence_field = sentence_data['sentence']
            sentence_id_field = sentence_data['sentence_id']

            wordpiece_tokenizer = sentence_field._token_indexers['tokens'].wordpiece_tokenizer
            tokens_len = len(sentence_field)
            type_ids = [0]
            for idx in range(tokens_len):
                word_pieces = wordpiece_tokenizer(sentence_field[idx].text)
                if idx >= trigger_span_start and idx <= trigger_span_end:
                    type_ids.extend([1]*len(word_pieces))
                else:
                    type_ids.extend([0]*len(word_pieces))
            type_ids.append(0)
            type_ids = np.array(type_ids)
            type_ids_field = ArrayField(type_ids)
            event_type_field = LabelField(label=et_id, skip_indexing=True)
            trigger_span_field = CustomSpanField(trigger_span_start, trigger_span_end, et_id, -1)
            role_field_list = [CustomSpanField(-1, -1, -1, -1)]
            roles_field = ListField(role_field_list)
            fields = {'sentence': sentence_field}
            fields['sentence_id'] = sentence_id_field
            fields['type_ids'] = type_ids_field
            fields['event_type'] = event_type_field
            fields['trigger'] = trigger_span_field
            fields['roles'] = roles_field
            instances.append(Instance(fields))
        return instances
    
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model=args.pretrained_bert,
        requires_grad=True,
        top_layer_only=True)

    trigger_extractor = TriggerExtractor(
        vocab=Vocabulary(),
        embedder=pretrained_bert,
        et_num=dataset_meta.get_et_num())

    # 加载trigger_model_path下的模型
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
    # model_state = torch.load(trigger_model_path, map_location=util.device_mapping(-1)) # 在cpu上加载
    model_state = torch.load(trigger_model_path, map_location=device)
    trigger_extractor.load_state_dict(model_state)
    trigger_extractor.to(device)
    trigger_extractor.eval()

    # 对数据进行预测
    trigger_results = {}
    batch_idx = 0
    id_data = {} # sentence_id 与 sentence对应关系
    # print(pre_dataset)
    for instance in pre_dataset:
        tmp_id = instance['sentence_id'].metadata
        if tmp_id in id_data:
            print(tmp_id)
        id_data[tmp_id] = instance

    for data in tqdm(iterator(pre_dataset, num_epochs=1)):
        # print(batch_idx)
        batch_idx += 1
        # print(device_num)
        data = util.move_to_device(data, cuda_device=device_num) 
        sentences = data['sentence']
        sentence_id = data['sentence_id']
        # sentences.to(device)
        output = trigger_extractor(sentences, sentence_id)
        logits = output['logits']
        pred_span = trigger_extractor.metric.get_span(logits)
        for idx, sid in enumerate(sentence_id):
            trigger_results[sid] = pred_span[idx]
    
    # 整理结果
    instances = []
    for sid, trigger_spans in trigger_results.items():
        if len(trigger_spans) > 0:
            sentence_instances = get_instance(id_data[sid], trigger_spans)
            instances.extend(sentence_instances)
    return instances


# role 提取步骤
def argument_extractor_deal(instances, iterator, argument_model_path, dataset_meta):
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model=args.pretrained_bert,
        requires_grad=True,
        top_layer_only=True)

    argument_extractor = ArgumentExtractor(
        vocab=Vocabulary(),
        embedder=pretrained_bert,
        role_num=dataset_meta.get_role_num(),
        event_roles=dataset_meta.event_roles,
        prob_threshold=args.extractor_argument_prob_threshold,
        af=0,
        ief=0)
    
    # 加载argument_model_path下的模型
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
    # model_state = torch.load(argument_model_path, map_location=util.device_mapping(-1))
    model_state = torch.load(argument_model_path, map_location=device)
    argument_extractor.load_state_dict(model_state)
    argument_extractor.to(device)
    argument_extractor.eval()
    
    batch_idx = 0
    pred_spans = {}
    for data in iterator(instances, num_epochs=1):
        print(batch_idx)
        batch_idx += 1
        data = util.move_to_device(data, cuda_device=device_num)
        sentence = data['sentence']
        sentence_id = data['sentence_id']
        type_ids = data['type_ids']
        event_type = data['event_type']
        trigger = data['trigger']
        roles = data['roles']
        output = argument_extractor(sentence, sentence_id, type_ids, event_type, trigger)
        batch_spans = argument_extractor.metric.get_span(output['start_logits'], output['end_logits'], event_type)

        for idb, batch_span in enumerate(batch_spans):
            s_id = sentence_id[idb]
            if s_id not in pred_spans:
                pred_spans[s_id] = []
            pred_spans[s_id].extend(batch_span)
    # print(pred_spans)
    return pred_spans


if __name__ == "__main__":
    
    # print(args.extractor_test_file)
    # print(args.save_trigger_dir)

    # ==== indexer and reader =====
    bert_indexer = {'tokens': PretrainedBertIndexer(
        pretrained_model=args.bert_vocab,
        use_starting_offsets=True,
        do_lowercase=False)}
    data_meta = DataMeta(event_id_file=args.data_meta_dir + "/events.id", role_id_file=args.data_meta_dir + "/roles.id")
    
    trigger_reader = TriggerReader(data_meta=data_meta, token_indexer=bert_indexer)
    role_reader = RoleReader(data_meta=data_meta, token_indexer=bert_indexer)

    # ==== dataset =====
    trigger_train_dataset = trigger_reader.read(args.extractor_train_file)
    role_train_dataset = role_reader.read(args.extractor_train_file)
    trigger_val_dataset = trigger_reader.read(args.extractor_val_file)
    role_val_dataset = role_reader.read(args.extractor_val_file)
    data_meta.compute_AF_IEF(role_train_dataset) # 根据AF_IEF来计算role的重要程度
    # print(trigger_val_dataset)
    # print(pre_dataset)

    # ==== iterator =====
    vocab = Vocabulary()
    iterator = BucketIterator(
        sorting_keys=[('sentence', 'num_tokens')],
        batch_size=args.extractor_batch_size)
    iterator.index_with(vocab)

    trigger_model_path = args.save_trigger_dir
    argument_model_path = args.save_role_dir
    
    result_dir = "./data/" + args.mode + "/tmp/" + args.bert_mode
    instance_pkl_path = result_dir + "/tirgger_deal_instance.pkl"
    result_pkl_path = result_dir + "/role_result.pkl"
    output_file =  result_dir + "/" + args.mode + ".json"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    text_reader = TextReader(data_meta=data_meta, token_indexer=bert_indexer)
    pre_dataset = text_reader.read(args.extractor_test_file) # 用来预测的文件
    
    print('=====> Extracting triggers...')
    if not os.path.exists(instance_pkl_path):
        instances = trigger_extractor_deal(pre_dataset=pre_dataset, iterator=iterator, trigger_model_path=trigger_model_path, dataset_meta=data_meta)
        pkl.dump(instances, open(instance_pkl_path, 'wb'))
    else:
        instances = pkl.load(open(instance_pkl_path, 'rb'))
    print("instances num:", len(instances))
    # exit(0)
    print('=====> Extracting arguments...')
    if not os.path.exists(result_pkl_path):
        pred_spans = argument_extractor_deal(instances=instances, iterator=iterator, argument_model_path=argument_model_path, dataset_meta=data_meta)
        pkl.dump(pred_spans, open(result_pkl_path, 'wb'))
    else:
        pred_spans = pkl.load(open(result_pkl_path, 'rb'))
    
    print('=====> output to json files: ', output_file)
    if args.mode == "DuEE":
        id_sentence = {} # sid 与sentence 对应关系
        for data in pre_dataset:
            id_sentence[data['sentence_id'].metadata] = data['origin_text'].metadata
        with codecs.open(output_file, 'w', 'UTF-8') as f:
            for sid, pred_span in pred_spans.items():
                text = id_sentence[sid]
                # print(text)
                tmp = {}
                tmp['id'] = sid
                tmp_elist = []
                for ids, span in enumerate(pred_span):
                    e_dict = {}
                    e_dict['event_type'] = data_meta.get_event_type_name(span[3])
                    e_dict['arguments'] = [
                        {
                            "role": data_meta.get_role_name(span[2]),
                            "argument": text[span[0]: span[1] + 1]
                        }
                    ]

                    tmp_elist.append(e_dict)
                tmp['event_list'] = tmp_elist
                tmp = json.dumps(tmp, ensure_ascii=False)
                f.write(tmp + "\n")

    elif args.mode == "DuEE-Fin":
        id_sentence = {} # sent_id 与 origin_text 对应关系
        sentid_textid = {} # sent_id 与 text_id 对应关系, sent_id由内容而定, 不唯一
        for data in pre_dataset:
            if data['sentence_id'].metadata in sentid_textid:
                sentid_textid[data['sentence_id'].metadata].append(data['text_id'].metadata)
            else:
                sentid_textid[data['sentence_id'].metadata] = [data['text_id'].metadata]
            
            id_sentence[data['sentence_id'].metadata] = data['origin_text'].metadata
        
        with codecs.open(output_file, 'w', 'UTF-8') as f:
            textid_result = {} # {id: event_list}
            for sid, pred_span in pred_spans.items():
                text = id_sentence[sid]
                # text_id = sentid_textid[sid]
                tmp_elist = []
                e_set = set()
                role_dict = {}
                for ids, span in enumerate(pred_span):
                    et = data_meta.get_event_type_name(span[3])
                    role_info = {
                        "role": data_meta.get_role_name(span[2]),
                        "argument": text[span[0]: span[1] + 1]
                    }
                    
                    tmp_elist.append({
                        "event_type": et,
                        "arguments": [role_info]
                    })

                    # 将出现的所有et与role进行存储, 百度官方baseline的postprocess方法
                    e_set.add(et)
                    if role_info['role'] in role_dict:
                        role_dict[role_info['role']].append(role_info)
                    else:
                        role_dict[role_info['role']] = [role_info]

                for text_id in sentid_textid[sid]:
                    # 相同text_id 结果进行合并
                    if text_id in textid_result:
                        textid_result[text_id].extend(tmp_elist)
                    else:
                        textid_result[text_id] = tmp_elist
            
            # 将结果组合成答案
            for text_id, event_list in textid_result.items():
                tmp = {}
                tmp['id'] = text_id
                
                # 将event_type相同的结果进行合并
                et_set_dict = {} # 用于et 中的role 去重, 百度官方会进行去重
                et_roles_dict = {} # 用于相同et的role合并
                for e in event_list:
                    tmp_et = e['event_type']
                    # 遇到了新的事件类型
                    if tmp_et not in et_roles_dict:
                        et_roles_dict[tmp_et] = []
                        et_set_dict[tmp_et] = set()
                    for rl in e['arguments']:
                        if rl["role"] + "-" + rl["argument"] not in et_set_dict[tmp_et]:
                            et_roles_dict[tmp_et].append(rl)
                            et_set_dict[tmp_et].add(rl["role"] + "-" + rl["argument"])
                
                event_list = [{"event_type": et, "arguments": roles} for et, roles in et_roles_dict.items()]
                
                tmp['event_list'] = event_list
                
                tmp = json.dumps(tmp, ensure_ascii=False)
                f.write(tmp + "\n")