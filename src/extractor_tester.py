from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import ListField, LabelField, ArrayField
from allennlp.nn import util
from allennlp.data.instance import Instance

import numpy as np
import os
import re
import torch

from extractor_model import TriggerExtractor, ArgumentExtractor
from acereader import CustomSpanField
from extractormetric import ExtractorMetric
from cfg import args


def find_latest_checkpoint(serialization_dir):
    have_checkpoint = (serialization_dir is not None and
                       any("model_state_epoch_" in x for x in os.listdir(serialization_dir)))
    if not have_checkpoint:
        return None

    serialization_files = os.listdir(serialization_dir)
    model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
    found_epochs = [
            re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
            for x in model_checkpoints
    ]
    int_epochs = []
    for epoch in found_epochs:
        pieces = epoch.split('.')
        if len(pieces) == 1:
            # Just a single epoch without timestamp
            int_epochs.append([int(pieces[0]), '0'])
        else:
            # has a timestamp
            int_epochs.append([int(pieces[0]), pieces[1]])
    last_epoch = sorted(int_epochs, reverse=True)[0]
    if last_epoch[1] == '0':
        epoch_to_load = str(last_epoch[0])
    else:
        epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])
    model_path = os.path.join(serialization_dir,
                              "model_state_epoch_{}.th".format(epoch_to_load))
    training_state_path = os.path.join(serialization_dir,
                                       "training_state_epoch_{}.th".format(epoch_to_load))
    return (model_path, training_state_path)


def restore_checkpoint(model, serialization_dir):
    latest_checkpoint = find_latest_checkpoint(serialization_dir)
    assert latest_checkpoint is not None
    model_path, training_state_path = latest_checkpoint
    model_state = torch.load(model_path, map_location=util.device_mapping(-1))
    model.load_state_dict(model_state)


def test_argument_extractor_pipeline(trigger_val_dataset, role_val_dataset, iterator, dataset_meta, role_extractor=None):
    def store_trigger_extractor_output(trigger_out_path):
        pretrained_bert = PretrainedBertEmbedder(
            pretrained_model=args.pretrained_bert,
            requires_grad=True,
            top_layer_only=True)

        trigger_extractor = TriggerExtractor(
            vocab=Vocabulary(),
            embedder=pretrained_bert,
            et_num=dataset_meta.get_et_num())
        restore_checkpoint(trigger_extractor, args.extractor_origin_trigger_dir)
        trigger_extractor.eval()
        trigger_results = {}
        batch_idx = 0
        for data in iterator(trigger_val_dataset, num_epochs=1):
            print(batch_idx)
            batch_idx += 1
            sentences = data['sentence']
            sentence_id = data['sentence_id']
            triggers = data['triggers']

            output = trigger_extractor(sentences, sentence_id, triggers)
            logits = output['logits']
            pred_span = trigger_extractor.metric.get_span(logits)
            trigger_extractor.metric(logits, triggers)
            for idx, sid in enumerate(sentence_id):
                trigger_results[int(sid)] = pred_span[idx]
        with open(trigger_out_path, 'w') as f:
            evaluation = trigger_extractor.metric.get_metric(True)
            s = ""
            for key, value in evaluation.items():
                s = s + key + ": " + str(value) + ' ; '
            s = s + '\n'
            f.write(s)
            trigger_results = sorted(trigger_results.items(), key=lambda item: item[0])
            for item in trigger_results:
                sid, pred_span = item[0], item[1]
                s = str(sid) + '\t'
                for ids, span in enumerate(pred_span):
                    s = s + str(span[0]) + " " + str(span[1]) + " " + dataset_meta.get_event_type_name(span[2])
                    if ids != len(pred_span) - 1:
                        s = s + '\t'
                s = s + '\n'
                f.write(s)

    def transform_trigger_out_to_instance(trigger_out_path):
        def get_instance(s_id, t_list):
            instances = []
            for trigger in t_list:
                trigger_span_start = trigger[0]
                trigger_span_end = trigger[1]
                et_id = trigger[2]

                sentence_field = trigger_val_dataset[s_id]['sentence']
                sentence_id_field = trigger_val_dataset[s_id]['sentence_id']

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

        instances = []
        with open(trigger_out, 'r') as f:
            lines = f.readlines()
            for idx in range(1, len(lines)):
                strs = lines[idx].strip().split('\t')
                sentence_id = int(strs[0])
                trigger_list = []
                for idt in range(1, len(strs)):
                    trigger = strs[idt].strip().split()
                    trigger_span_start = int(trigger[0].strip())
                    trigger_span_end = int(trigger[1].strip())
                    et_id = dataset_meta.get_event_type_id(trigger[2].strip())
                    trigger_list.append([trigger_span_start, trigger_span_end, et_id])
                if len(trigger_list) != 0:
                    sentence_instances = get_instance(sentence_id, trigger_list)
                    instances.extend(sentence_instances)
        return instances

    def collect_true_spans():
        sentence_spans = {}
        for instance in role_val_dataset:
            sentence_id = instance['sentence_id'].label
            if sentence_id not in sentence_spans:
                sentence_spans[sentence_id] = []
            for idx in range(len(instance['roles'])):
                role = instance['roles'][idx]
                sentence_spans[sentence_id].append([role.span_start, role.span_end, role.span_label, role.extra_id])
        return sentence_spans

    def get_argument_extractor_output(instances, role_extractor):
        if role_extractor is None:
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
                af=dataset_meta.af,
                ief=dataset_meta.ief)
        else:
            argument_extractor = role_extractor.cpu()
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
        if role_extractor is None:
            restore_checkpoint(argument_extractor, serialization_dir)
        argument_extractor.eval()
        batch_idx = 0
        pred_spans = {}
        for data in iterator(instances, num_epochs=1):
            print(batch_idx)
            batch_idx += 1
            sentence = data['sentence']
            sentence_id = data['sentence_id']
            type_ids = data['type_ids']
            event_type = data['event_type']
            trigger = data['trigger']
            roles = data['roles']
            output = argument_extractor(sentence, sentence_id, type_ids, event_type, trigger, roles)
            batch_spans = argument_extractor.metric.get_span(output['start_logits'], output['end_logits'], event_type)

            for idb, batch_span in enumerate(batch_spans):
                s_id = int(sentence_id[idb])
                if s_id not in pred_spans:
                    pred_spans[s_id] = []
                pred_spans[s_id].extend(batch_span)
        return pred_spans

    def argument_extractor_evaluation(pred_spans, gold_spans):
        metric = ExtractorMetric()
        for s_id in gold_spans:
            if s_id in pred_spans:
                p_span = pred_spans[s_id]
            else:
                p_span = []
            g_span = gold_spans[s_id]
            metric([p_span], [g_span])
        return metric.get_metric(reset=True)

    trigger_out = './data/tmp/out.trigger'
    if not os.path.exists(trigger_out):
        store_trigger_extractor_output(trigger_out)
    print('=====> Collecting true spans...')
    true_spans = collect_true_spans()
    print('=====> Transforming trigger out to instances...')
    instances = transform_trigger_out_to_instance(trigger_out)
    print('=====> Extracting arguments...')
    pred_spans = get_argument_extractor_output(instances, role_extractor)
    evaluation = argument_extractor_evaluation(pred_spans, true_spans)
    if args.train_argument_with_generation:
        output_file = os.path.join('./data/tmp', str(args.extractor_generated_mask_rate)+"x"+str(args.extractor_generated_timex))
        if args.extractor_sorted:
            output_file = output_file + '-sorted'
    elif args.train_argument_only_with_generation:
        output_file = os.path.join('./data/tmp', str(args.extractor_generated_mask_rate)+"x"+str(args.extractor_generated_timex))
        if args.extractor_sorted:
            output_file = output_file + '-sorted'
        output_file = output_file + '-onlyed'
    else:
        if args.use_loss_weight:
            output_file = './data/tmp/origin.with-loss-weight'
        else:
            output_file = './data/tmp/origin'
    with open(output_file, 'w') as f:
        f.write(str(evaluation))
    print(evaluation)


def test_trigger_extractor(trigger_extractor, iterator, train_dataset, val_dataset, dataset_meta):
    trigger_extractor = trigger_extractor.cpu()
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_vocab)

    for data in iterator(val_dataset):
        sentences = data['sentence']
        sentences_id = data['sentence_id']
        triggers = data['triggers']

        tokens = sentences['tokens']
        offset = sentences['tokens-offsets']

        output = trigger_extractor(sentences, sentences_id, triggers)
        logits = output['logits']
        pred_span = trigger_extractor.metric.get_span(logits)

        print('########### word piece sentence ##########')
        print(bert_tokenizer.convert_ids_to_tokens(tokens[0].numpy()))

        print('########### origin sentence ###########')
        print(bert_tokenizer.convert_ids_to_tokens(tokens[0][offset[0]].numpy()))

        print('########## gold trigger ###########')
        trigger = triggers[0]
        for argu in trigger:
            if argu[0] == -1:
                break
            print('trigger: ', dataset_meta.get_trigger_name(int(argu[3])), "\n",
                  "event_type: ", dataset_meta.get_event_type_name(int(argu[2])), "\n"
                  '(', int(argu[0]), int(argu[1]), '): ', bert_tokenizer.convert_ids_to_tokens(tokens[0][offset[0]][argu[0]: argu[1]+1].numpy()))
            print()

        print('###### pred trigger ######')
        trigger = pred_span[0]
        for argu in trigger:
            print("event_type: ", dataset_meta.get_event_type_name(int(argu[2])), "\n"
                  '(', argu[0], argu[1], '): ', bert_tokenizer.convert_ids_to_tokens(tokens[0][offset[0]][argu[0]: argu[1]+1].numpy()))
            print()
        raise NotImplementedError


def test_argument_extractor(argument_extractor, iterator, train_dataset, val_dataset, dataset_meta):
    argument_extractor = argument_extractor.cpu()
    argument_extractor = argument_extractor.eval()
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_vocab)

    for data in iterator(val_dataset):
        sentences = data['sentence']
        type_ids = data['type_ids']
        sentence_id = data['sentence_id']
        roles = data['roles']
        trigger = data['trigger']
        event_type = data['event_type']

        tokens = sentences['tokens']
        offset = sentences['tokens-offsets']

        output = argument_extractor(sentences, sentence_id, type_ids, event_type, trigger, roles)
        s_logits = output['start_logits']
        e_logits = output['end_logits']
        filtered_span = argument_extractor.metric.get_span(s_logits, e_logits, event_type)

        print('########### word piece sentence ##########')
        print(bert_tokenizer.convert_ids_to_tokens(tokens[0].numpy()))

        print('########### origin sentence ###########')
        print(bert_tokenizer.convert_ids_to_tokens(tokens[0][offset[0]].numpy()))

        print('########### trigger ##########')
        trigger_start = int(trigger[0, 0])
        trigger_end = int(trigger[0, 1])
        print(bert_tokenizer.convert_ids_to_tokens(tokens[0][offset[0]][trigger_start: trigger_end+1].numpy()))
        print(dataset_meta.get_event_type_name(int(event_type[0])))

        print('########## gold arguments ###########')
        role = roles[0]
        for argu in role:
            if argu[0] == -1:
                break
            print('role: ', dataset_meta.get_role_name(int(argu[2])), "\n",
                  "event_type: ", dataset_meta.get_event_type_name(int(argu[3])), "\n"
                  '(', int(argu[0]), int(argu[1]), '): ', bert_tokenizer.convert_ids_to_tokens(tokens[0][offset[0]][argu[0]: argu[1]+1].numpy()))
            print()

        print('###### pred arguments ######')
        role = filtered_span[0]
        for argu in role:
            print('role: ', dataset_meta.get_role_name(int(argu[2])), "\n",
                  "event_type: ", dataset_meta.get_event_type_name(int(argu[3])), "\n"
                  '(', argu[0], argu[1], '): ', bert_tokenizer.convert_ids_to_tokens(tokens[0][offset[0]][argu[0]: argu[1]+1].numpy()))
            print()
        raise NotImplementedError
