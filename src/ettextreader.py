# input: ET + Text  output: role
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import Field, TextField, ListField, LabelField, ArrayField, MetadataField
from allennlp.data.tokenizers import Token

import numpy as np
from overrides import overrides
import torch
import json
import codecs


class DataMeta:
    def __init__(
        self,
        event_id_file='./data/DUEE_events.id',
        role_id_file='./data/DUEE_roles.id'
    ):
        self.event_types = {} # {"et": id}
        self.role_labels = {} # {"role": id}

        with codecs.open(event_id_file, 'r', "UTF-8") as f:
            strs = f.readlines()
            for str in strs:
                str_list = str.strip().split('\t')
                id = int(str_list[0])
                et = str_list[1]
                self.event_types[et] = id

        with codecs.open(role_id_file, 'r', 'UTF-8') as f:
            strs = f.readlines()
            for str in strs:
                str_list = str.strip().split('\t')
                id = int(str_list[0])
                role = str_list[1]
                self.role_labels[role] = id
        self.trigger_labels = []
        self.event_roles = [[] for i in range(len(self.event_types))] # 每一个event_types都有一个 event_role列表

    # ==============AF/IEF==============
    def compute_AF_IEF(self, dataset):
        event_num = len(self.event_types)
        role_num = len(self.role_labels)

        self.af = np.zeros((role_num, event_num), dtype=float)
        self.ief = np.zeros((role_num, ), dtype=float)

        for idx, instance in enumerate(dataset):
            event_id = instance['event_type'].label

            for role in instance['roles']:
                role_id = role.span_label
                if role_id == -1:
                    continue
                self.af[role_id, event_id] += 1

        for idx in range(role_num):
            self.ief[idx] = np.log(event_num/np.sum(self.af[idx] != 0))
        role_count_per_event = np.sum(self.af, axis=0)
        self.af = self.af / (role_count_per_event+1e-13)

    # =============event roles============
    # 后续更改为从schema文件中获取, 而不是从train等数据集中
    def add_event_role(self, event_id, role_id):
        if role_id not in self.event_roles[event_id]:
            self.event_roles[event_id].append(role_id)

    # ==============event types=================
    def get_event_type_id(self, label):
        assert label in self.event_types
        return self.event_types[label]

    def get_event_type_name(self, id):
        for key, value in self.event_types.items():
            if value == id:
                return key

    def get_et_num(self):
        return len(self.event_types)

    # ===============roles===============
    def get_role_id(self, role):
        assert role in self.role_labels
        return self.role_labels[role]

    def _get_role_name(self, id):
        for key, value in self.role_labels.items():
            if value == id:
                return key

    def get_role_name(self, id):
        if isinstance(id, int):
            return self._get_role_name(id)
        elif isinstance(id, list):
            s = ''
            for item in id:
                s = s + self._get_role_name(item)+', '
            return s

    def get_role_num(self):
        return len(self.role_labels)

    # ===============triggers===============
    def get_trigger_id(self, trigger):
        if trigger not in self.trigger_labels:
            self.trigger_labels.append(trigger)
        return self.trigger_labels.index(trigger)

    def get_trigger_name(self, id):
        return self.trigger_labels[id]

    def get_trigger_num(self):
        return len(self.trigger_labels)

# 不识别trigger, 将et内容拼在sentence前面进行训练
# 修改参考: https://guide.allennlp.org/representing-text-as-features#6
class RoleReaderPro(DatasetReader):
    def __init__(self, data_meta, token_indexer):
        super().__init__()
        self.token_indexer = token_indexer
        self.data_meta = data_meta
        self.wordpiece_tokenizer = token_indexer['tokens'].wordpiece_tokenizer

    def str_to_instance(self, line):
        # DUEE Reader
        instances = []
        line = json.loads(line)
        sentence_id = line['id']
        words = line['text']

        for event in line['event_list']:
            et = event['event_type']
            et_id = self.data_meta.get_event_type_id(event['event_type'])            
            
            # et + text拼接, 注意 et 拼在前面的话会影响role index 标签
            type_ids = [0] # [CLS]为0的位置
            type_ids = type_ids + [0] * len(et) # 事件类别信息
            type_ids.append(0) # 中间的[SEP]标记
            type_ids.extend([1] * len(words)) # 文本的type标记
            type_ids.append(1) # 最后的[SEP]标记
            
            type_ids = np.array(type_ids)
            
            
            # 将et拼在words前面
            tokens = [Token(w) for w in et] + [Token('[SEP]')] + [Token(word) for word in words] # et + text
            sentence_field = TextField(tokens, self.token_indexer)

            sentence_id_field = MetadataField(sentence_id)
            type_ids_field = ArrayField(type_ids) # 用来标识trigger的位置
            event_type_field = LabelField(label=et_id, skip_indexing=True, label_namespace='event_labels')

            # 处理role的信息
            role_field_list = []
            for argument in event['arguments']:
                if 'argument_start_index' not in argument: # DuEE-Fin 中role 为环节的时候没有start_index
                    continue
                # print(argument)
                role = argument['argument']
                role_type = argument['role']
                # role_span_start = argument['argument_start_index'] # 当类别信息拼在句子前面时, 需要修改start_index信息
                role_span_start = argument['argument_start_index'] + len(et) + 1 # et长度 + [SEP]
                
                role_span_end = role_span_start + len(role) - 1
                role_id = self.data_meta.get_role_id(role_type)
                self.data_meta.add_event_role(et_id, role_id) # 加载数据的时候更新schema
                role_field_list.append(CustomSpanField(role_span_start, role_span_end, role_id, et_id))
            if role_field_list == []:
                role_field_list.append(CustomSpanField(-1, -1, -1, -1))
            roles_field = ListField(role_field_list)
            fields = {'sentence': sentence_field}
            fields['sentence_id'] = sentence_id_field
            fields['type_ids'] = type_ids_field
            fields['event_type'] = event_type_field
            # fields['trigger'] = trigger_span_field
            fields['roles'] = roles_field
            instances.append(Instance(fields))
        return instances

    def _read(self, event_file):
        with codecs.open(event_file, 'r', 'UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                instances = self.str_to_instance(line)
                for instance in instances:
                    yield instance


class TextReaderPro(DatasetReader):
    def __init__(self, data_meta, token_indexer):
        super().__init__()
        self.token_indexer = token_indexer
        self.data_meta = data_meta
        self.wordpiece_tokenizer = token_indexer['tokens'].wordpiece_tokenizer

    def str_to_instance(self, line):
        # 纳入et信息
        
        line = json.loads(line)
        
        words = line['text']
        et_list = line['et_list']
        instances = []

        for et in et_list:
            words_field = MetadataField(words)
            
            '''
            # 原句信息
            type_ids = [0] # [CLS]为0的位置
            type_ids.extend([0] * len(words))
            type_ids.append(0) # 中间的[SEP]标记
            # 事件类别信息
            # type_ids = type_ids + [1] # 拼接事件类型id
            type_ids = type_ids + [1] * len(et) # 拼接事件类型文本信息
            type_ids.append(1) # 最后的[SEP]标记
            type_ids = np.array(type_ids)
            '''

            # et + text拼接
            type_ids = [0] # [CLS]为0的位置
            # 事件类别信息, 用id 代替事件类别信息
            type_ids = type_ids + [0] * len(et)
            # type_ids = type_ids + [0] # et_id
            type_ids.append(0) # 中间的[SEP]标记
            
            type_ids.extend([1] * len(words))
            type_ids.append(1) # 最后的[SEP]标记
            
            type_ids = np.array(type_ids)

            # 获取事件id
            et_id = self.data_meta.get_event_type_id(et)
            event_type_field = LabelField(label=et_id, skip_indexing=True)

            # tokens = [Token(word) for word in words] + [Token('[SEP]')] + [Token(str(et_id))]  # 拼接事件类型id
            # tokens = [Token(word) for word in words] + [Token('[SEP]')] + [Token(w) for w in et]  # 拼接事件类型文本信息
            tokens = [Token(w) for w in et] + [Token('[SEP]')] + [Token(word) for word in words] # et + text
            # tokens = [Token(str(et_id))] + [Token('[SEP]')] + [Token(word) for word in words] # et_id + text
            sentence_field = TextField(tokens, self.token_indexer)
            type_ids_field = ArrayField(type_ids) # 用来标识et的位置

            fields = {'sentence': sentence_field}
            fields['origin_text'] = words_field
            fields['type_ids'] = type_ids_field
            fields['event_type'] = event_type_field

            if 'sent_id' in line: # DuEE-Fin 用来标识句子id
                sentence_id = line['sent_id']
                sentence_id_field = MetadataField(sentence_id)
                fields['sentence_id'] = sentence_id_field

                text_id = line['id']
                text_id_field = MetadataField(text_id)
                fields['text_id'] = text_id_field
            else: # DuEE
                sentence_id = line['id']
                sentence_id_field = MetadataField(sentence_id)
                fields['sentence_id'] = sentence_id_field
            
            instances.append(Instance(fields))

        return instances

    def _read(self, event_file):
        with codecs.open(event_file, 'r', 'UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                instances = self.str_to_instance(line)
                for instance in instances:
                    yield instance


class CustomSpanField(Field[torch.Tensor]):
    def __init__(self,
                 span_start: int,
                 span_end: int,
                 span_label: int,
                 extra_id: int) -> None:
        self.span_start = span_start
        self.span_end = span_end
        self.span_label = span_label
        self.extra_id = extra_id

        if not isinstance(span_start, int) or not isinstance(span_end, int):
            raise TypeError(f"SpanFields must be passed integer indices. Found span indices: "
                            f"({span_start}, {span_end}) with types "
                            f"({type(span_start)} {type(span_end)})")
        if span_start > span_end:
            raise ValueError(f"span_start must be less than span_end, "
                             f"but found ({span_start}, {span_end}).")

    @overrides
    def get_padding_lengths(self):
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths) -> torch.Tensor:
        # pylint: disable=unused-argument
        tensor = torch.LongTensor([self.span_start, self.span_end, self.span_label, self.extra_id])
        return tensor

    @overrides
    def empty_field(self):
        return CustomSpanField(-1, -1, -1, -1)

    def __str__(self) -> str:
        return f"SpanField with spans: ({self.span_start}, {self.span_end}, {self.span_label}, {self.extra_id})."

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 4:
            return other == (self.span_start, self.span_end, self.span_label, self.extra_id)
        else:
            return id(self) == id(other)

