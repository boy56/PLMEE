from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import Field, TextField, ListField, LabelField, ArrayField
from allennlp.data.tokenizers import Token

import numpy as np
from overrides import overrides
import torch


class DataMeta:
    def __init__(
        self,
        event_id_file='./data/events.id',
        role_id_file='./data/roles.id'
    ):
        self.event_types = {}
        self.role_labels = {}

        with open(event_id_file, 'r') as f:
            strs = f.readlines()
            for str in strs:
                str_list = str.strip().split('\t')
                id = int(str_list[0])
                et = str_list[1]
                self.event_types[et] = id

        with open(role_id_file, 'r') as f:
            strs = f.readlines()
            for str in strs:
                str_list = str.strip().split('\t')
                id = int(str_list[0])
                role = str_list[1]
                self.role_labels[role] = id
        self.trigger_labels = []
        self.event_roles = [[] for i in range(len(self.event_types))]

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


class TriggerReader(DatasetReader):
    def __init__(self, data_meta, token_indexer=None):
        super().__init__()
        self.token_indexer = token_indexer or {'tokens': SingleIdTokenIndexer()}
        self.data_meta = data_meta

    def str_to_instance(self, line):
        strs = line.strip().split('\t')
        sentence_id = int(strs[0])
        sentence_id_field = LabelField(label=sentence_id, skip_indexing=True, label_namespace='sentence_id')
        words = strs[1].strip().split()
        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexer)

        trigger_field_list = []
        for idx in range(2, len(strs)):
            arguments = strs[idx].strip().split(';')
            trigger = arguments[0].strip().split()
            trigger_span_start = int(trigger[0].strip())
            trigger_span_end = int(trigger[1].strip())

            trigger_id = self.data_meta.get_trigger_id(" ".join(words[trigger_span_start: trigger_span_end+1]))
            et_id = self.data_meta.get_event_type_id(trigger[2].strip())
            trigger_field_list.append(
                CustomSpanField(trigger_span_start, trigger_span_end, et_id, trigger_id))
        if trigger_field_list == []:
            trigger_field_list.append(CustomSpanField(-1, -1, -1, -1))
        trigger_field = ListField(trigger_field_list)
        fields = {'sentence': sentence_field}
        fields['sentence_id'] = sentence_id_field
        fields['triggers'] = trigger_field
        return Instance(fields)

    def _read(self, event_file):
        with open(event_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                yield self.str_to_instance(line)


class RoleReader(DatasetReader):
    def __init__(self, data_meta, token_indexer):
        super().__init__()
        self.token_indexer = token_indexer
        self.data_meta = data_meta
        self.wordpiece_tokenizer = token_indexer['tokens'].wordpiece_tokenizer

    def str_to_instance(self, line):
        instances = []
        strs = line.strip().split('\t')
        sentence_id = int(strs[0])
        words = strs[1].strip().split()

        for idx in range(2, len(strs)):
            arguments = strs[idx].strip().split(';')
            trigger = arguments[0].strip().split()

            tokens = [Token(word) for word in words]
            sentence_field = TextField(tokens, self.token_indexer)
            et_id = self.data_meta.get_event_type_id(trigger[2].strip())
            trigger_span_start = int(trigger[0].strip())
            trigger_span_end = int(trigger[1].strip())
            trigger_id = self.data_meta.get_trigger_id(" ".join(words[trigger_span_start: trigger_span_end+1]))

            type_ids = [0]
            for i, word in enumerate(words):
                word_pieces = self.wordpiece_tokenizer(word)
                if i >= trigger_span_start and i <= trigger_span_end:
                    type_ids.extend([1]*len(word_pieces))
                else:
                    type_ids.extend([0]*len(word_pieces))
            type_ids.append(0)
            type_ids = np.array(type_ids)

            tokens = [Token(word) for word in words]
            sentence_field = TextField(tokens, self.token_indexer)
            sentence_id_field = LabelField(label=sentence_id, skip_indexing=True, label_namespace='sentence_id')
            type_ids_field = ArrayField(type_ids)
            event_type_field = LabelField(label=et_id, skip_indexing=True, label_namespace='event_labels')
            trigger_span_field = CustomSpanField(trigger_span_start, trigger_span_end, et_id, trigger_id)

            role_field_list = []
            for ida in range(1, len(arguments)):
                role = arguments[ida].strip().split()
                role_span_start = int(role[0].strip())
                role_span_end = int(role[1].strip())
                role_id = self.data_meta.get_role_id(role[2].strip())
                self.data_meta.add_event_role(et_id, role_id)
                role_field_list.append(CustomSpanField(role_span_start, role_span_end, role_id, et_id))
            if role_field_list == []:
                role_field_list.append(CustomSpanField(-1, -1, -1, -1))
            roles_field = ListField(role_field_list)
            fields = {'sentence': sentence_field}
            fields['sentence_id'] = sentence_id_field
            fields['type_ids'] = type_ids_field
            fields['event_type'] = event_type_field
            fields['trigger'] = trigger_span_field
            fields['roles'] = roles_field
            instances.append(Instance(fields))
        return instances

    def _read(self, event_file):
        with open(event_file, 'r') as f:
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
