import os
import re
import argparse
import random
from xml.etree.ElementTree import ElementTree
from html.parser import HTMLParser
from pytorch_pretrained_bert.tokenization import BasicTokenizer as BertTokenizer
from allennlp.data.tokenizers import WordTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/acedata')
parser.add_argument('--ace_corpus', type=str, default='./data/ace_2005_td_v7/data/English')
parser.add_argument('--do_lower_case', type=bool, default=False)
parser.add_argument('--events_id_file', type=str, default='./data/events.id')
parser.add_argument('--roles_id_file', type=str, default='./data/roles.id')
parser.add_argument('--train_file', type=str, default='./data/events.train')
parser.add_argument('--test_file', type=str, default='./data/events.val')
parser.add_argument('--pattern_file', type=str, default='./data/events.pattern.new')
parser.add_argument('--roles_dir', type=str, default='./data/roles')
ace_args, _ = parser.parse_known_args()


class SGMLParser(HTMLParser):
    content = ""
    line_content = []

    def init(self, path):
        self.path = path

    def handle_data(self, text):
        text = text.replace("&", "&----")
        self.line_content.append(text)
        self.content += text


class EventType:
    def __init__(self, name):
        self.count = 1
        self.name = name

    def plus_one(self):
        self.count += 1

    def __str__(self):
        s = self.name + ": " + str(self.count)
        return s


class Argument:
    def __init__(self, role, role_name, span_start, span_end):
        self.role = role
        self.role_name = role_name
        self.span_start = span_start
        self.span_end = span_end

    def __str__(self):
        s = ""
        s = s + 'role_name: ' + self.role_name + '\n'
        s = s + 'role: ' + self.role + '\n'
        s = s + 'start: ' + str(self.span_start) + '\n'
        s = s + 'end: ' + str(self.span_end) + '\n'
        return s


class Event:
    def __init__(
        self,
        event_type,
    ):
        assert isinstance(event_type, str)
        self.event_type = event_type
        self.trigger = None
        self.roles = []

    def set_trigger(self, trigger):
        assert isinstance(trigger, Argument)
        self.trigger = trigger

    def add_role(self, role):
        assert isinstance(role, Argument)
        self.roles.append(role)

    def exist_overlapping(self):
        t_start, t_end = self.trigger.span_start, self.trigger.span_end
        for i in range(len(self.roles)):
            start1, end1 = self.roles[i].span_start, self.roles[i].span_end
            if (t_start <= start1 and start1 <= t_end) or (t_start <= end1 and end1 <= t_end):
                return True
            if (start1 <= t_start and t_start <= end1) or (start1 <= t_end and t_end <= end1):
                return True
            for j in range(len(self.roles)):
                if j == i:
                    continue
                start2, end2 = self.roles[j].span_start, self.roles[j].span_end
                if (start2 <= start1 and start1 <= end2) or (start2 <= end1 and end1 <= end2):
                    return True
                if (start1 <= start2 and start2 <= end1) or (start1 <= end2 and end2 <= end1):
                    return True
        return False

    def overlape_with(self, event):
        self_spans = self.get_all_spans()
        other_spans = event.get_all_spans()
        for i in range(len(self_spans)):
            start1, end1 = self_spans[i][0], self_spans[i][1]
            for j in range(len(other_spans)):
                start2, end2 = other_spans[j][0], other_spans[j][1]
                if (start2 <= start1 and start1 <= end2) or (start2 <= end1 and end1 <= end2):
                    return True
                if (start1 <= start2 and start2 <= end1) or (start1 <= end2 and end2 <= end1):
                    return True
        return False

    def get_all_spans(self):
        spans = [(self.trigger.span_start, self.trigger.span_end)]
        for i in range(len(self.roles)):
            spans.append((self.roles[i].span_start, self.roles[i].span_end))
        return spans

    def get_clean_pariticipants(self):
        t_start, t_end = self.trigger.span_start, self.trigger.span_end
        clean_participants = []
        for i in range(len(self.roles)):
            start1, end1 = self.roles[i].span_start, self.roles[i].span_end
            if (start1 <= t_start) and (t_end <= end1):
                flag = False
                break
            flag = True
            for j in range(len(self.roles)):
                if j == i:
                    continue
                start2, end2 = self.roles[j].span_start, self.roles[j].span_end
                if ((start1 < start2) and (end2 <= end1)) or ((start1 <= start2) and (end2 < end1)):
                    flag = False
                    break
            if flag:
                clean_participants.append((self.roles[i].role_name, self.roles[i].role))
        return clean_participants

    def __str__(self):
        s = ""
        s = s + self.event_type + '\n'
        s = s + 'trigger: ' + self.trigger.role
        s = s + '(' + str(self.trigger.span_start) + ', ' + str(self.trigger.span_end) + ')\n'
        for arg in self.roles:
            s = s + arg.role_name + ': ' + arg.role
            s = s + '(' + str(arg.span_start) + ', ' + str(arg.span_end) + ')\n'
        return s


class Sentence:
    def __init__(
        self,
        seq,
        begin_index=None,
        end_index=None
    ):
        self.tokenizer = BertTokenizer(do_lower_case=ace_args.do_lower_case)
        self.seq = seq
        self.begin_index = begin_index
        self.end_index = end_index

        assert len(seq) == (end_index - begin_index + 1)
        self.tokens, self.token_begin_index, self.token_end_index = \
            self._split_token_with_index(seq, begin_index)
        self.events = []

    def get_span(self, begin_index, end_index):
        span_start = self.token_begin_index.index(begin_index)
        span_end = self.token_end_index.index(end_index)
        return span_start, span_end

    def _split_token_with_index(self, seq, begin_index):
        space_count = []
        idx = 0
        while idx < len(seq):
            start = idx
            while seq[idx] == " " or seq[idx] == "\t":
                idx += 1
            if idx != start:
                space_count.append(idx-start)
            else:
                idx += 1
        space_count.append(0)

        tokens = seq.strip().split()
        assert len(space_count) == len(tokens)

        token_begin_index = [0 for i in range(len(tokens))]

        cur = begin_index
        for idx in range(len(tokens)):
            token_begin_index[idx] = cur
            cur += (space_count[idx] + len(tokens[idx]))

        new_tokens = []
        new_begin_index = []
        new_end_index = []
        for idx in range(len(tokens)):
            tmp_tokens = self.tokenizer.tokenize(tokens[idx])
            if isinstance(self.tokenizer, WordTokenizer):
                tmp_tokens = [token.text for token in tmp_tokens]
            cur = token_begin_index[idx]
            for i in range(len(tmp_tokens)):
                new_tokens.append(tmp_tokens[i])
                new_begin_index.append(cur)
                cur += len(tmp_tokens[i])
                new_end_index.append(cur-1)
        return new_tokens, new_begin_index, new_end_index

    def add_event(self, event):
        assert isinstance(event, Event)
        self.events.append(event)

    def __str__(self):
        return self.seq


class Participants:
    def __init__(self, role):
        self.role = role
        self.values = set()

    def add_value(self, value):
        if value not in self.values:
            self.values.add(value)

    def __str__(self):
        return "; ".join(self.values)


def parseSGML(filename):
    debug = True
    sgm_parser = SGMLParser()

    file_content = open(filename, "r").read()
    sgm_parser.content = ""
    sgm_parser.line_content = []
    sgm_parser.feed(file_content)
    content = sgm_parser.content
    line_content = sgm_parser.line_content
    content = content.replace("\n", " ")
    if filename.find("FLOPPINGACES_20041114.1240.03") >= 0 or \
       filename.find("CNN_CF_20030304.1900.04") >= 0 or \
       filename.find("BACONSREBELLION_20050226.1317") >= 0 or \
       filename.find("CNN_ENG_20030616_130059.25") >= 0 or \
       filename.find("FLOPPINGACES_20050217.1237.014") >= 0:
        content = content.replace("&----", "&")
        line_content = [line.replace("&----", "&") for line in line_content]
    sentences = []

    sent_id = 0
    line_id = 0
    while line_id < len(line_content):
        line = line_content[line_id]
        pre_content = "".join(line_content[:line_id])
        char_st = len(pre_content)

        while line_id < len(line_content)-2 and line_content[line_id+1] == "&----":
            line = line + line_content[line_id+1] + line_content[line_id+2]
            line_id += 2
        line_id += 1
        line = line.replace("\n", " ")
        char_ed = char_st + len(line)
        if debug:
            print("-----------------------------", line_id, (char_st, char_ed))
            print("S-"+line+"-E")

        if len(line.strip()) < 1:
            continue

        sents_in_line = line.strip().split()
        last_end = 0
        for sent in sents_in_line:
            sent = sent.replace("\n", " ").strip()
            sent_st_in_line = line.find(sent, last_end)
            sent_ed_in_line = sent_st_in_line + len(sent) - 1
            last_end = sent_ed_in_line
            sent_st = char_st + sent_st_in_line
            sent_ed = sent_st + len(sent) - 1
            sent_id += 1
            if debug:
                print("------##", sent_id, (sent_st_in_line, sent_ed_in_line), (sent_st, sent_ed))
                print(sent)
            sentences.append(((sent_st, sent_ed), sent))
    for sent_id, (sent_span, sent) in enumerate(sentences[:]):
        print("##", sent_id, sent_span, sent)

    return sentences[3:], content


def extractEvents(
    filename,
    e_types,
):
    xmlTree = ElementTree(file=filename)
    file_sentences = []
    extent_sentence_list = []
    r_participants = {}
    for eventEle in xmlTree.iter(tag="event"):
        extractEvent(
            file_sentences, extent_sentence_list, eventEle, e_types, r_participants, filename)
    return file_sentences, extent_sentence_list, r_participants


def extractEvent(sentence_list, extent_sentence_list, eventEle, e_types, r_participants, filename):
    etype = eventEle.attrib["TYPE"] + '.' + eventEle.attrib["SUBTYPE"]

    if etype not in e_types:
        e_types[etype] = EventType(etype)
    else:
        e_types[etype].plus_one()

    for eventMention in eventEle:
        if eventMention.tag != "event_mention":
            continue

        extentElement = eventMention[0][0]
        extent_seq = re.sub(r"\n", " ", extentElement.text)
        extent_seq_begin_index = int(extentElement.attrib["START"])
        extent_seq_end_index = int(extentElement.attrib["END"])

        sentenceElement = eventMention[1][0]
        seq = re.sub(r"\n", " ", sentenceElement.text)
        seq_begin_index = int(sentenceElement.attrib["START"])
        seq_end_index = int(sentenceElement.attrib["END"])

        if ('a' <= seq[-1] and seq[-1] <= 'z') or ('A' <= seq[-1] and seq[-1] <= 'Z'):
            seq = seq + '.'
            seq_end_index += 1

        extent_sentence = None
        sentence = None

        # =========extent sentence=============
        for idx in range(len(extent_sentence_list)):
            if extent_seq == extent_sentence_list[idx].seq:
                if extent_seq_begin_index == extent_sentence_list[idx].begin_index and extent_seq_end_index == extent_sentence_list[idx].end_index:
                    extent_sentence = extent_sentence_list[idx]
                    break
        if extent_sentence is None:
            extent_sentence = Sentence(
                seq=extent_seq,
                begin_index=extent_seq_begin_index,
                end_index=extent_seq_end_index)
            extent_sentence_list.append(extent_sentence)

        # =========sentence=============
        for idx in range(len(sentence_list)):
            if seq == sentence_list[idx].seq:
                if seq_begin_index == sentence_list[idx].begin_index and seq_end_index == sentence_list[idx].end_index:
                    sentence = sentence_list[idx]
                    break
        if sentence is None:
            sentence = Sentence(
                seq=seq,
                begin_index=seq_begin_index,
                end_index=seq_end_index)
            sentence_list.append(sentence)

        anchorEle = eventMention[2][0]
        anchor_text = re.sub("\n", " ", anchorEle.text)
        anchor_begin_index = int(anchorEle.attrib["START"])
        anchor_end_index = int(anchorEle.attrib["END"])

        # ==========extent sentence=============
        extent_event = Event(etype)
        extent_anchor_span_start, extend_anchor_span_end = extent_sentence.get_span(anchor_begin_index, anchor_end_index)
        extent_trigger = Argument(
            role=anchor_text,
            role_name='trigger',
            span_start=extent_anchor_span_start,
            span_end=extend_anchor_span_end)
        extent_event.set_trigger(extent_trigger)

        # ===========sentence=============
        event = Event(etype)
        anchor_span_start, anchor_span_end = sentence.get_span(anchor_begin_index, anchor_end_index)
        trigger = Argument(
            role=anchor_text,
            role_name="trigger",
            span_start=anchor_span_start,
            span_end=anchor_span_end)
        event.set_trigger(trigger)

        origin_roles = []
        for eventMentionArgument in eventMention:
            if eventMentionArgument.tag != "event_mention_argument":
                continue
            argElement = eventMentionArgument[0][0]

            role_name = eventMentionArgument.attrib["ROLE"]
            role_value = re.sub("\n", " ", argElement.text)
            role_begin_index = int(argElement.attrib["START"])
            role_end_index = int(argElement.attrib["END"])

            is_exist = True
            for role_rec in origin_roles:
                if (role_rec[0] == role_name) and (role_begin_index >= role_rec[1]) and (role_end_index <= role_rec[2]):
                    is_exist = False
                    break
            if not is_exist:
                continue
            else:
                origin_roles.append([role_name, role_begin_index, role_end_index])

            # ==========extent sentence=============
            extent_role_span_start, extent_role_span_end = extent_sentence.get_span(role_begin_index, role_end_index)
            extent_role = Argument(
                role=role_value,
                role_name=role_name,
                span_start=extent_role_span_start,
                span_end=extent_role_span_end)
            extent_event.add_role(extent_role)

            # ===========sentence=============
            role_span_start, role_span_end = sentence.get_span(role_begin_index, role_end_index)
            role = Argument(
                role=role_value,
                role_name=role_name,
                span_start=role_span_start,
                span_end=role_span_end)
            event.add_role(role)

            # role participants
            if role_name not in r_participants:
                r_participants[role_name] = Participants(role_name)

        clean_participants = event.get_clean_pariticipants()
        if clean_participants is None:
            print(event)
        for role_name, role in clean_participants:
            r_participants[role_name].add_value(role)

        # ==========extent sentence=============
        extent_sentence.add_event(extent_event)

        # ==========extent sentence============
        sentence.add_event(event)


def get_pattern_from_instance(instances):
    def get_event_pattern(tokens, event):
        pattern_str = event.event_type + '\t'
        p_list = []
        arguments = [event.trigger] + event.roles
        arguments = sorted(arguments, key=lambda item: item.span_start)

        l_cur = 0
        for argument in arguments:
            span_start, span_end = argument.span_start, argument.span_end
            mask_num = span_start - l_cur
            p_list.extend(['O' for i in range(mask_num)])
            p_list.extend([argument.role_name for i in range(span_start, span_end+1)])
            l_cur = span_end + 1
        if l_cur < len(tokens):
            p_list.extend(['O' for i in range(len(tokens) - l_cur)])
        assert len(tokens) == len(p_list)
        pattern_str = pattern_str + " ".join(tokens) + '\t'
        pattern_str = pattern_str + " ".join(p_list)
        return pattern_str

    patterns = []
    for idx, instance in enumerate(instances):
        tokens = instance.tokens
        events = instance.events
        if len(tokens) < 5:
            continue
        flag = True
        for event in events:
            if event.exist_overlapping():
                flag = False
                break
        for i in range(len(events)):
            event = events[i]
            for j in range(i+1, len(events)):
                another_event = events[j]
                if event.overlape_with(another_event):
                    flag = False
                    break
        if flag:
            patterns.append(instance)
            # if len(event.roles) == 0:
            #     continue
            # pattern = get_event_pattern(tokens, event)
            # if pattern is not None:
            #     patterns.append(pattern)
    return patterns


def serialize_event(event):
    s = ""
    trigger = event.trigger
    s = s + str(trigger.span_start) + ' ' + str(trigger.span_end) + ' ' + str(event.event_type)
    if len(event.roles) != 0:
        s = s + " ; "
        for idx, role in enumerate(event.roles):
            s = s + str(role.span_start) + ' ' + str(role.span_end) + ' ' + str(role.role_name)
            if idx != len(event.roles) - 1:
                s = s + " ; "
    return s


def serialize_sentence(sentence, id):
    s = str(id) + '\t'
    s = s + " ".join(sentence.tokens)
    s = s + '\t'
    for idx, event in enumerate(sentence.events):
        event_str = serialize_event(event)
        s = s + event_str
        if idx != len(sentence.events) - 1:
            s = s + '\t'
    s = s + '\n'
    return s


def store_sentence_to_file(sentences, file_path):
    with open(file_path, 'w') as f:
        for id, sentence in enumerate(sentences):
            sentence_string = serialize_sentence(sentence, id)
            f.write(sentence_string)


if __name__ == '__main__':
    event_types = {}
    role_participants = {}

    train_sentences = []
    val_sentences = []

    # ===========extent sentences===========
    extent_sentences = []

    dir_list = os.listdir(ace_args.ace_corpus)
    nw_file_count = 0
    for dir_name in dir_list:
        for sublist in ['timex2norm']:  # ['adj', 'fp1', 'fp2', 'timex2norm']:
            dir_name_new = os.path.join(
                ace_args.ace_corpus,
                dir_name,
                sublist)
            file_list = os.listdir(dir_name_new)
            random.shuffle(file_list)
            for file_name in file_list:
                if file_name.endswith('apf.xml'):
                    # sentence_in_doc, content = parseSGML(os.path.join(dir_name_new, file_name))
                    # print(sentence_in_doc)
                    # raise NotImplementedError
                    file_sentences, file_extent_sentences, r_participants = extractEvents(
                        os.path.join(dir_name_new, file_name),
                        event_types)
                    if dir_name == 'nw' and nw_file_count < 40:
                        val_sentences.extend(file_sentences)
                        nw_file_count += 1
                    else:
                        extent_sentences.extend(file_extent_sentences)
                        train_sentences.extend(file_sentences)
                        for key in r_participants:
                            if key not in role_participants:
                                role_participants[key] = Participants(key)
                            for value in r_participants[key].values:
                                role_participants[key].add_value(value)

    # 计算每种事件类型下，角色的出现次数
    event_role_appear_num = {}
    for s in train_sentences+val_sentences:
        for e in s.events:
            et = e.event_type
            if et not in event_role_appear_num:
                event_role_appear_num[et] = {}
            for r in e.roles:
                rt = r.role_name
                if rt not in event_role_appear_num[et]:
                    event_role_appear_num[et][rt] = 0
                event_role_appear_num[et][rt] += 1

    event_role_ratio = {}
    for et in event_role_appear_num:
        er = event_role_appear_num[et]
        tot = sum(er.values())
        event_role_ratio[et] = {}
        for rt in er:
            event_role_ratio[et][rt] = er[rt] / tot
        print(et+":")
        for rt in event_role_ratio[et]:
            print('\t' + rt + ": " + str(event_role_ratio[et][rt]))
        print()
    raise NotImplementedError

    # 按条件查找事件
    # for sentence in train_sentences:
    #     if len(sentence.events) != 1:
    #         continue
    #     event = sentence.events[0]
    #     if len(event.roles) < 2 or len(event.roles) > 3:
    #         continue
    #
    #     if len(sentence.seq.split()) > 15:
    #         continue
    #     print(sentence)
    #     print()
    # raise NotImplementedError

    # 测试触发词是否有重叠的现象
    for sentence in train_sentences+val_sentences:
        for event in sentence.events:
            role_list = event.roles

            is_overlapping = False
            for idi in range(len(role_list)):
                for idj in range(idi+1, len(role_list)):
                    t1 = role_list[idi]
                    t2 = role_list[idj]

                    value1 = t1.span_start - t2.span_start
                    value2 = t1.span_start - t2.span_end
                    value3 = t1.span_end - t2.span_start
                    value4 = t1.span_end - t2.span_end

                    if value1 > 0 and value2 > 0 and value3 > 0 and value4 > 0:
                        continue
                    elif value1 < 0 and value2 < 0 and value3 < 0 and value4 < 0:
                        continue
                    else:
                        is_overlapping = True
                        break
            if is_overlapping:
                print(sentence)
    raise NotImplementedError

    # ================为每个事件，角色，触发词分配id==================
    if not os.path.exists(ace_args.events_id_file):
        with open(ace_args.events_id_file, 'w') as f:
            for idx, et in enumerate(event_types):
                s = str(idx) + '\t' + et + '\t' + str(event_types[et].count) + '\n'
                f.write(s)

    if not os.path.exists(ace_args.roles_id_file):
        with open(ace_args.roles_id_file, 'w') as f:
            for idx, role in enumerate(role_participants):
                s = str(idx) + '\t' + role + '\t' + str(len(role_participants[role].values)) + '\n'
                f.write(s)

    # ================extractor的训练集和测试集==================
    store_sentence_to_file(train_sentences, ace_args.train_file)
    store_sentence_to_file(val_sentences, ace_args.test_file)

    print(len(train_sentences), len(val_sentences))
    print(len(role_participants))

    # ================extent句子，只包括训练集的内容==================
    # store_sentence_to_file(extent_sentences, './data/events.extent')
    # 获取pattern并存储
    # pattern_strs = get_pattern_from_instance(extent_sentences)

    # pattern_strs = get_pattern_from_instance(train_sentences)
    # with open(ace_args.pattern_file, 'w') as f:
    #     for idx, pattern in enumerate(pattern_strs):
    #         if idx != len(pattern_strs) - 1:
    #             f.write(str(idx)+'\t'+pattern+'\n')
    #         else:
    #             f.write(str(idx)+'\t'+pattern)
    pattern_strs = get_pattern_from_instance(train_sentences)
    store_sentence_to_file(pattern_strs, ace_args.pattern_file)

    tokenizer = BertTokenizer(do_lower_case=ace_args.do_lower_case)
    role_dir_path = ace_args.roles_dir
    for role_name in role_participants:
        role_file_path = os.path.join(role_dir_path, role_name)
        with open(role_file_path, 'w') as f:
            participant = role_participants[role_name]
            for idx, value in enumerate(participant.values):
                tokens = tokenizer.tokenize(value)
                if idx != len(participant.values)-1:
                    f.write(" ".join(tokens)+'\n')
                else:
                    f.write(" ".join(tokens))
