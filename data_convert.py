# 将DUEE数据文件改成PLMEE读取的格式
import codecs
import json
import tqdm
import os
from allennlp.data.tokenizers import Token

current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前目录文件夹
EventExp_dir = os.path.dirname(current_dir) # 获取当前文件的父目录

train_path = "data/DuEE-Fin/basepre/train.json"
test_path = "data/DuEE-Fin/basepre/dev.json"
path_list = [train_path, test_path]

event_dict = {} # 所有的事件类型
role_dict = {} # 所有的role类型
for filename in path_list:
    with codecs.open(filename, 'r', 'UTF-8') as rf:
        for line in rf.readlines():
            line = json.loads(line)

            # print(line)
            # words = line['text']
            # tokens = [Token(word) for word in words]
            # print(tokens)

            for event in line['event_list']:
                event_type = event['event_type']
                if event_type in event_dict: event_dict[event_type] += 1
                else: event_dict[event_type] = 1

                for argument in event['arguments']:
                    role = argument['role']
                    if role in role_dict: role_dict[role] += 1
                    else: role_dict[role] = 1



# 将event_dict 与 role_dict 处理为event.id、role.id 文件
with codecs.open("PLMEE/data/DuEE-Fin/events.id", 'w', 'UTF-8') as wf:
    for i, (et, value) in enumerate(event_dict.items()):
        wf.write(str(i) + "\t" + et + "\t" + str(value) + "\n")
del role_dict['环节']
with codecs.open("PLMEE/data/DuEE-Fin/roles.id", 'w', 'UTF-8') as wf:
    for i, (rl, value) in enumerate(role_dict.items()):
        wf.write(str(i) + "\t" + rl + "\t" + str(value) + "\n")



    