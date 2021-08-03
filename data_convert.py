# 数据文件预处理 (以DUEE数据为例)
import codecs
import json
import tqdm
import os

# 1、根据 train 与 dev 构建 events.id (事件类型统计) 与 roles.id(角色类型统计)
train_path = "data/DUEE/train.json"
test_path = "data/DUEE/dev.json"
output_path = "data/DUEE/"

path_list = [train_path, test_path]

event_dict = {} # 所有的事件类型
role_dict = {} # 所有的role类型
for filename in path_list:
    with codecs.open(filename, 'r', 'UTF-8') as rf:
        for line in rf.readlines():
            line = json.loads(line)

            for event in line['event_list']:
                event_type = event['event_type']
                if event_type in event_dict: event_dict[event_type] += 1
                else: event_dict[event_type] = 1

                for argument in event['arguments']:
                    role = argument['role']
                    if role in role_dict: role_dict[role] += 1
                    else: role_dict[role] = 1



# 将event_dict 与 role_dict 处理为event.id、role.id 文件
with codecs.open(output_path + "events.id", 'w', 'UTF-8') as wf:
    for i, (et, value) in enumerate(event_dict.items()):
        wf.write(str(i) + "\t" + et + "\t" + str(value) + "\n")

with codecs.open(output_path + "roles.id", 'w', 'UTF-8') as wf:
    for i, (rl, value) in enumerate(role_dict.items()):
        wf.write(str(i) + "\t" + rl + "\t" + str(value) + "\n")


# 2、根据schema文件构建events.id 与 roles.id




    