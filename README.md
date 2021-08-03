# PLMEE
用于百度事件抽取比赛, 采用数据集为DuEE 1.0 (数据集链接: https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true), 下载后放入自建的data文件夹<br>

参考论文: Exploring Pre-trained Language Models for Event Extraction and Generation.

## 功能应用
- 支持原版PLMEE实验 (命令参数 `istrigger` 需要设置为`True`)
- 支持 事件类型 + 文本 作为输入的论元抽取 (命令参数 `istrigger` 设置为`False`, 根据命令参数`isETid` 指定 `ettext + text` 的训练/预测方式 或者 `etid + text` 的训练/预测)

## 所需环境
- allennlp=0.9.0 
- torch=1.4.0

## data目录文件结构
```text
data/DuEE
├── events.id
├── roles.id
├── test.json
├── train.json
└── dev.json
```
- 其中 events.id 与 roles.id 通过 data_convert.py转换而来(该脚本以DuEE数据集为示例)

## src 文件夹各项文件功能说明
- `cfg.py`: 用于设置和存储trainer中使用的各项参数
- `XXreader.py`: 针对于各种数据集的数据读取
- `extractor_trainer.py`: 用于模型训练的主要运行文件
- `extractor_model.py`: PLMEE的主要模型结构
- `extractor_metric.py`: 用于模型评估, 采用ACE05的评估方式
- `extractor_tester.py`: 用于模型测试, 尚未完全实现
- `predict.py`: 用于模型的预测推理环节, 通过制定待预测文件以及训练好的模型来输出对应的结果文件 
  
## 训练(train)命令
```sh
python src/Extractor_trainer.py --pretrained_bert /XXX/chinese_roberta_wwm_large_ext --bert_vocab /XXX/chinese_roberta_wwm_large_ext/vocab.txt --do_train_trigger --data_meta_dir ./data/DuEE --extractor_origin_trigger_dir ./save/DuEE/bert_large/trigger --extractor_origin_role_dir ./save/DuEE/bert_large/role --extractor_epoc 20 --extractor_batch_size 12 --extractor_train_file ./data/DuEE/train.json --extractor_val_file ./data/DuEE/dev.json
```

## 预测(predict)命令
```sh
python src/predict.py --pretrained_bert XXX/chinese_roberta_wwm_ext --bert_vocab XXX/chinese_roberta_wwm_ext/vocab.txt --extractor_batch_size 16 --data_meta_dir ./data/DuEE --extractor_train_file ./data/DuEE/train.json --extractor_val_file ./data/DuEE/dev.json --extractor_test_file ./data/DuEE/dev.json --task_name DuEE --save_trigger_dir ./save/DuEE/bert_base/trigger/model_state_epoch_19.th --save_role_dir ./save/DuEE/bert_base/role/model_state_epoch_19.th
```


## 比赛结果 DuEE test1 baseline

| |pre|recall|f1|
| :-: | :-: | :-: | :-: |
|roberta-base|84.97|81.84|83.38|
|roberta-large|86.34|82.41|84.33|