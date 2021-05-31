# PLMEE
用于百度事件抽取比赛, 采用数据集为DuEE 1.0 (数据集链接: https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true), 下载后放入自建的data文件夹<br>

参考论文: Exploring Pre-trained Language Models for Event Extraction and Generation.

## 所需环境
- allennlp=0.9.0 
- torch=1.4.0
  
## 训练(train)命令
```sh
python src/Extractor_trainer.py --pretrained_bert /XXX/chinese_roberta_wwm_large_ext --bert_vocab /XXX/chinese_roberta_wwm_large_ext/vocab.txt --do_train_trigger True --data_meta_dir ./data/DuEE --extractor_origin_trigger_dir ./save/DuEE/bert_large/trigger --extractor_origin_role_dir ./save/DuEE/bert_large/role --extractor_epoc 20 --extractor_batch_size 12 --extractor_train_file ./data/DuEE/train.json --extractor_val_file ./data/DuEE/dev.json
```

## 预测(predict)命令
```sh
python src/predict.py --pretrained_bert XXX/chinese_roberta_wwm_ext --bert_vocab XXX/chinese_roberta_wwm_ext/vocab.txt --extractor_batch_size 16 --data_meta_dir ./data/DuEE --extractor_train_file ./data/DuEE/train.json --extractor_val_file ./data/DuEE/dev.json --extractor_test_file ./data/DuEE/dev.json --mode DuEE --bert_mode bert_base --save_trigger_dir ./save/DuEE/bert_base/trigger/model_state_epoch_19.th --save_role_dir ./save/DuEE/bert_base/role/model_state_epoch_19.th
```

## 比赛结果 DuEE test1 baseline

| |pre|recall|f1|
| :-: | :-: | :-: | :-: |
|roberta-base|84.97|81.84|83.38|
|roberta-large|86.34|82.41|84.33|