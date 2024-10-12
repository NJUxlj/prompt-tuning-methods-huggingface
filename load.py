from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import Config



import torch


def get_tokenized_dataset(dataset: Dataset):

    tokenizer = AutoTokenizer.from_pretrained(Config['tokenizer_path'])  
    def tokenize_function(example):
        '''
        别直接用，请根据你数据集的字段名进行修改
        '''
        return tokenizer(example['text'], example['label'], padding = "max_length", truncation=True)

    
    # def convert_label_to_int(example):
    #     label_to_int = {"not_equivalent": 1, "equivalent": 0, "UNLABELED":2} 

    #     # 这里注意，数据集的每一行始终是一个嵌套列表 [[xxxxxx],[xx]]
    #     # 其中，[xxxxxx] 是 input_ids
    #     # 因此label必须多加一维
    #     example['label'] = torch.LongTensor([label_to_int[example['label']] ])
    #     return example  
    
    # dataset = dataset.map(convert_label_to_int)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset, tokenizer







if __name__ == '__main__':
    dataset = load_dataset("carlosejimenez/seq2seq-mrpc")

    # tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

    tokenized_train_dataset = get_tokenized_dataset(dataset['train'])


    print(tokenized_train_dataset['train'][1])