import os
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding  
from transformers import Trainer, TrainingArguments
from peft import get_peft_model
from peft import LoraConfig, TaskType
from datasets import load_dataset

from load import get_tokenized_dataset
# from evaluate import compute_metrics
from config import Config


import torch
import numpy as np


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# export HF_ENDPOINT=https://hf-mirror.com

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")


peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)


model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

if torch.cuda.is_available():
    model = model.cuda()





def train(model):
    import evaluate
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        try:  
            predictions = np.argmax(logits, axis=-1)  
        except ValueError as e:  
            print(f"ValueError: {e}")  
            # 查看不一致问题的批次  
            for i, logit in enumerate(logits):  
                print(f"Logit {i} shape: {logit.shape}")  
            raise  
        return metric.compute(predictions=predictions, references=labels)


    dataset = load_dataset("carlosejimenez/seq2seq-mrpc")

    # tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

    tokenized_train_datasets, tokenizer = get_tokenized_dataset(dataset['train'])
    tokenized_test_datasets, tokenizer = get_tokenized_dataset(dataset['test'])


    # DataCollatorWithPadding会自动调整输入的长度，通过在最短的序列上填充  
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)  


    # Confirm shapes  
    # train_shape = tokenized_datasets["train"][0]['input_ids'].shape  
    # print("train_shape = ", train_shape)
    # assert len(train_shape) == 2, f"Unexpected shape: {train_shape}" 

    # 训练流程
    training_args = TrainingArguments(
        output_dir=Config['train_output_dir'],
        learning_rate=1e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_test_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


    model.save_pretrained(Config['model_output_dir'])




def inference():
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    import torch

    model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    model = model.to("cuda")
    model.eval()
    inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])








if __name__ == '__main__':
    train(model)