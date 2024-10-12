from transformers import AutoModelForSeq2SeqLM, LoraConfig
from peft import LoraModel, LoraConfig
from config import Config




os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'


# step1. Lora配置
config = LoraConfig(peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.01)
# step2. 预训练模型加载
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
# step3. 显示生成Lora模型
lora_model = LoraModel(config, model)





import datasets
from transformers import Trainer, DataCollatorForSeq2Seq

if resume_from_checkpoint:
    lora_weight = torch.load(ckpt_name)
    set_peft_model_state_dict(model, lora_weight)


dataset_path = Config['train_data_path']
train_data = datasets.load_from_disk(dataset_path)

class ModifiedTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        # 改写trainer的save_model，在checkpoint的时候只存lora权重
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

trainer = ModifiedTrainer(
    model=model,
    train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
            num_train_epochs=10,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=10,
            save_steps=200,
            output_dir=output_dir
        ),
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
trainer.train()
model.save_pretrained(train_args.output_dir)