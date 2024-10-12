import os


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'



'''
PEFT支持:
    LoRA、Prefix Tuning、P-Tuning、
    Prompt Tuning、AdaLoRA、Adaption Prompt

共六种对大模型高效调参的方法，分别对应框架六个核心方法类即:
    LoraModel、PrefixEncoder、PromptEncoder、
    PromptEmbedding、AdaLoraModel、AdaptionPromptModel，

每个核心方法对应配置文件类为:
    LoraConfig、PrefixTuningConfig、PromptEncoderConfig、
    PromptTuningConfig、AdaLoraConfig、AdaptionPromptConfig

'''


'''
PEFT框架当前支持五种下游子任务，即:
    PeftModelForSequenceClassification、PeftModelForSeq2SeqLM、
    PeftModelForCausalLM、PeftModelForTokenClassification、
    PeftModelForQuestionAnswering，
    
这些子任务类就是借助PEFT框架形成的最终的peft_model，用于后续训练与推理。

'''



def main():
    pass





if __name__ == '__main__':
    main()