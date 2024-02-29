import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import transformers
import warnings
import wandb
import os
import gc
import json
from datasets import load_dataset
from torch import cuda

# 오류 로그 표시 안하도록 설정
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 토크나이저 병렬처리 방지(오류 방지)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true' # __cell__ 오류 방지

# base_mdoel 및 학습 파일 경로 설정
base_model = "beomi/OPEN-SOLAR-KO-10.7B" #"beomi/OPEN-SOLAR-KO-10.7B" # google/gemma-7b
file_name = 'datasets/labeled_train.jsonl'

# QLoRA 모델을 사용하기 위한 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float32,
    device_map={"":0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token # 패딩 토큰을 문장의 끝으로 설정 </s>
tokenizer.padding_side = "right" # 패딩을 문장 뒤에 추가

# 데이터셋 로드
data = load_dataset('json', data_files=file_name, split="train")

# 지시문 설정
instruction = '''당신은 전문 건축업자입니다. 다음 질문에 대해 답변해 주세요. '''

# 데이터셋에 text 맵핑
data = data.map(
    lambda x: {'text': f"### 질문: {instruction}{x['input']}\n\n### 답변: {x['output']}"}
)

# 데이터 분할 test:eval=90:10
split_data = data.train_test_split(test_size=0.1) # 10%를 검증셋으로 사용
train_set = split_data['train']
eval_set = split_data['test']

# 데이터셋 토큰화
train_set = train_set.map(lambda samples: tokenizer(samples["text"], padding=True, truncation=True, return_tensors="pt"), batched=True)
eval_set = eval_set.map(lambda samples: tokenizer(samples["text"], padding=True, truncation=True, return_tensors="pt"), batched=True)

# lora 파라미터 설정
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 모델에서 어뎁터 분리
model = get_peft_model(model, peft_params)

# prameter
epochs = 10
batch_size = 4
accumulation_steps = 16
optimizer = "paged_adamw_32bit"
lr = 3e-4

class ClearCacheCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if cuda.is_available():
            cuda.empty_cache()
            self.clear_accelerator()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if cuda.is_available():
            cuda.empty_cache()
            self.clear_accelerator()

    def on_step_begin(self, args, state, control, **kwargs):
        if cuda.is_available():
            cuda.empty_cache()
            self.clear_accelerator()

    def on_init_end(self, args, state, control, **kwargs):
        if cuda.is_available():
            cuda.empty_cache()
            self.clear_accelerator()

    def on_log(self, args, state, control, **kwargs):
        if cuda.is_available():
            cuda.empty_cache()
            self.clear_accelerator()

    def clear_accelerator(self):
        gc.collect()
    
            
training_params = TrainingArguments(
    output_dir="models",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=accumulation_steps,
    optim=optimizer,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=lr,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
    dataloader_num_workers=8,
)

trainer = Trainer(
    model=model,
    args=training_params,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[ClearCacheCallback()],
)

if cuda.is_available():
    trainer.accelerator.clear()
    cuda.empty_cache()
    gc.collect()
    
trainer.train()