from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, TrainerCallback, Trainer
from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import transformers
from torch import cuda
import os
import gc
import warnings
import wandb

# 오류 로그 표시 안하도록 설정
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 토크나이저 병렬처리 방지(오류 방지)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true' # __cell__ 오류 방지

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="taewan2002/wallpaper-defects-qna",
        metadata={"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    num_train_epochs: int = field(default=8, metadata={"help": "How many optimizer update epoch to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    logging_steps: int = field(default=1, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="taewan2002/solar-qlora-wallpaper-deffects-qna",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the GG model - this is the local one, update it to the one on the Hub
model_id = "beomi/OPEN-SOLAR-KO-10.7B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
    torch_dtype=torch.float32,
    attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2",
    device_map={"":0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id # 패딩 토큰을 문장의 끝으로 설정 </s>
tokenizer.padding_side = "right" # 패딩을 문장 뒤에 추가

lora_config = LoraConfig(
    r=script_args.lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout
)

model = get_peft_model(model, lora_config)

# StratifiedKFold 인스턴스 생성
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 데이터셋 로드 및 DataFrame으로 변환
data = load_dataset(path=script_args.dataset_name, split="train")
df = pd.DataFrame(data)

# 지시문 설정 및 text 맵핑
instruction = '''당신은 전문 건축업자입니다. 다음 질문에 대해 답변해 주세요. '''
df['text'] = df.apply(lambda x: f"{instruction}\n### 질문: {x['input']}\n\n### 답변: {x['output']}", axis=1)

train_index, test_index = next(skf.split(df, df['category']))

# 훈련 및 검증 데이터셋 구성
train_df = df.iloc[train_index]
eval_df = df.iloc[test_index]

# 데이터셋을 다시 Hugging Face의 Dataset 형태로 변환
train_set = Dataset.from_pandas(train_df)
eval_set = Dataset.from_pandas(eval_df)

# 데이터셋 토큰화
train_set = train_set.map(lambda samples: tokenizer(samples["text"], padding=True, truncation=True, return_tensors="pt"), batched=True)
eval_set = eval_set.map(lambda samples: tokenizer(samples["text"], padding=True, truncation=True, return_tensors="pt"), batched=True)

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
    
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_train_epochs,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=-1,
    warmup_ratio=script_args.warmup_ratio,
    lr_scheduler_type=script_args.lr_scheduler_type,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    report_to="wandb",
    group_by_length=True,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
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