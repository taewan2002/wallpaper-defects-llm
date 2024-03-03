from datasets import load_dataset
from transformers import ElectraTokenizer, ElectraForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# 데이터셋 로드
data = load_dataset("taewan2002/wallpaper-defects-qna", split="train")

# 데이터 전처리 함수
def preprocess_data(examples):
    inputs = ["Q: " + q + " A: " + a for q, a in zip(examples['input'], examples['output'])]
    labels = examples['average_score']
    return {'input_texts': inputs, 'labels': labels}

# 데이터셋에 전처리 함수 적용
processed_data = data.map(preprocess_data)

# 토크나이저 및 모델 설정
tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = ElectraForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", num_labels=6)  # 점수는 1부터 5까지이므로, 0 포함 총 6개 레이블

# 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples['input_texts'], padding="max_length", truncation=True)

# 데이터셋 토큰화
tokenized_data = processed_data.map(tokenize_function, batched=True)

# 라벨 준비
labels = processed_data['labels']

# StratifiedKFold를 사용하여 데이터셋 분할
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train_idx, eval_idx = next(skf.split(np.zeros(len(labels)), labels))

# 훈련 및 평가 데이터셋 선택
train_dataset = tokenized_data.select(train_idx)
eval_dataset = tokenized_data.select(eval_idx)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="taewan2002/wallpaper-deffects-qna-reward",
    num_train_epochs=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # 성능이 개선되지 않는 최대 에포크 횟수
)

# 평가 메트릭 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

# Trainer 설정 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

trainer.train()