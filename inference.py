from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

version = 20
model_path = "taewan2002/gemma-qlora-wallpaper-deffects-qna/checkpoint-1781"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float32,
)

instruction = '''링크를 포함하지 마세요. 당신은 전문 건축업자입니다. 다음 질문에 대해 답변해 주세요.'''

dataset_dir = "datasets/test.csv"
output_dir = f"outputs/test_{version}.jsonl"

# CSV 파일 읽기
df = pd.read_csv(dataset_dir)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    text = f"{instruction}\n\n### 질문: {row['질문']}\n### 답변: "
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=300, 
        no_repeat_ngram_size=3,
        top_k=1,
        top_p=0.9,
        num_beams=2, 
        early_stopping=True,
        repetition_penalty=1.2,
        do_sample=True,
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("### 답변: ")[1]

    # 결과 저장
    result = {
        "id": row['id'],
        "input": row['질문'],
        "output": answer
    }
    
    with open(output_dir, 'a') as outfile:
        outfile.write(json.dumps(result, ensure_ascii=False))
        outfile.write("\n")


output_list = []
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
with open(output_dir, 'r') as json_file:
    for line in json_file:
        output = json.loads(line)
        output_list.append(output["output"])

submission = pd.read_csv('datasets/sample_submission.csv')
id = submission.id
submission.drop(['id'], axis= 1, inplace =True)

submission = submission.astype(float)

for i in range(len(submission)):
    try:
        submission.loc[i] = model.encode(output_list[i]).tolist()
    except:
        print(i)

total_submission = pd.DataFrame({'id': id})
sub = pd.concat([total_submission, submission] , axis = 1)
sub.to_csv(f'outputs/submission_{version}.csv', index = False)