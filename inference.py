from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

version = 3

tokenizer = AutoTokenizer.from_pretrained("models/checkpoint-107")
model = AutoModelForCausalLM.from_pretrained(
    "models/checkpoint-107",
    device_map="auto",
    torch_dtype=torch.float32,
)

instruction = '''당신은 전문 건축업자입니다. 다음 질문에 대해 답변해 주세요. '''

dataset_dir = "datasets/test.csv"
output_dir = f"outputs/test_{version}.jsonl"

# CSV 파일 읽기
df = pd.read_csv(dataset_dir)

for index, row in tqdm(df.iterrows()):
    # 입력 텍스트 구성
    text = f"### 질문: {instruction}{row['질문']}\n\n### 답변: "
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # 모델을 사용해 출력 생성
    outputs = model.generate(**inputs, max_new_tokens=300)
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

    print(f"generate {row['id']} complate")


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