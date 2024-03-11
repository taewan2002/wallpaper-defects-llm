from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


modelPath = "distiluse-base-multilingual-cased-v1"
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = CSVLoader(file_path='datasets/train_data.csv',encoding='utf-8')
data = loader.load()

db = FAISS.from_documents(data, embedding=embeddings)
db.save_local("faiss_index")

db = FAISS.load_local("faiss_index", embeddings)

retriever = db.as_retriever(search_kwargs={"k": 4})

model_id = "taewan2002/gemma-qlora-wallpaper-deffects-qna/checkpoint-1781"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32,
)
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=300, 
    no_repeat_ngram_size=3)
hf = HuggingFacePipeline(pipeline=pipe)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 당신은 전문 건축업자입니다. 다음 질문에 대해 답변해 주세요.
template = """마지막에 질문에 답하려면 다음과 같은 맥락을 사용합니다.\n{context}\n링크와 태그와 같이 추가 정보 없이 답변만 진행해주세요. 당신은 전문 건축업자입니다. 다음 질문에 대해 답변해 주세요.\n\n### 질문: {question}\n### 답변: """
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | hf
    | StrOutputParser()
)

version = 2

dataset_dir = "datasets/test.csv"
output_dir = f"outputs/rag_test_{version}.jsonl"

# CSV 파일 읽기
df = pd.read_csv(dataset_dir)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # 결과를 저장할 리스트 초기화
    results_list = []

    # RAG 체인 스트림을 통해 결과 생성 및 리스트에 저장
    for chunk in rag_chain.stream(row['질문']):
        results_list.append(chunk)

    # 결과 저장
    result = {
        "id": row['id'],
        "input": row['질문'],
        "output": results_list[0]
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
sub.to_csv(f'outputs/rag_submission_{version}.csv', index = False)
