import json
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

total_data_list = []

with open(f'datasets/train_3.jsonl', 'r') as json_file:
    for line in json_file:
        data = json.loads(line)
        total_data_list.append(data)

prompt = '''
당신은 전문 건축업자 입니다. 다음 질의응답을 보고, 각 평가항목에 대한 점수를 매겨주세요.

평가 항목은 다음과 같습니다.
- 답변의 정확성(0~5점): 답변이 질문에 대해 정확한 정보를 제공하고 있는가?
- 답변의 완성도(0~5점): 답변이 충분한 정보를 제공하고 있는가?
- 답변의 자연스러움(0~5점): 답변이 자연스럽게 표현되어 있는가?

ex)
### 입력
Q: PD에 면한 벽체 결로와 벽장 결로는 어떻게 다르나요?
A: PD에 면한 벽체 결로는 주로 단열이 부족하거나 습기가 차단되지 않아 벽체 온도가 낮아서 발생하는 반면, 벽장 결로는 환기가 부족하여 습기가 많이 차서 발생합니다. PD면 벽체는 단열과 방습에 주의하고, 벽장은 환기를 잘 시키는 것이 중요해요.

### 출력
- 답변의 정확성(0~5점): 4
- 답변의 완성도(0~5점): 5
- 답변의 자연스러움(0~5점): 4
'''

for i in range(0, len(total_data_list)):
    print(f"start {i}th question")

    Q = total_data_list[i]['input']
    A = total_data_list[i]['output']
    category = total_data_list[i]['category']

    text = f'''
### 입력
Q: {Q}
A: {A}

### 출력'''

    retry = 0
    while retry < 5:
        try:
            result = model.generate_content(f"{prompt}{text}")
            print(result.text)
            accuracy_score = result.text.split("- 답변의 정확성(0~5점): ")[1].split("\n- 답변의 완성도(0~5점): ")[0]
            completeness_score = result.text.split("\n- 답변의 완성도(0~5점): ")[1].split("\n- 답변의 자연스러움(0~5점): ")[0]
            naturalness_score = result.text.split("\n- 답변의 자연스러움(0~5점): ")[1]
            average_score = (int(accuracy_score) + int(completeness_score) + int(naturalness_score)) / 3

            print(f"답변의 정확성: {accuracy_score}, 답변의 완성도: {completeness_score}, 답변의 자연스러움: {naturalness_score}, 평균 점수: {average_score}")

            output_dict = {
                "input": Q,
                "output": A,
                "category": category,
                "average_score": f"{average_score:.2f}",
            }
            print(output_dict)

            with open(f'datasets/train_4.jsonl', 'a') as json_file:
                json_file.write(json.dumps(output_dict, ensure_ascii=False))
                json_file.write('\n')
                
            break
        except:
            print(f"retry {retry}th")
            retry += 1

    if retry == 5:
        output_dict = {
            "input": Q,
            "output": A,
            "category": category,
            "average_score": "3.00",
        }
        with open(f'datasets/train_4.jsonl', 'a') as json_file:
            json_file.write(json.dumps(output_dict, ensure_ascii=False))
            json_file.write('\n')