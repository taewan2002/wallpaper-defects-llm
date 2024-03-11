import json
import google.generativeai as genai
import random
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-1.0-pro-latest')

# 각 카테고리의 데이터가 골고루 증강될 수 있도록 리스트를 구성합니다.
category_list = ['건축구조', '기타', '마감재', '마감하자', '시공', '인테리어', '타 마감하자']

## Step 1: Generate new Q&A with gemini-pro
def generate_qna(Q1, A1, Q2, A2):
    prompt = '''
당신은 전문 건축업자 입니다. 다음 두 질의응답을 바탕으로 하나의 질문과 답변을 만들어 주세요.

다음과 같은 사항을 고려해주세요.
- 답변의 길이는 100자에서 300자 사이어야 한다.
- 질문자가 궁금해하는 것을 해결할 수 있어야 한다.
- 사실에 근거하여 자세히 답변해야 한다.
- 답변은 존댓말을 사용하는 한국어로 이루어져야 한다.
- 두 질문을 이어붙여도 된다.

ex) 
### 입력
Q1: 창호 결로를 해결하기 위한 가장 효과적인 방법은 무엇인가요?
A1: 창호 결로를 예방하기 위한 대책은 KS에 규정된 프레임을 선택하고, 열관류율이 적은 유리를 선택하는 것이 좋습니다. 또한, 창호 결로 발생 시험을 실시하여 안전한 제품을 선택하는 것이 중요합니다. 또한, 주기적인 창문 청소와 관리를 통해 습기가 축적되는 것을 방지하는 것이 도움이 될 수 있습니다.

Q2: AD, PD에 면한 벽체 결로에 대한 대책은 어떤 것이 있나요?
A2: AD, PD에 면한 벽체 결로에 대한 대책은 단열재를 미실하게 시공하여 결로가 생기는 벽체의 표면 온도를 노점온도 이상으로만 유지해주는 것이 중요합니다. 그 외에도 실외습기의 유입을 차단하고, 적절한 환기 시스템을 활용하여 공간 내 습도를 유지하고 벽체 표면을 건조하게 유지하는 것이 필요합니다. 또한, 외피의 방수 및 수증기 차단 기능을 강화하여 벽체 내부로의 습기 유입을 최소화해야 합니다. 일반적으로는 벽체의 표면뿐만 아니라 내부적으로도 겉지문 및 방수층의 철저한 시공이 필요합니다.

### 출력
Q: 창호 결로와 면한 결로는 어떤 차이가 있나요?
A: 창문에 물이 맺히는 것과 벽에 물이 맺히는 것은 다릅니다. 창문에 물이 맺히는 것은 따뜻한 공기가 차가운 창문을 만나서 생기는데, 이를 막으려면 따뜻한 공기가 창문을 통해 쉽게 나가지 않도록 좋은 창문을 고르는 것이 중요해요. 벽에 물이 맺히는 것은 벽이 충분히 따뜻하지 못하거나, 집안의 습한 공기가 밖으로 잘 나가지 못해서 생깁니다. 이를 해결하기 위해서는 벽을 따뜻하게 유지하고, 집안의 공기가 잘 통하도록 만들어야 해요. 이 두 가지 문제 모두 집을 짓거나 관리할 때 올바른 자재를 고르고, 잘 짓는 것이 중요합니다.
'''

    text = f'''
### 입력
Q1: {Q1}
A1: {A1}

Q2: {Q2}
A2: {A2}

### 출력'''

    retry = 0
    while retry < 3:
        try:
            result = model.generate_content(f"{prompt}{text}")
            input = result.text.split("Q: ")[1].split("\nA: ")[0]
            output = result.text.split("\nA: ")[1]
            break
        except:
            retry += 1
    
    return input, output

## Step 2: Check answer length
def check_length(A):
    return len(A) < 100 or len(A) > 300


## Step 3: Check average score with gemini-pro
def check_score(Q, A):
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

    text = f'''
### 입력
Q: {Q}
A: {A}

### 출력'''

    retry = 0
    while retry < 5:
        try:
            result = model.generate_content(f"{prompt}{text}")
            accuracy_score = result.text.split("- 답변의 정확성(0~5점): ")[1].split("\n- 답변의 완성도(0~5점): ")[0]
            completeness_score = result.text.split("\n- 답변의 완성도(0~5점): ")[1].split("\n- 답변의 자연스러움(0~5점): ")[0]
            naturalness_score = result.text.split("\n- 답변의 자연스러움(0~5점): ")[1]
            average_score = (int(accuracy_score) + int(completeness_score) + int(naturalness_score)) / 3
            return average_score
        except:
            retry += 1

    return 3.0


def main():
    for i in range(350):

        for category in category_list:
            print(f"start data augmentation")

            category_set = {}
            total_data_list = []

            # 새로 생성된 데이터도 포함하여 복원 추출하기 위해 다시 데이터를 읽어옵니다.
            with open(f'datasets/train_6.jsonl', 'r') as json_file:
                line_num = 0
                for line in json_file:
                    data = json.loads(line)
                    total_data_list.append(data)
                    if data['category'] not in category_set:
                        category_set[data['category']] = []
                    else:
                        category_set[data['category']].append(data)
                    line_num += 1
            
            # 각 카테고리에서 랜덤으로 2개의 질문을 가져옵니다.
            random_question = random.sample(category_set[category], 2)

            Q1 = random_question[0]['input']
            A1 = random_question[0]['output']
            Q2 = random_question[1]['input']
            A2 = random_question[1]['output']
            
            ## Step 1: Generate new Q&A with gemini-pro
            try:
                Q3, A3 = generate_qna(Q1, A1, Q2, A2)
            except:
                print("Generate new Q&A fail")
                continue

            ## Step 2: Check answer length
            if check_length(A3):
                print("Check answer length fail")
                continue

            ## Step 3: Check average score with gemini-pro
            average_score = check_score(Q3, A3)
            if float(average_score) < 3:
                print("Check average socore fail")
                continue

            output_dict = {
                'input': Q3,
                'output': A3, 
                'category': category,
                'average_score': f"{average_score:.2f}"
            }

            with open(f'datasets/train_6.jsonl', 'a') as json_file:
                json_file.write(json.dumps(output_dict, ensure_ascii=False))
                json_file.write('\n')

            print("save new data")

main()