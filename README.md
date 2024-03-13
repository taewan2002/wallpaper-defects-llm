# wallpaper-defects-llm
![](https://cdn-images-1.medium.com/max/1600/1*0yIbHSCUX6joc_2NsXb0jg.png)
도배 하자 질의 응답 처리 : 한솔데코 시즌2 AI 경진대회

## 대회 개요
NLP(자연어 처리) 기반의 QA (질문-응답) 시스템을 통해 도배하자와 관련된 깊이 있는 질의응답 처리 능력을 갖춘 AI 모델 개발에 도전합니다.

## 코드 실행 방법
### 학습
학습된 모델은 [taewan2002/gemma-qlora-wallpaper-deffects-qna](https://huggingface.co/taewan2002/gemma-qlora-wallpaper-deffects-qna)에 업로드 돼 있습니다. adapter파일을 다운로드 하여 학습 없이 실행할 수 있습니다. 루트 경로에 datasets 디렉토리를 생성하고 그 안에 제공된 파일들을 넣어줘야 합니다.

1. 리포지토리를 클론합니다.

2. 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

3. hf에 로그인합니다.(데이터 다운로드를 위해 필요합니다.)
```bash
huggingface-cli login
```

4. wandb를 로그인 해줍니다.
```bash
wandb login
```

5. 학습을 시작합니다.
```bash
python train.py
```

### 추론
1. 학습된 모델의 경로를 `inference.py`에 입력합니다. 튜닝된 모델이 기본값으로 입력되어 있습니다.
```bash
model_path = "taewan2002/gemma-qlora-wallpaper-deffects-qna/checkpoint-1781"
```

2. 추론을 시작합니다.
```bash
python inference.py
```

## 전략 및 과정 설명
![](https://github.com/taewan2002/wallpaper-defects-llm/assets/89565530/b2e7519d-5865-42f9-8389-02240672f420)

데이터 전처리의 핵심 전략은 다음과 같습니다.
![](https://github.com/taewan2002/wallpaper-defects-llm/assets/89565530/8bddf339-6498-447c-8350-ab4f9cb09773)

더 자세한 내용은 [여기](https://taewan2002.medium.com/%EB%8F%84%EB%B0%B0-%ED%95%98%EC%9E%90-q-a-%EC%B1%97%EB%B4%87-%EB%A7%8C%EB%93%A4%EA%B8%B0-a836a7392c50)에서 확인할 수 있습니다.
