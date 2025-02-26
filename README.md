# 🤖 Natural-Science-QnA-Chatbot

## 🌍 프로젝트 개요
자연과학 Q&A 챗봇은 사용자가 입력한 자연과학 관련 질문에 대해 AI가 자주 묻는 질문을 추천해주고, 답변을 찾아 제공하는 시스템입니다. Sentence-BERT 및 TF-IDF 모델을 활용하여 kaggle의 microbiology qna를 기반으로 질문의 의미를 파악하고 최적의 답변을 찾아줍니다.

## 🚀 주요 기능
- 자연과학 관련 질문에 대한 AI 기반 자동 응답 (FAQ)
- 한국어 및 영어 질문 지원 (Google Translator 활용)
- Sentence-BERT 및 TF-IDF 기반 질문 임베딩 및 유사도 계산
- Streamlit 기반 웹 인터페이스 제공
- 유사 질문 추천 기능
- 챗봇 성능 평가 (MRR 및 Top-K Accuracy 측정)

## 🛠 사용된 기술
- **프론트엔드**: Streamlit
- **백엔드**: Python
- **모델**: Sentence-BERT (all-MiniLM-L6-v2), TF-IDF
- **라이브러리**:
  - `sentence-transformers`
  - `torch`
  - `scikit-learn`
  - `deep-translator`
  - `fuzzywuzzy`
  - `pandas`
  - `numpy`

## 📊 사용한 데이터셋
[Kaggle의 microbiology qna 데이터셋](https://www.kaggle.com/datasets/moonstone34/microbiology-qna/code)

## 💻 설치 및 실행 방법
### 1. 모델 학습 및 저장
```bash
python train.py
```

### 2. 챗봇 실행
```bash
streamlit run biochat.py
```

### 3. 챗봇 성능 평가
#### 🏆 SBERT 모델 평가
```bash
python eval_sbert.py
```

#### 📊 TF-IDF 모델 평가
```bash
python eval_tfidf.py
```

## 📂 파일 구조
```
├── biochat.py          # Streamlit 기반 챗봇 UI
├── train.py            # Sentence-BERT 모델 학습 및 저장
├── eval_sbert.py       # SBERT 기반 챗봇 성능 평가
├── eval_tfidf.py       # TF-IDF 기반 챗봇 성능 평가
├── tfidf_train.ipynb   # TF-IDF 학습 데이터 전처리 및 모델 저장
├── qna_model.pkl       # 학습된 QA 데이터 저장 파일
└── README.md           # 프로젝트 문서
```

## 📈 유사도 기반 성능 평가 (예측 질문 vs. 원본 질문)
- **Sentence-BERT 기반 챗봇 (한글 질문:번역된 데이터)**:
  - MRR: 0.9273
  - Top-1 Accuracy: 0.9100
  - Top-3 Accuracy: 0.9400
  - Top-5 Accuracy: 0.9600

- **TF-IDF 기반 챗봇 (한글 질문:번역된 데이터)**:
  - MRR: 0.7603
  - Top-1 Accuracy: 0.6800
  - Top-3 Accuracy: 0.8400
  - Top-5 Accuracy: 0.8700

SBERT 모델이 의미 기반 검색이 가능하여 TF-IDF 모델 대비 우수한 성능을 보였습니다.

## 👥 기여자 및 기여 내용
- [@seohyeon123](https://github.com/seohyeon123)  
  - SBERT, TF-IDF 추론 기능 구현  
  - 영문 성능 평가  
- [@hayoomee1214](https://github.com/hayoomee1214)  
  - SBERT 임베딩 저장 및 추론 기능 구현
  - Front-end 개발 담당
- [@DefJamNoJam](https://github.com/DefJamNoJam)
  - TF-IDF 임베딩 저장 및 추론 기능 구현   
  - 번역 기능 개발 및 한글 성능 평가
- 윤주완
  - Documentation 작성
  -  Front-end 개발 지원

## 🔥 향후 개선 사항
- QnA 데이터셋 확장 및 지속적인 업데이트
- 없는 질문에 대한 답변 생성 기능 추가 (ChatGPT를 결합한 모델 생성)
- UI/UX 개선을 통한 사용자 경험 향상
