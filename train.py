import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import pickle

# 데이터 로드
file_path = "microbiology_qna.csv"
df = pd.read_csv(file_path)

# 필요한 컬럼 선택
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Sentence-BERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')  # 가벼운 모델 사용

# 질문 임베딩 생성
question_embeddings = model.encode(questions, convert_to_tensor=True)

# 모델 및 데이터 저장
with open("qna_model.pkl", "wb") as f:
    pickle.dump((questions, answers, question_embeddings), f)

print("모델 학습 및 저장 완료!")
