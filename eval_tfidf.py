import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import pandas as pd

def normalize_text(text):
    """문자열의 공백 및 특수문자를 제거하고 소문자로 변환"""
    text = re.sub(r'\W+', ' ', text)
    return text.strip().lower()

def is_correct(predicted, actual, similarity_threshold=100):
    """
    predicted와 actual 문자열을 fuzzy matching하여 유사도가 
    similarity_threshold 이상이면 True 반환
    """
    predicted = normalize_text(predicted)
    actual = normalize_text(actual)
    similarity_score = fuzz.partial_ratio(predicted, actual)
    return similarity_score >= similarity_threshold

def evaluate_faq_bot_tfidf(model_path, test_pairs, k_values=[1, 3, 5], similarity_threshold=100):
    """
    FAQ 챗봇의 성능을 평가합니다 (TF-IDF 기반, 예측 FAQ 질문 vs. 원본 FAQ 질문 비교).

    모델 파일은 TF-IDF vectorizer, TF-IDF 행렬, 그리고 FAQ 질문이 포함된 DataFrame을
    pickle 파일에 저장되어 있어야 합니다.
    
    Args:
        model_path (str): 저장된 TF-IDF 모델 파일 경로 (pickle 파일).
        test_pairs (list of tuple): (original_question, similar_question) 형태의 테스트 데이터.
            - original_question: FAQ에 등록된 정답 질문(ground truth)
            - similar_question: 사용자가 입력한 질문(변형)
        k_values (list): Top-K 평가를 위한 K 값 리스트.
        similarity_threshold (int): fuzzy matching 유사도 임계값 (0~100).

    Returns:
        dict: MRR 및 Top-K 정확도 결과.
    """
    # TF-IDF 모델 로드 (vectorizer, tfidf_matrix, 그리고 FAQ 질문이 담긴 DataFrame)
    with open(model_path, "rb") as f:
        loaded_data = pickle.load(f)
    vectorizer = loaded_data["vectorizer"]
    tfidf_matrix = loaded_data["tfidf_matrix"]
    df = loaded_data["df"]  # DataFrame에 FAQ 질문이 저장되어 있다고 가정 (컬럼명: "question")
    questions = df["question"].tolist()  # FAQ 질문 리스트로 변환

    print("✅ 저장된 TF-IDF 모델을 로드했습니다.")

    ranks = []
    top_k_accuracies = {k: 0 for k in k_values}

    for original_question, similar_question in test_pairs:
        # 사용자가 입력한 질문(변형)을 벡터화
        query_vector = vectorizer.transform([similar_question])
        # TF-IDF 행렬과 코사인 유사도 계산
        cosine_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # 코사인 유사도 내림차순으로 정렬하여 Top-k 인덱스 추출
        sorted_indices = np.argsort(cosine_scores)[::-1]
        max_k = max(k_values)
        top_indices = sorted_indices[:max_k]
        
        # 예측된 FAQ 질문 리스트 (상위 max_k 개)
        predicted_questions = [questions[idx] for idx in top_indices]
        
        # 디버깅용 출력
        print(f"\n🔍 User Query (similar_question): {similar_question}")
        for idx, score in zip(top_indices, cosine_scores[top_indices]):
            print(f"🔹 Retrieved FAQ Question: {questions[idx]} (코사인 유사도: {score:.4f})")
        
        # Rank 계산: 예측된 FAQ 질문과 실제 원본 질문 간의 fuzzy matching 평가
        rank = next((i + 1 for i, pred_q in enumerate(predicted_questions) 
                     if is_correct(pred_q, original_question, similarity_threshold)), None)
        ranks.append(rank if rank else float('inf'))
        
        # Top-K 정확도 계산
        for k in k_values:
            if any(is_correct(pred_q, original_question, similarity_threshold) 
                   for pred_q in predicted_questions[:k]):
                top_k_accuracies[k] += 1

    # MRR 계산
    mrr = sum(1 / rank for rank in ranks if rank != float('inf')) / len(ranks)
    
    # Top-K 정확도 비율 계산
    for k in k_values:
        top_k_accuracies[k] = top_k_accuracies[k] / len(test_pairs)
    
    results = {"MRR": mrr}
    results.update({f"Top-{k} Accuracy": acc for k, acc in top_k_accuracies.items()})
    return results

def load_test_pairs(file_path):
    """
    CSV 파일에서 테스트 페어를 로드합니다.
    CSV 파일은 'original_question' (FAQ 원본 질문)와 
    'similar_question' (사용자 변형 질문) 열을 포함해야 합니다.
    """
    try:
        df = pd.read_csv(file_path, delimiter=",", quotechar='"', encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, delimiter=",", quotechar='"', encoding="ISO-8859-1", on_bad_lines="skip")
    
    if 'original_question' not in df.columns or 'similar_question' not in df.columns:
        raise ValueError("CSV 파일에 'original_question' 및 'similar_question' 열이 필요합니다.")
    
    test_pairs = list(zip(df['original_question'], df['similar_question']))
    return test_pairs

# 사용 예시
if __name__ == "__main__":
    test_pairs = load_test_pairs("Generated_Question_Pairs.csv")
    model_path = "tfidf_chatbot_model.pkl"  # TF-IDF 모델 파일 경로 (DataFrame, vectorizer, tfidf_matrix 포함)
    results = evaluate_faq_bot_tfidf(model_path, test_pairs)
    
    print("\n=== FAQ Chatbot TF-IDF Performance ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print("유사도 기반 평가 (예측 질문 vs. 원본 질문)")
