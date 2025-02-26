import torch
import pickle
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from fuzzywuzzy import fuzz
import re


def normalize_text(text):
    """문자열의 공백 및 특수문자를 제거하고 소문자로 변환"""
    text = re.sub(r'\W+', ' ', text)  # 모든 특수문자를 공백으로 대체
    return text.strip().lower()


def is_correct(predicted, actual, similarity_threshold=100):
    """문자열이 일정 수준 이상 유사하면 True 반환"""
    predicted = normalize_text(predicted)
    actual = normalize_text(actual)

    # 문자열 유사도 계산 (fuzz.partial_ratio 사용)
    similarity_score = fuzz.partial_ratio(predicted, actual)
    return similarity_score >= similarity_threshold


def evaluate_faq_bot(model_path, test_pairs, k_values=[1, 3, 5]):
    """
    FAQ 챗봇의 성능을 평가합니다.
    여기서는 예측된 FAQ 질문과 실제 원본 질문의 유사성을 기준으로 평가합니다.
    
    test_pairs는 (original_question, similar_question) 형태로,
    - original_question: FAQ에 등록된 정답 질문 (ground truth)
    - similar_question: 유저가 입력한 질문(원본 질문과 유사한 변형)
    """
    # 모델 및 FAQ 데이터(질문, 답변, 질문 임베딩) 로드
    with open(model_path, "rb") as f:
        questions, answers, question_embeddings = pickle.load(f)

    # 임베딩 생성 모델 로드
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 평가 메트릭 초기화
    ranks = []
    top_k_accuracies = {k: 0 for k in k_values}

    # 테스트 루프: 각 테스트 페어에서 유저 질문(similar_question)을 이용해 FAQ 질문을 검색
    for original_question, similar_question in test_pairs:
        # 유저 질문(변형된 질문)을 임베딩으로 변환
        query_embedding = model.encode(similar_question, convert_to_tensor=True)

        # FAQ 질문 임베딩과 코사인 유사도 계산
        cosine_scores = util.cos_sim(query_embedding, question_embeddings)[0]

        # 가장 높은 유사도를 보이는 Top-k FAQ 질문 인덱스 반환
        top_results = torch.topk(cosine_scores, k=max(k_values))

        # 예측된 FAQ 질문 리스트 (questions 리스트에서 가져옴)
        predicted_questions = [questions[idx] for idx in top_results.indices.tolist()]

        # 디버깅용 출력
        print(f"\n🔍 User Query (similar_question): {similar_question}")
        for idx, score in zip(top_results.indices.tolist(), top_results.values.tolist()):
            print(f"🔹 Retrieved FAQ Question: {questions[idx]} (유사도: {score:.4f})")

        # Rank 계산: 예측된 FAQ 질문과 실제 원본 질문(original_question)의 유사성을 기준으로 함
        rank = next((i + 1 for i, pred_q in enumerate(predicted_questions) if is_correct(pred_q, original_question)), None)
        ranks.append(rank if rank else float('inf'))

        # Top-K 정확도 계산
        for k in k_values:
            if any(is_correct(pred_q, original_question) for pred_q in predicted_questions[:k]):
                top_k_accuracies[k] += 1

    # MRR (Mean Reciprocal Rank) 계산
    mrr = sum(1 / rank for rank in ranks if rank != float('inf')) / len(ranks)

    # Top-K 정확도 비율 계산
    for k in k_values:
        top_k_accuracies[k] = top_k_accuracies[k] / len(test_pairs)

    # 결과 반환
    results = {"MRR": mrr}
    results.update({f"Top-{k} Accuracy": acc for k, acc in top_k_accuracies.items()})
    return results


def load_test_pairs(file_path):
    """CSV 파일에서 테스트 페어를 로드하고 전처리 수행
       CSV 파일은 'original_question' (원본 FAQ 질문)와 'similar_question' (유저 질문 변형) 열을 포함해야 합니다.
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
    # CSV 파일에서 테스트 데이터 로드
    test_pairs = load_test_pairs("Generated_Question_Pairs.csv")

    # FAQ 모델 경로 지정
    model_path = "qna_model.pkl"

    # 평가 실행
    results = evaluate_faq_bot(model_path, test_pairs)

    # 결과 출력
    print("\n=== FAQ Chatbot Performance ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print("유사도 기반 정확도 평가 (예측 질문 vs. 원본 질문)")
