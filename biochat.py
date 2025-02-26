import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# QnA 모델 및 데이터 로드
with open("qna_model.pkl", "rb") as f:
    questions, answers, question_embeddings = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# 세션 상태 정의
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "similar_indices" not in st.session_state:
    st.session_state.similar_indices = []
if "update_state" not in st.session_state:  # 추가: update_state 초기화
    st.session_state.update_state = False

def add_message(role, text):
    st.session_state.chat_history.append((role, text))

# 번역 함수 정의
def ko_to_en(text_ko: str) -> str:
    if not text_ko:
        return ""
    # 한국어 -> 영어 번역
    translated = GoogleTranslator(source='ko', target='en').translate(text_ko)
    return translated

def en_to_ko(text_en: str) -> str:
    if not text_en:
        return ""
    # 영어 -> 한국어 번역
    translated = GoogleTranslator(source='en', target='ko').translate(text_en)
    return translated

# Streamlit UI 시작
st.title("자연과학 Q&A 챗봇")
st.text("자연과학에 대해 궁금한 내용을 키워드로 입력해보세요!")
# 사용자 입력 받기
user_question_ko = st.chat_input("질문을 입력하세요.")

if user_question_ko:
    # (A) 한국어 → 영어 번역 (deep-translator)
    user_question_en = ko_to_en(user_question_ko)

    # 사용자 메시지 출력 (즉시 채팅 화면에 표시)
    add_message("사용자", user_question_ko)

    # (B) 모델 유사도 계산
    user_embedding = model.encode(user_question_en, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]

    # (C) 상위 5개의 유사한 질문 인덱스 추출
    top_5_similar_indices = torch.topk(similarities, 5).indices.tolist()
    st.session_state.similar_indices = top_5_similar_indices

# -- (D) 채팅 기록 출력 (한국어)
for role, text in st.session_state.chat_history:
    if role == "사용자":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 5px 0;'>
                <div style='background-color: #FFEB3B; padding: 10px; border-radius: 10px; max-width: 70%;'>
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:  # "챗봇"
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; margin: 5px 0;'>
                <div style='background-color: #F1F1F1; padding: 10px; border-radius: 10px; max-width: 70%;'>
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -- (E) 추천 질문 버튼 출력
if st.session_state.similar_indices:
    st.subheader("💡 입력하신 내용과 가장 유사한 질문을 찾았습니다.")
    for idx in st.session_state.similar_indices:
        question_en = questions[idx]
        question_ko = en_to_ko(question_en)  # 영어 질문을 한국어로 번역

        # 사용자가 클릭할 수 있도록 버튼 생성
        def on_click_recommend(qidx=idx):
            recommended_q_en = questions[qidx]
            rec_embedding = model.encode(recommended_q_en, convert_to_tensor=True)
            rec_sims = util.pytorch_cos_sim(rec_embedding, question_embeddings)[0]
            rec_best_match_idx = torch.argmax(rec_sims).item()
            st.session_state.best_match_idx = rec_best_match_idx

            answer_en = answers[rec_best_match_idx]
            recommended_q_ko = en_to_ko(recommended_q_en)  # 추천 질문을 한국어로 번역
            answer_ko = en_to_ko(answer_en)

            add_message("사용자", recommended_q_ko)
            add_message("챗봇", answer_ko)

            st.session_state.similar_indices = torch.topk(rec_sims, 5).indices.tolist()  # 새로 추천된 질문으로 갱신

            # 상태 변경 후 다시 렌더링하기 위해 session_state 값을 업데이트
            st.session_state.update_state = not st.session_state.update_state  # 상태를 반전시켜 페이지 갱신을 유도

        # 각 추천 질문에 대해 버튼 생성
        st.button(question_ko, key=f"recommend_{idx}", on_click=on_click_recommend)