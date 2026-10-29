import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide"
)

# 데이터 URL 및 저장 경로 설정
DATA_URL = "https://raw.githubusercontent.com/jsy1026/data/refs/heads/main/stroke.csv"
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "user_notes.json")


# JSON 저장 및 불러오기 함수
def load_user_notes():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_notes(notes):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=4)


# 저장된 기록 불러오기
saved_data = load_user_notes()


# 데이터 로드 (캐싱 적용)
@st.cache_data
def get_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df = get_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 앱 제목 및 소개
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("본 실습실은 뇌졸중 예측 데이터를 확인하고 구조를 분석하는 공간입니다.")

st.divider()

# 상단 숫자 카드 4개
total_count = len(df)
col_count = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = (
    (stroke_count / total_count) * 100 if total_count > 0 else 0.0
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(label="전체 사람 수", value=f"{total_count:,}명")
m2.metric(label="열 개수", value=f"{col_count}개")
m3.metric(label="stroke가 1인 사람 수", value=f"{stroke_count:,}명")
m4.metric(label="stroke 비율", value=f"{stroke_ratio:.2f}%")

st.divider()

# 데이터 구조 및 열 설명 표
st.subheader("데이터 구조 및 열 설명")
st.caption(
    "교재를 참고하여 각 열의 우리말 뜻을 입력한 후 저장 버튼을 누르세요."
)

columns_info = []
saved_meanings = saved_data.get("meanings", {})

for col in df.columns:
    dtype_str = str(df[col].dtype)
    null_count = int(df[col].isnull().sum())
    meaning = saved_meanings.get(col, "")
    columns_info.append(
        {
            "열 이름": col,
            "우리말 뜻": meaning,
            "값의 종류": dtype_str,
            "빈 값 개수": null_count,
        }
    )

info_df = pd.DataFrame(columns_info)

edited_info_df = st.data_editor(
    info_df,
    hide_index=True,
    column_config={
        "열 이름": st.column_config.TextColumn(disabled=True),
        "우리말 뜻": st.column_config.TextColumn(
            placeholder="교재를 보고 우리말 뜻을 입력하세요"
        ),
        "값의 종류": st.column_config.TextColumn(disabled=True),
        "빈 값 개수": st.column_config.NumberColumn(disabled=True),
    },
    use_container_width=True,
)

st.divider()

# 데이터 미리보기 (처음 5줄)
st.subheader("데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 데이터 출처 입력 영역
st.subheader("데이터 출처")
default_source = saved_data.get("source", "")
source_input = st.text_area(
    "교재를 참고하여 데이터 출처 내용을 작성하세요:",
    value=default_source,
    placeholder="여기에 데이터 출처를 입력하세요.",
    height=100,
)

# 저장 버튼
if st.button("내용 저장하기", type="primary"):
    meanings_to_save = dict(
        zip(edited_info_df["열 이름"], edited_info_df["우리말 뜻"])
    )
    data_to_save = {"meanings": meanings_to_save, "source": source_input}
    save_user_notes(data_to_save)
    st.success("내_기록/user_notes.json 파일에 정상적으로 저장되었습니다.")
