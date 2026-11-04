import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 페이지 설정
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실", page_icon="📝", layout="wide"
)

# 데이터 URL 설정
DATA_URL = "https://raw.githubusercontent.com/hanseongchan9-ops/data/refs/heads/main/stroke.csv"


# 데이터 로드 및 전처리 함수 (분류 모델 페이지와 동일한 방식)
@st.cache_data
def get_model_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df = df.dropna(subset=["bmi"])  # 결측치 제거 예시 통일

    # 범주형 변수 원핫 인코딩 등 동일한 전처리 수행
    df_processed = pd.get_dummies(
        df,
        columns=[
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ],
        drop_first=True,
    )

    X = df_processed.drop(columns=["id", "stroke"])
    y = df_processed["stroke"]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return df, X_train, X_test, y_train, y_test, test_idx


try:
    df_raw, X_train, X_test, y_train, y_test, test_idx = get_model_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 모델 학습
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)


# 혼동행렬 지표 계산 함수 (행: 실제, 열: 예측)
def calculate_confusion_matrix_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # 재현율 분모: 실제 뇌졸중인 사람 수 (tp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 정밀도 분모: 뇌졸중이라 예측한 사람 수 (tp + fp)
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precision_str = f"{precision:.4f}"
        f1_str = f"{f1:.4f}"
    else:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
        precision = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "recall": recall,
        "precision_val": precision,
        "precision_str": precision_str,
        "f1_str": f1_str,
    }


lr_metrics = calculate_confusion_matrix_metrics(y_test, lr_pred)
dt_metrics = calculate_confusion_matrix_metrics(y_test, dt_pred)

# 다수 클래스 기준 모델 (훈련용에서 많은 쪽으로만 답하는 모델) 예측
majority_class = y_train.mode()[0]
baseline_pred = np.full_like(y_test, majority_class)
base_metrics = calculate_confusion_matrix_metrics(y_test, baseline_pred)

# 페이지 제목
st.title("📝 분류 모델 채점표 (혼동행렬 및 성능 평가)")
st.markdown(
    "테스트 데이터를 바탕으로 모델의 성능을 혼동행렬과 평가지표를 통해 다각도로 채점하는 공간입니다."
)

st.divider()

# 테스트 데이터 정보 표시
test_total_count = len(y_test)
test_stroke_count = int(y_test.sum())
st.markdown(
    f"**테스트용 사람 수:** {test_total_count}명 | **그중 뇌졸중인 사람 수:** {test_stroke_count}명"
)

st.divider()

# 1. 혼동행렬 2x2 표 표시 (각 모델별)
st.subheader("1. 모델별 혼동행렬 (2×2 표)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 로지스틱 회귀(확률로 답하는 모델)")
    lr_cm_df = pd.DataFrame(
        [
            [
                f"TN: {lr_metrics['tn']}",
                f"FP: {lr_metrics['fp']}",
                f"합계: {lr_metrics['tn'] + lr_metrics['fp']}",
            ],
            [
                f"FN: {lr_metrics['fn']}",
                f"TP: {lr_metrics['tp']}",
                f"합계: {lr_metrics['fn'] + lr_metrics['tp']}",
            ],
            [
                f"합계: {lr_metrics['tn'] + lr_metrics['fn']}",
                f"합계: {lr_metrics['fp'] + lr_metrics['tp']}",
                f"전체: {test_total_count}",
            ],
        ],
        index=["실제 정상 (0)", "실제 뇌졸중 (1)", "합계"],
        columns=["예측 정상 (0)", "예측 뇌졸중 (1)", "합계"],
    )
    st.dataframe(lr_cm_df, use_container_width=True)

with col2:
    st.markdown("##### 의사결정트리(질문으로 답하는 모델)")
    dt_cm_df = pd.DataFrame(
        [
            [
                f"TN: {dt_metrics['tn']}",
                f"FP: {dt_metrics['fp']}",
                f"합계: {dt_metrics['tn'] + dt_metrics['fp']}",
            ],
            [
                f"FN: {dt_metrics['fn']}",
                f"TP: {dt_metrics['tp']}",
                f"합계: {dt_metrics['fn'] + dt_metrics['tp']}",
            ],
            [
                f"합계: {dt_metrics['tn'] + dt_metrics['fn']}",
                f"합계: {dt_metrics['fp'] + dt_metrics['tp']}",
                f"전체: {test_total_count}",
            ],
        ],
        index=["실제 정상 (0)", "실제 뇌졸중 (1)", "합계"],
        columns=["예측 정상 (0)", "예측 뇌졸중 (1)", "합계"],
    )
    st.dataframe(dt_cm_df, use_container_width=True)

st.divider()

# 2. 성능 지표 비교 표 (정확도, 재현율, 정밀도, F1) + 기준 모델 포함
st.subheader("2. 모델별 성능 지표 비교 표")

summary_table = pd.DataFrame(
    [
        {
            "모델명": "로지스틱 회귀(확률로 답하는 모델)",
            "정확도": f"{lr_metrics['accuracy']:.4f}",
            "재현율": f"{lr_metrics['recall']:.4f}",
            "정밀도": lr_metrics["precision_str"],
            "F1 스코어": lr_metrics["f1_str"],
        },
        {
            "모델명": "의사결정트리(질문으로 답하는 모델)",
            "정확도": f"{dt_metrics['accuracy']:.4f}",
            "재현율": f"{dt_metrics['recall']:.4f}",
            "정밀도": dt_metrics["precision_str"],
            "F1 스코어": dt_metrics["f1_str"],
        },
        {
            "모델명": "훈련용에서 많은 쪽으로만 답하는 모델",
            "정확도": f"{base_metrics['accuracy']:.4f}",
            "재현율": f"{base_metrics['recall']:.4f}",
            "정밀도": base_metrics["precision_str"],
            "F1 스코어": base_metrics["f1_str"],
        },
    ]
)
st.dataframe(summary_table.set_index("모델명"), use_container_width=True)

st.divider()

# 3. 오류 분석 목록 (놓친 환자 및 헛짚은 사람)
st.subheader("3. 오류 분석 세부 목록")

# 모델 선택 UI
selected_model_name = st.selectbox(
    "목록을 확인할 모델을 선택하세요:",
    [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
    ],
)

chosen_pred = lr_pred if "로지스틱" in selected_model_name else dt_pred
chosen_metrics = lr_metrics if "로지스틱" in selected_model_name else dt_metrics

# 데이터프레임 매핑을 위해 테스트셋 원본 데이터 추출
test_df_subset = df_raw.loc[test_idx].copy()
test_df_subset["y_true"] = y_test.values
test_df_subset["y_pred"] = chosen_pred

# 왼쪽: 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자 = FN)
fn_df = test_df_subset[
    (test_df_subset["y_true"] == 1) & (test_df_subset["y_pred"] == 0)
].copy()
fn_df = fn_df.sort_values(by="age", ascending=False)

# 오른쪽: 아닌데 뇌졸중이라고 예측한 사람 (헛짚은 사람 = FP)
fp_df = test_df_subset[
    (test_df_subset["y_true"] == 0) & (test_df_subset["y_pred"] == 1)
].copy()
fp_df = fp_df.sort_values(by="age", ascending=False)

col_left, col_right = st.columns(2)

with col_left:
    fn_total_len = len(fn_df)
    st.markdown(
        f"##### 📉 놓친 환자 (실제 1, 예측 0) [전체 인원수: {fn_total_len}명]"
    )
    display_fn = fn_df.head(20).reset_index(drop=True)
    display_fn.index = display_fn.index + 1  # 1부터 번호 시작
    display_fn_show = display_fn[
        ["age", "avg_glucose_level", "hypertension", "heart_disease"]
    ].rename(
        columns={
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )
    st.dataframe(display_fn_show, use_container_width=True)

with col_right:
    fp_total_len = len(fp_df)
    st.markdown(
        f"##### 📈 헛짚은 사람 (실제 0, 예측 1) [전체 인원수: {fp_total_len}명]"
    )
    display_fp = fp_df.head(20).reset_index(drop=True)
    display_fp.index = display_fp.index + 1  # 1부터 번호 시작
    display_fp_show = display_fp[
        ["age", "avg_glucose_level", "hypertension", "heart_disease"]
    ].rename(
        columns={
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )
    st.dataframe(display_fp_show, use_container_width=True)

# 혼동행렬의 FN, FP 칸과 일치 여부 확인 문구
st.info(
    f"확인: 위 목록의 인원 수(놓친 환자 {fn_total_len}명, 헛짚은 사람 {fp_total_len}명)는 선택한 모델 혼동행렬의 FN({chosen_metrics['fn']}) 및 FP({chosen_metrics['fp']}) 칸의 사람 수와 정확히 일치합니다."
)
