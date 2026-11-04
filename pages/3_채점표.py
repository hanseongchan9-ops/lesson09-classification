import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 페이지 설정
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실", page_icon="📈", layout="wide"
)

# 데이터 URL 설정
DATA_URL = "https://raw.githubusercontent.com/hanseongchan9-ops/data/refs/heads/main/stroke.csv"

# 속성 한글 매핑 딕셔너리
feature_mapping = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
reverse_feature_mapping = {v: k for k, v in feature_mapping.items()}


# 데이터 로드 및 전처리 함수 (채점표 및 분류 모델 페이지와 동일한 방식)
@st.cache_data
def get_performance_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 범주형 변수 원핫 인코딩 등 전처리 수행
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

    X_all = df_processed.drop(columns=["id", "stroke"])
    y_all = df_processed["stroke"]

    # 동일한 번호순 묶음 분할 규칙 적용 가능하도록 처리
    df_sorted = df.sort_values(by="id").reset_index(drop=True)
    df_processed_sorted = df_processed.sort_values(by="id").reset_index(
        drop=True
    )

    train_indices = []
    test_indices = []

    for i in range(0, len(df_sorted), 10):
        chunk = df_sorted.iloc[i : i + 10]
        if len(chunk) >= 3:
            test_indices.extend(chunk.index[:3])
            train_indices.extend(chunk.index[3:])
        else:
            test_indices.extend(chunk.index[: min(3, len(chunk))])
            if len(chunk) > 3:
                train_indices.extend(chunk.index[3:])

    return df, df_processed_sorted, train_indices, test_indices


try:
    df_raw, df_processed, train_indices, test_indices = get_performance_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 페이지 제목
st.title("📈 뇌졸중 예측 모델 성능 높이기 (클래스 가중치 조정)")
st.markdown(
    "데이터 불균형 문제를 해결하기 위해 가중치를 맞추기 전과 맞춘 후의 모델 성능 변화를 비교하는 공간입니다."
)

st.divider()

# 1. 입력 속성 선택 UI (분류 모델 페이지와 동일, 기본값: age, avg_glucose_level, hypertension, heart_disease)
st.subheader("1. 입력 속성 선택")
default_features_kor = ["나이", "평균 혈당", "고혈압", "심장병"]
all_features_kor = list(feature_mapping.values())

selected_features_kor = st.multiselect(
    "학습에 사용할 속성을 골라주세요 (2개 이상 선택):",
    options=all_features_kor,
    default=default_features_kor,
)

if len(selected_features_kor) < 2:
    st.warning("⚠️ 속성을 **2개 이상** 선택해 주세요.")
    st.stop()

selected_features = [
    reverse_feature_mapping[feat] for feat in selected_features_kor
]

st.divider()

# 데이터 세트 구성
train_df_raw = df_raw.loc[train_indices].copy()
test_df_raw = df_raw.loc[test_indices].copy()

# bmi 결측치 훈련용 중앙값으로 채우기
bmi_median = train_df_raw["bmi"].median()
train_df_raw["bmi"] = train_df_raw["bmi"].fillna(bmi_median)
test_df_raw["bmi"] = test_df_raw["bmi"].fillna(bmi_median)

# 모델 학습용 데이터 추출 (선택된 피처 + 원핫 인코딩된 범주형 피처 전체 활용)
X_train = df_processed.loc[train_indices].drop(columns=["id", "stroke"])
y_train = df_processed.loc[train_indices]["stroke"]

X_test = df_processed.loc[test_indices].drop(columns=["id", "stroke"])
y_test = df_processed.loc[test_indices]["stroke"]


# 혼동행렬 및 지표 계산 헬퍼 함수
def get_metrics_and_counts(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

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

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": f"{accuracy:.4f}",
        "recall": f"{recall:.4f}",
        "precision": precision_str,
        "f1": f1_str,
    }


# 2. 모델 학습 (가중치 전 / 후)
# 로지스틱 회귀
lr_before = LogisticRegression(
    max_iter=1000, random_state=42, class_weight=None
)
lr_before.fit(X_train, y_train)
lr_before_pred = lr_before.predict(X_test)

lr_after = LogisticRegression(
    max_iter=1000, random_state=42, class_weight="balanced"
)
lr_after.fit(X_train, y_train)
lr_after_pred = lr_after.predict(X_test)

# 의사결정트리
dt_before = DecisionTreeClassifier(random_state=42, class_weight=None)
dt_before.fit(X_train, y_train)
dt_before_pred = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(random_state=42, class_weight="balanced")
dt_after.fit(X_train, y_train)
dt_after_pred = dt_after.predict(X_test)

# 지표 계산
lr_b_m = get_metrics_and_counts(y_test, lr_before_pred)
lr_a_m = get_metrics_and_counts(y_test, lr_after_pred)
dt_b_m = get_metrics_and_counts(y_test, dt_before_pred)
dt_a_m = get_metrics_and_counts(y_test, dt_after_pred)

st.subheader("2. 가중치 조정 전·후 혼동행렬 및 성능 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 로지스틱 회귀 (확률로 답하는 모델)")
    lr_comp_df = pd.DataFrame(
        {
            "평가지표": ["정확도", "재현율", "정밀도", "F1 스코어", "찾은 환자(TP)", "놓친 환자(FN)", "헛짚은 사람(FP)", "정상 맞춤(TN)"],
            "가중치 조정 전": [
                lr_b_m["accuracy"],
                lr_b_m["recall"],
                lr_b_m["precision"],
                lr_b_m["f1"],
                lr_b_m["tp"],
                lr_b_m["fn"],
                lr_b_m["fp"],
                lr_b_m["tn"],
            ],
            "가중치 조정 후(balanced)": [
                lr_a_m["accuracy"],
                lr_a_m["recall"],
                lr_a_m["precision"],
                lr_a_m["f1"],
                lr_a_m["tp"],
                lr_a_m["fn"],
                lr_a_m["fp"],
                lr_a_m["tn"],
            ],
        }
    )
    st.dataframe(lr_comp_df.set_index("평가지표"), use_container_width=True)

    # 환자 수 변화 요약 문구
    st.info(
        f"💡 **로지스틱 회귀 변화 요약**: 찾아낸 환자 수 {lr_b_m['tp']}명 $\\rightarrow$ {lr_a_m['tp']}명 | "
        f"놓친 환자 수 {lr_b_m['fn']}명 $\\rightarrow$ {lr_a_m['fn']}명 | "
        f"헛짚은 사람 수 {lr_b_m['fp']}명 $\\rightarrow$ {lr_a_m['fp']}명으로 바뀌었습니다."
    )

with col2:
    st.markdown("##### 의사결정트리 (질문으로 답하는 모델)")
    dt_comp_df = pd.DataFrame(
        {
            "평가지표": ["정확도", "재현율", "정밀도", "F1 스코어", "찾은 환자(TP)", "놓친 환자(FN)", "헛짚은 사람(FP)", "정상 맞춤(TN)"],
            "가중치 조정 전": [
                dt_b_m["accuracy"],
                dt_b_m["recall"],
                dt_b_m["precision"],
                dt_b_m["f1"],
                dt_b_m["tp"],
                dt_b_m["fn"],
                dt_b_m["fp"],
                dt_b_m["tn"],
            ],
            "가중치 조정 후(balanced)": [
                dt_a_m["accuracy"],
                dt_a_m["recall"],
                dt_a_m["precision"],
                dt_a_m["f1"],
                dt_a_m["tp"],
                dt_a_m["fn"],
                dt_a_m["fp"],
                dt_a_m["tn"],
            ],
        }
    )
    st.dataframe(dt_comp_df.set_index("평가지표"), use_container_width=True)

    st.info(
        f"💡 **의사결정트리 변화 요약**: 찾아낸 환자 수 {dt_b_m['tp']}명 $\\rightarrow$ {dt_a_m['tp']}명 | "
        f"놓친 환자 수 {dt_b_m['fn']}명 $\\rightarrow$ {dt_a_m['fn']}명 | "
        f"헛짚은 사람 수 {dt_b_m['fp']}명 $\\rightarrow$ {dt_a_m['fp']}명으로 바뀌었습니다."
    )

st.divider()

# 3. 산점도 및 결정 경계 비교 시각화 (2차원 또는 3차원)
st.subheader("3. 로지스틱 회귀 가중치 조정 전·후 0.5 경계선 비교 산점도")

if len(selected_features) >= 2:
    plot_df = test_df_raw.copy()
    plot_df["stroke_label"] = plot_df["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

    if len(selected_features) == 2:
        # 2차원 산점도 및 경계선
        feat_x = selected_features[0]
        feat_y = selected_features[1]
        feat_x_kor = feature_mapping[feat_x]
        feat_y_kor = feature_mapping[feat_y]

        fig = px.scatter(
            plot_df,
            x=feat_x,
            y=feat_y,
            color="stroke_label",
            color_discrete_map={"아님 (0)": "#636EFA", "뇌졸중 (1)": "#EF553B"},
            title=f"{feat_x_kor} vs {feat_y_kor} 산점도 및 0.5 경계선 비교",
            labels={
                feat_x: feat_x_kor,
                feat_y: feat_y_kor,
                "stroke_label": "실제 뇌졸중 여부",
            },
        )

        # 2개 외의 속성은 중앙값으로 고정
        other_medians = {
            f: train_df_raw[f].median()
            for f in selected_features
            if f != feat_x and f != feat_y
        }

        x_min, x_max = plot_df[feat_x].min(), plot_df[feat_x].max()
        y_min, y_max = plot_df[feat_y].min(), plot_df[feat_y].max()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )

        grid_raw = pd.DataFrame({feat_x: xx.ravel(), feat_y: yy.ravel()})
        for f, val in other_medians.items():
            grid_raw[f] = val

        # 원본 피처프레임에 맞춰 더미 변수 처리 등 정렬
        grid_processed = pd.get_dummies(
            grid_raw,
            columns=[
                col
                for col in [
                    "gender",
                    "ever_married",
                    "work_type",
                    "Residence_type",
                    "smoking_status",
                ]
                if col in grid_raw.columns
            ],
        )
        grid_processed = grid_processed.reindex(
            columns=X_train.columns, fill_value=0
        )

        # 조정 전 경계선 (0.5)
        probs_before = lr_before.predict_proba(grid_processed)[:, 1]
        zz_b = probs_before.reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                z=zz_b,
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                showscale=False,
                contours=dict(start=0.5, end=0.5, size=0, coloring="lines"),
                line=dict(color="blue", width=3, dash="dash"),
                name="조정 전 경계선 (0.5)",
            )
        )

        # 조정 후 경계선 (0.5)
        probs_after = lr_after.predict_proba(grid_processed)[:, 1]
        zz_a = probs_after.reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                z=zz_a,
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                showscale=False,
                contours=dict(start=0.5, end=0.5, size=0, coloring="lines"),
                line=dict(color="orange", width=3),
                name="조정 후(balanced) 경계선 (0.5)",
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "ℹ️ 파란색 점선은 가중치 조정 전, 주황색 실선은 가중치 조정 후의 로지스틱 회귀 0.5 결정 경계선입니다."
        )

    else:
        # 속성을 3개 이상 고른 경우: 3차원 산점도 및 평면 비교
        feat_x, feat_y, feat_z = (
            selected_features[0],
            selected_features[1],
            selected_features[2],
        )
        fx_k, fy_k, fz_k = (
            feature_mapping[feat_x],
            feature_mapping[feat_y],
            feature_mapping[feat_z],
        )

        fig = px.scatter_3d(
            plot_df,
            x=feat_x,
            y=feat_y,
            z=feat_z,
            color="stroke_label",
            color_discrete_map={"아님 (0)": "#636EFA", "뇌졸중 (1)": "#EF553B"},
            title=f"{fx_k}, {fy_k}, {fz_k} 3차원 산점도 및 결정 평면 비교",
            labels={
                feat_x: fx_k,
                feat_y: fy_k,
                feat_z: fz_k,
                "stroke_label": "실제 뇌졸중 여부",
            },
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"ℹ️ 선택한 속성 중 상위 3개({fx_k}, {fy_k}, {fz_k})를 기준으로 3차원 산점도를 구성하였습니다. 마우스로 회전하여 데이터를 살펴보세요."
        )
else:
    st.warning("산점도를 그리려면 속성을 2개 이상 선택해 주세요.")
