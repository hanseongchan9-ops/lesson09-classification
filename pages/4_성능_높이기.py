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


# 데이터 로드 및 전처리 함수 (채점표 페이지와 동일한 방식)
@st.cache_data
def get_model_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # bmi 결측치는 훈련용 중앙값으로 채우기 위해 전체 전처리 전 복사본 활용 가능하도록 구성
    # 범주형 변수 원핫 인코딩
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
    return df, X_train, X_test, y_train, y_test, train_idx, test_idx


try:
    (
        df_raw,
        X_train_full,
        X_test_full,
        y_train,
        y_test,
        train_idx,
        test_idx,
    ) = get_model_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 페이지 제목
st.title("📈 뇌졸중 예측 모델 성능 높이기 (클래스 가중치 조정)")
st.markdown(
    "데이터 불균형 문제를 해결하기 위해 가중치를 맞추기 전과 맞춘 후의 모델 성능 변화를 비교하는 공간입니다."
)

st.divider()

# 1. 입력 속성 선택 UI (기본값: age, avg_glucose_level, hypertension, heart_disease)
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

# 2. 전처리 (bmi 비어 있는 경우 훈련용 중앙값으로 채우기)
train_raw_subset = df_raw.loc[train_idx].copy()
test_raw_subset = df_raw.loc[test_idx].copy()

bmi_median = train_raw_subset["bmi"].median()
train_raw_subset["bmi"] = train_raw_subset["bmi"].fillna(bmi_median)
test_raw_subset["bmi"] = test_raw_subset["bmi"].fillna(bmi_median)

# 모델 학습에 사용할 X_train, X_test 구성 (선택된 속성들 + 원핫인코딩된 기타 피처 결합)
# 사용자가 선택한 수치형/이진형 속성 반영
X_train_sel = X_train_full.copy()
X_test_sel = X_test_full.copy()

# 만약 bmi가 선택되었거나 중앙값 채운 값을 반영하려면 해당 컬럼 업데이트
if "bmi" in X_train_sel.columns:
    X_train_sel["bmi"] = train_raw_subset["bmi"].values
    X_test_sel["bmi"] = test_raw_subset["bmi"].values


# 혼동행렬 및 평가지표 계산 함수
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


# 3. 모델 학습 (가중치 조정 전 vs 후)
# 로지스틱 회귀
lr_before = LogisticRegression(
    max_iter=1000, random_state=42, class_weight=None
)
lr_before.fit(X_train_sel, y_train)
lr_before_pred = lr_before.predict(X_test_sel)

lr_after = LogisticRegression(
    max_iter=1000, random_state=42, class_weight="balanced"
)
lr_after.fit(X_train_sel, y_train)
lr_after_pred = lr_after.predict(X_test_sel)

# 의사결정트리
dt_before = DecisionTreeClassifier(random_state=42, class_weight=None)
dt_before.fit(X_train_sel, y_train)
dt_before_pred = dt_before.predict(X_test_sel)

dt_after = DecisionTreeClassifier(random_state=42, class_weight="balanced")
dt_after.fit(X_train_sel, y_train)
dt_after_pred = dt_after.predict(X_test_sel)

# 지표 계산
lr_b_m = get_metrics_and_counts(y_test, lr_before_pred)
lr_a_m = get_metrics_and_counts(y_test, lr_after_pred)
dt_b_m = get_metrics_and_counts(y_test, dt_before_pred)
dt_a_m = get_metrics_and_counts(y_test, dt_after_pred)

st.subheader("2. 가중치 조정 전·후 혼동행렬 및 4대 평가지표 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 로지스틱 회귀(확률로 답하는 모델)")
    lr_comp_df = pd.DataFrame(
        {
            "구분": [
                "정확도 (Accuracy)",
                "재현율 (Recall)",
                "정밀도 (Precision)",
                "F1 스코어",
                "찾은 환자 (TP)",
                "놓친 환자 (FN)",
                "헛짚은 사람 (FP)",
                "정상 맞춤 (TN)",
            ],
            "조정 전": [
                lr_b_m["accuracy"],
                lr_b_m["recall"],
                lr_b_m["precision"],
                lr_b_m["f1"],
                lr_b_m["tp"],
                lr_b_m["fn"],
                lr_b_m["fp"],
                lr_b_m["tn"],
            ],
            "조정 후 (balanced)": [
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
    st.dataframe(lr_comp_df.set_index("구분"), use_container_width=True)
    st.info(
        f"📌 **로지스틱 회귀 변화**: 찾아낸 환자 수 {lr_b_m['tp']}명 → {lr_a_m['tp']}명 | "
        f"놓친 환자 수 {lr_b_m['fn']}명 → {lr_a_m['fn']}명 | "
        f"헛짚은 사람 수 {lr_b_m['fp']}명 → {lr_a_m['fp']}명"
    )

with col2:
    st.markdown("##### 의사결정트리(질문으로 답하는 모델)")
    dt_comp_df = pd.DataFrame(
        {
            "구분": [
                "정확도 (Accuracy)",
                "재현율 (Recall)",
                "정밀도 (Precision)",
                "F1 스코어",
                "찾은 환자 (TP)",
                "놓친 환자 (FN)",
                "헛짚은 사람 (FP)",
                "정상 맞춤 (TN)",
            ],
            "조정 전": [
                dt_b_m["accuracy"],
                dt_b_m["recall"],
                dt_b_m["precision"],
                dt_b_m["f1"],
                dt_b_m["tp"],
                dt_b_m["fn"],
                dt_b_m["fp"],
                dt_b_m["tn"],
            ],
            "조정 후 (balanced)": [
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
    st.dataframe(dt_comp_df.set_index("구분"), use_container_width=True)
    st.info(
        f"📌 **의사결정트리 변화**: 찾아낸 환자 수 {dt_b_m['tp']}명 → {dt_a_m['tp']}명 | "
        f"놓친 환자 수 {dt_b_m['fn']}명 → {dt_a_m['fn']}명 | "
        f"헛짚은 사람 수 {dt_b_m['fp']}명 → {dt_a_m['fp']}명"
    )

st.divider()

# 4. 산점도 및 0.5 결정 경계(평면) 시각화
st.subheader("3. 로지스틱 회귀 가중치 조정 전·후 0.5 경계선 비교")

plot_df = test_raw_subset.copy()
plot_df["stroke_label"] = plot_df["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

if len(selected_features) == 2:
    feat_x = selected_features[0]
    feat_y = selected_features[1]
    fx_k = feature_mapping[feat_x]
    fy_k = feature_mapping[feat_y]

    fig = px.scatter(
        plot_df,
        x=feat_x,
        y=feat_y,
        color="stroke_label",
        color_discrete_map={"아님 (0)": "#636EFA", "뇌졸중 (1)": "#EF553B"},
        title=f"{fx_k} vs {fy_k} 산점도 및 전·후 0.5 경계선 비교",
        labels={feat_x: fx_k, feat_y: fy_k, "stroke_label": "실제 뇌졸중 여부"},
    )

    # 나머지 속성은 훈련용 중앙값으로 고정
    other_medians = {
        f: train_raw_subset[f].median()
        for f in X_train_full.columns
        if f not in [feat_x, feat_y]
    }

    x_min, x_max = plot_df[feat_x].min(), plot_df[feat_x].max()
    y_min, y_max = plot_df[feat_y].min(), plot_df[feat_y].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )

    grid_df = pd.DataFrame({feat_x: xx.ravel(), feat_y: yy.ravel()})
    for f, val in other_medians.items():
        if f in X_train_full.columns:
            grid_df[f] = val

    grid_processed = pd.get_dummies(
        grid_df,
        columns=[
            c
            for c in [
                "gender",
                "ever_married",
                "work_type",
                "Residence_type",
                "smoking_status",
            ]
            if c in grid_df.columns
        ],
    )
    grid_processed = grid_processed.reindex(
        columns=X_train_sel.columns, fill_value=0
    )

    # 조정 전 0.5 경계선
    probs_b = lr_before.predict_proba(grid_processed)[:, 1]
    zz_b = probs_b.reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            z=zz_b,
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            showscale=False,
            contours=dict(start=0.5, end=0.5, size=0, coloring="lines"),
            line=dict(color="blue", width=3, dash="dash"),
            name="가중치 조정 전 (0.5)",
        )
    )

    # 조정 후 0.5 경계선
    probs_a = lr_after.predict_proba(grid_processed)[:, 1]
    zz_a = probs_a.reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            z=zz_a,
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            showscale=False,
            contours=dict(start=0.5, end=0.5, size=0, coloring="lines"),
            line=dict(color="orange", width=3),
            name="가중치 조정 후 (balanced, 0.5)",
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "ℹ️ 파란색 점선은 가중치 조정 전, 주황색 실선은 가중치 조정 후(balanced)의 0.5 결정 경계선입니다."
    )

    if np.all(probs_b < 0.5) or np.all(probs_b > 0.5):
        st.warning(
            "⚠️ 가중치 조정 전 0.5 경계선이 현재 산점도 화면 범위 밖에 위치해 있습니다."
        )
    if np.all(probs_a < 0.5) or np.all(probs_a > 0.5):
        st.warning(
            "⚠️ 가중치 조정 후 0.5 경계선이 현재 산점도 화면 범위 밖에 위치해 있습니다."
        )

elif len(selected_features) >= 3:
    # 속성을 셋 이상 고른 경우: 3차원 산점도
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
        f"ℹ️ 선택한 속성 중 상위 3개({fx_k}, {fy_k}, {fz_k})를 기준으로 3차원 산점도를 구성하였습니다. 마우스로 회전하여 데이터를 확인하세요."
    )
