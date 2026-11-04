import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 페이지 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🤖", layout="wide"
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


# 데이터 로드
@st.cache_data
def get_raw_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df_raw = get_raw_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 페이지 제목
st.title("🤖 뇌졸중 예측 분류 모델 실습")
st.markdown(
    "입력 속성을 선택하고 로지스틱 회귀와 의사결정트리 모델을 학습 및 평가하는 공간입니다."
)

st.divider()

# 1. 속성 선택 UI (기본값: bmi 제외한 넷)
st.subheader("1. 입력 속성 선택")
default_features_kor = ["나이", "평균 혈당", "고혈압", "심장병"]
all_features_kor = list(feature_mapping.values())

selected_features_kor = st.multiselect(
    "학습에 사용할 속성을 골라주세요 (2개 이상 선택):",
    options=all_features_kor,
    default=default_features_kor,
)

# 두 개보다 적게 고르면 안내문 표시 후 중단
if len(selected_features_kor) < 2:
    st.warning("⚠️ 속성을 **2개 이상** 선택해 주세요.")
    st.stop()

selected_features = [
    reverse_feature_mapping[feat] for feat in selected_features_kor
]

st.divider()

# 2. 데이터 분할 규칙 적용 (번호 순 정렬 -> 열 명씩 묶어 앞 3명 테스트, 나머지 7명 훈련)
df_sorted = df_raw.sort_values(by="id").reset_index(drop=True)

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

train_df = df_sorted.loc[train_indices].copy()
test_df = df_sorted.loc[test_indices].copy()

# 3. 전처리 (bmi 중앙값 처리 등)
bmi_median = train_df["bmi"].median()

for dataset in [train_df, test_df]:
    if "bmi" in dataset.columns:
        dataset["bmi"] = dataset["bmi"].fillna(bmi_median)

X_train = train_df[selected_features]
y_train = train_df["stroke"]

X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 4. 모델 학습
# 로지스틱 회귀
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc = lr_model.score(X_test, y_test)

# 의사결정트리 (질문 3번 제한, 마디당 최소 샘플 5명 미만 분할 금지, 난수 고정)
dt_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_model.fit(X_train, y_train)
dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# 기준 모델 (다수 클래스로만 답하는 모델)
majority_class = y_train.mode()[0]
baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()

# 5. 정확도 카드 표시
st.subheader("2. 모델 성능 비교 (정확도)")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="로지스틱 회귀 (확률로 답하는 모델)",
        value=f"{lr_test_acc * 100:.2f}%",
        help="테스트 데이터 기준 정확도",
    )
    st.caption(
        f"훈련 정확도: {lr_train_acc * 100:.2f}% | 테스트 정확도: {lr_test_acc * 100:.2f}%"
    )

with c2:
    st.metric(
        label="의사결정트리 (질문으로 답하는 모델)",
        value=f"{dt_test_acc * 100:.2f}%",
        help="테스트 데이터 기준 정확도",
    )
    st.caption(
        f"훈련 정확도: {dt_train_acc * 100:.2f}% | 테스트 정확도: {dt_test_acc * 100:.2f}%"
    )

with c3:
    st.metric(
        label="다수 클래스 기준 모델",
        value=f"{baseline_test_acc * 100:.2f}%",
        help="훈련용에서 많은 쪽으로만 답하는 모델",
    )
    st.caption(
        f"훈련 정확도: {baseline_train_acc * 100:.2f}% | 테스트 정확도: {baseline_test_acc * 100:.2f}%"
    )

st.divider()

# 6. 산점도 및 결정 경계 시각화
st.subheader("3. 2차원 산점도 및 분류 경계 시각화")

if len(selected_features) >= 2:
    feat_x = selected_features[0]
    feat_y = selected_features[1]
    feat_x_kor = feature_mapping[feat_x]
    feat_y_kor = feature_mapping[feat_y]

    plot_df = test_df.copy()
    plot_df["stroke_label"] = plot_df["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

    fig = px.scatter(
        plot_df,
        x=feat_x,
        y=feat_y,
        color="stroke_label",
        color_discrete_map={"아님 (0)": "#636EFA", "뇌졸중 (1)": "#EF553B"},
        title=f"{feat_x_kor} vs {feat_y_kor} 테스트 데이터 산점도",
        labels={
            feat_x: feat_x_kor,
            feat_y: feat_y_kor,
            "stroke_label": "실제 뇌졸중 여부",
        },
    )

    other_features_medians = {}
    for f in selected_features:
        if f != feat_x and f != feat_y:
            other_features_medians[f] = train_df[f].median()

    x_min, x_max = plot_df[feat_x].min(), plot_df[feat_x].max()
    y_min, y_max = plot_df[feat_y].min(), plot_df[feat_y].max()
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_range, y_range)

    grid_data = pd.DataFrame(
        {feat_x: xx.ravel(), feat_y: yy.ravel()}
    )
    for f, val in other_features_medians.items():
        grid_data[f] = val

    grid_data = grid_data[selected_features]

    try:
        probs = lr_model.predict_proba(grid_data)[:, 1]
        zz = probs.reshape(xx.shape)

        fig.add_trace(
            go.Contour(
                z=zz,
                x=x_range,
                y=y_range,
                showscale=False,
                contours=dict(
                    start=0.5, end=0.5, size=0, coloring="lines"
                ),
                line=dict(color="green", width=3),
                name="로지스틱 회귀 경계 (0.5)",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        fixed_info_str = ", ".join(
            [
                f"{feature_mapping[f]}={val:.2f}"
                for f, val in other_features_medians.items()
            ]
        )
        if fixed_info_str:
            st.info(
                f"ℹ️ 가로축({feat_x_kor}), 세로축({feat_y_kor}) 외에 모델 계산에 사용된 나머지 속성들은 훈련용 데이터의 중앙값({fixed_info_str})으로 고정하여 계산되었습니다."
            )
        else:
            st.info(
                f"ℹ️ 선택한 두 개의 속성만으로 로지스틱 회귀 경계선이 계산되었습니다."
            )

        if np.all(probs < 0.5) or np.all(probs > 0.5):
            st.warning(
                "⚠️ 로지스틱 회귀의 0.5 기준선이 현재 산점도 범위 밖에 위치해 있어 화면에 나타나지 않을 수 있습니다."
            )

    except Exception as ex:
        st.error(f"결정 경계선 그리기 중 오류 발생: {ex}")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 7. 의사결정트리 구조 시각화
st.subheader("4. 의사결정트리 구조 (질문 가지 그림)")


def get_custom_tree_dot(tree_model, feature_names, feature_map_dict):
    from sklearn.tree import export_graphviz

    feature_names_kor = [feature_map_dict.get(f, f) for f in feature_names]
    dot_data = export_graphviz(
        tree_model,
        out_file=None,
        feature_names=feature_names_kor,
        class_names=["아님", "뇌졸중"],
        filled=True,
        rounded=True,
        special_characters=True,
        precision=2,
    )
    return dot_data


try:
    dot_string = get_custom_tree_dot(dt_model, selected_features, feature_mapping)
    st.graphviz_chart(dot_string)
except Exception as e:
    st.info("트리 시각화를 생성하는 중입니다.")

n_nodes = dt_model.tree_.node_count
is_leaf = dt_model.tree_.children_left == -1
leaf_nodes = np.where(is_leaf)[0]
total_leaves = len(leaf_nodes)

leaf_values = dt_model.tree_.value[leaf_nodes]
negative_leaves = sum(
    1 for val in leaf_values if np.argmax(val[0]) == 0
)

used_features_idx = set(dt_model.tree_.feature)
used_features_idx.discard(-2)
used_feature_names_kor = [
    feature_mapping[selected_features[i]]
    for i in used_features_idx
    if i < len(selected_features)
]
used_features_str = (
    ", ".join(used_feature_names_kor) if used_feature_names_kor else "없음"
)

st.markdown(
    f"""
- **답을 내는 마디(리프 노드)의 총 개수:** {total_leaves}개
- **그중 '아님'이라고 답하는 마디 개수:** {negative_leaves}개
- **고른 속성 가운데 이 나무가 실제로 물은 속성:** {used_features_str}
"""
)
