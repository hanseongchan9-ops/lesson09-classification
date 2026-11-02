import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="분류 모델", page_icon="🧠", layout="wide")

st.title("🧠 뇌졸중 예측 분류 모델 분석")

# 1. 데이터 로드
@st.cache_data
def load_data():
    for path in ['healthcare-dataset-stroke-data.csv', 'data/healthcare-dataset-stroke-data.csv', '../healthcare-dataset-stroke-data.csv']:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    return None

df = load_data()

if df is None:
    st.error("⚠️ 뇌졸중 데이터셋 파일('healthcare-dataset-stroke-data.csv')을 찾을 수 없습니다. 프로젝트 루트 폴더나 data 폴더에 파일을 위치시켜 주세요.")
    st.stop()

# 2. 속성 및 한글 이름 매핑 설정
feature_map = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}
reverse_feature_map = {v: k for k, v in feature_map.items()}

# 처음에는 체질량지수(bmi)를 뺀 넷이 골라져 있게 설정
default_features = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease']
default_korean_features = [feature_map[f] for f in default_features]

st.sidebar.header("⚙️ 모델 설정 (탐색 및 입력)")
st.sidebar.markdown("입력으로 사용할 속성을 선택해 주세요:")

selected_korean = st.sidebar.multiselect(
    "입력 속성 선택",
    options=list(feature_map.values()),
    default=default_korean_features
)

selected_features = [reverse_feature_map[k] for k in selected_korean]

# 두 개보다 적게 고르면 안내문을 표시하고 멈춤
if len(selected_features) < 2:
    st.warning("⚠️ 입력 속성을 **2개 이상** 선택해 주세요. (사이드바에서 속성을 추가해 주세요)")
    st.stop()

# 3. 데이터 분할 (번호 순 정렬 후 10명씩 묶어 앞 3명 테스트용, 나머지 7명 훈련용)
df_sorted = df.sort_index().reset_index(drop=True)

test_indices = [i for i in range(len(df_sorted)) if i % 10 < 3]
train_indices = [i for i in range(len(df_sorted)) if i % 10 >= 3]

train_df = df_sorted.iloc[train_indices].copy()
test_df = df_sorted.iloc[test_indices].copy()

# 체질량지수(bmi)는 고른 경우에만 훈련용의 중앙값으로 채우기
if 'bmi' in selected_features:
    bmi_median = train_df['bmi'].median()
    train_df['bmi'] = train_df['bmi'].fillna(bmi_median)
    test_df['bmi'] = test_df['bmi'].fillna(bmi_median)

X_train = train_df[selected_features]
y_train = train_df['stroke']
X_test = test_df[selected_features]
y_test = test_df['stroke']

# 4. 모델 학습
# 로지스틱 회귀 (확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# 의사결정트리 (질문으로 답하는 모델: 질문 3번까지, 마지막 마디 5명 미만이면 분할 중지, 난수 고정)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)

# 기준 모델 (입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)
dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_model.fit(X_train, y_train)

# 5. 정확도 계산
lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))

dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))

dummy_train_acc = accuracy_score(y_train, dummy_model.predict(X_train))
dummy_test_acc = accuracy_score(y_test, dummy_model.predict(X_test))

st.subheader("📊 모델 정확도 비교")
st.markdown("카드에는 **테스트 데이터의 정확도**를 크게 적고, 그 아래에 **훈련 데이터의 정확도**와 **테스트 데이터의 정확도**를 나란히 작게 표시합니다.")

col1, col2, col3 = st.columns(3)

def render_metric_card(title, test_acc, train_acc, test_acc_sub):
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #f9f9f9; text-align: center; margin-bottom: 10px;">
        <h4 style="margin: 0; color: #333;">{title}</h4>
        <h1 style="margin: 10px 0; color: #1f77b4; font-size: 2.5em;">{test_acc:.1%}</h1>
        <p style="margin: 0; font-size: 0.9em; color: #666;">
            훈련: <b>{train_acc:.1%}</b> &nbsp;|&nbsp; 테스트: <b>{test_acc_sub:.1%}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col1:
    render_metric_card("로지스틱 회귀<br>(확률로 답하는 모델)", lr_test_acc, lr_train_acc, lr_test_acc)

with col2:
    render_metric_card("의사결정트리<br>(질문으로 답하는 모델)", dt_test_acc, dt_train_acc, dt_test_acc)

with col3:
    render_metric_card("기준 모델<br>(입력을 보지 않고 많은 쪽으로 답하는 모델)", dummy_test_acc, dummy_train_acc, dummy_test_acc)

st.markdown("---")

# 6. 산점도 및 결정 경계 시각화
st.subheader("📉 속성 관계 산점도 및 결정 경계")
st.markdown("고른 속성 가운데 **둘을 골라** 가로축과 세로축으로 삼고, 테스트 데이터를 점으로 찍어 실제 뇌졸중 여부를 색으로 나타냅니다.")

scat_col1, scat_col2 = st.columns(2)
with scat_col1:
    x_axis_korean = st.selectbox("가로축 속성", selected_korean, index=0)
with scat_col2:
    default_y_idx = 1 if len(selected_korean) > 1 else 0
    y_axis_korean = st.selectbox("세로축 속성", selected_korean, index=default_y_idx)

x_feat = reverse_feature_map[x_axis_korean]
y_feat = reverse_feature_map[y_axis_korean]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 고정
other_feats = [f for f in selected_features if f not in [x_feat, y_feat]]
median_values = {f: train_df[f].median() for f in other_feats}

median_desc = ", ".join([f"{feature_map[f]}: {val:.2f}" for f, val in median_values.items()]) if other_feats else "없음 (모든 선택 속성이 축으로 사용됨)"
st.info(f"💡 **두 축 외의 속성 고정 값 (훈련용 중앙값 기준):** {median_desc}")

# 산점도 및 배경 영역 계산
test_df_plot = test_df.copy()
test_df_plot['stroke_label'] = test_df_plot['stroke'].map({1: '뇌졸중 (양성)', 0: '아님'})

x_min, x_max = test_df[x_feat].min() * 0.9, test_df[x_feat].max() * 1.1
y_min, y_max = test_df[y_feat].min() * 0.9, test_df[y_feat].max() * 1.1

if set([x_feat, y_feat]).issubset({'hypertension', 'heart_disease'}):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

grid_df_dict = {x_feat: xx.ravel(), y_feat: yy.ravel()}
for f in other_feats:
    grid_df_dict[f] = median_values[f]
grid_input = pd.DataFrame(grid_df_dict)[selected_features]

lr_probs = lr_model.predict_proba(grid_input)[:, 1]
lr_zz = lr_probs.reshape(xx.shape)
dt_zz = dt_model.predict(grid_input).reshape(xx.shape)

fig = px.scatter(
    test_df_plot,
    x=x_feat,
    y=y_feat,
    color='stroke_label',
    labels={x_feat: x_axis_korean, y_feat: y_axis_korean, 'stroke_label': '실제 뇌졸중 여부'},
    title=f"{x_axis_korean} vs {y_axis_korean} 산점도 및 모델 경계",
    color_discrete_map={'뇌졸중 (양성)': '#d62728', '아님': '#1f77b4'}
)

# 의사결정트리 영역 옅은 색 칠하기 (Contour)
fig.add_trace(
    go.Contour(
        z=dt_zz,
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        showscale=False,
        colorscale=[[0, 'rgba(31, 119, 180, 0.08)'], [1, 'rgba(214, 39, 40, 0.08)']],
        hoverinfo='skip',
        name='의사결정트리 영역'
    )
)

# 로지스틱 회귀 0.5 결정 경계선 그리기
fig.add_trace(
    go.Contour(
        z=lr_zz,
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        showscale=False,
        contours=dict(coloring='lines', showlabels=False, start=0.5, end=0.5, size=0.1),
        line=dict(color='black', width=2, dash='dash'),
        name='로지스틱 회귀 경계선 (0.5)'
    )
)

fig.update_layout(legend_title_text='범례')
st.plotly_chart(fig, use_container_width=True)

# 결정 경계선이 그림 범위 안에 있는지 확인
boundary_in_plot = (lr_zz.min() < 0.5) and (lr_zz.max() > 0.5)
if boundary_in_plot:
    st.write("✅ 로지스틱 회귀 모델의 0.5 확률 가르는 선이 현재 그림 범위 안에 있습니다.")
else:
    st.write("⚠️ 로지스틱 회귀 모델의 0.5 확률 가르는 선이 현재 그림 범위 밖에 있습니다.")

st.markdown("---")

# 7. 의사결정트리 가지 그림 (Graphviz)
st.subheader("🌳 의사결정트리 구조 시각화")
st.markdown("의사결정트리 모델이 질문을 던져 가지가 갈라지는 구조입니다. 각 마디에는 훈련용 사람 수, 실제 뇌졸중인 사람 수, 비율이 표시됩니다.")

dot_data = export_graphviz(
    dt_model,
    out_file=None,
    feature_names=[feature_map[f] for f in selected_features],
    class_names=['아님', '뇌졸중'],
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2
)

st.graphviz_chart(dot_data)

# 트리 구조 통계 요약
n_nodes = dt_model.tree_.node_count
children_left = dt_model.tree_.children_left
children_right = dt_model.tree_.children_right
value = dt_model.tree_.value

leaf_nodes = [i for i in range(n_nodes) if children_left[i] == children_right[i]]
total_leaves = len(leaf_nodes)

no_stroke_leaves = 0
for leaf in leaf_nodes:
    vals = value[leaf][0]
    if vals[0] >= vals[1]:
        no_stroke_leaves += 1

used_features_idx = dt_model.tree_.feature
used_features = set([selected_features[i] for i in used_features_idx if i != -2])
used_features_kor = [feature_map[f] for f in used_features]
used_features_str = ", ".join(used_features_kor) if used_features_kor else "없음"

st.markdown(f"""
- **답을 내는 마디(리프 노드)의 총 개수:** {total_leaves}개
- **그중 '아님'이라고 답하는 마디의 개수:** {no_stroke_leaves}개
- **고른 속성 가운데 이 나무가 실제로 물은 것:** {used_features_str}
""")
