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

# 1. 데이터 로드 (다양한 경로 탐색)
@st.cache_data
def load_data():
    paths = [
        'healthcare-dataset-stroke-data.csv', 
        'data/healthcare-dataset-stroke-data.csv', 
        '../healthcare-dataset-stroke-data.csv',
        './healthcare-dataset-stroke-data.csv'
    ]
    for path in paths:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    return None

df = load_data()

if df is None:
    st.error("⚠️ 뇌졸중 데이터셋 파일('healthcare-dataset-stroke-data.csv')을 찾을 수 없습니다. 프로젝트 폴더(또는 data 폴더)에 파일을 확인해 주세요.")
    st.stop()

# 열 이름 공백 제거 (혹시 모를 공백 에러 방지)
df.columns = df.columns.str.strip()

# 2. 속성 및 한글 이름 매핑 설정
feature_map = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}
reverse_feature_map = {v: k for k, v in feature_map.items()}

# 데이터셋에 실제 존재하는 열만 필터링
available_features = [f for f in feature_map.keys() if f in df.columns]
if 'stroke' not in df.columns:
    st.error("⚠️ 데이터셋에 'stroke' 열이 존재하지 않습니다.")
    st.stop()

default_features = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease']
default_selected = [feature_map[f] for f in default_features if f in available_features]

st.sidebar.header("⚙️ 모델 설정 (탐색 및 입력)")
st.sidebar.markdown("입력으로 사용할 속성을 선택해 주세요:")

selected_korean = st.sidebar.multiselect(
    "입력 속성 선택",
    options=[feature_map[f] for f in available_features],
    default=default_selected
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
if 'bmi' in selected_features and 'bmi' in train_df.columns:
    train_df['bmi'] = pd.to_numeric(train_df['bmi'], errors='coerce')
    test_df['bmi'] = pd.to_numeric(test_df['bmi'], errors='coerce')
    bmi_median = train_df['bmi'].median()
    train_df['bmi'] = train_df['bmi'].fillna(bmi_median)
    test_df['bmi'] = test_df['bmi'].fillna(bmi_median)

X_train = train_df[selected_features].astype(float)
y_train = train_df['stroke'].astype(int)
X_test = test_df[selected_features].astype(float)
y_test = test_df['stroke'].astype(int)

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

# 6. 산점도 및 결정 경계 시각
