import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------
# STEP 1: PAGE CONFIGURATION (Lệnh Streamlit đầu tiên bắt buộc)
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️"
)

# -----------------------------------------------------------------------------
# STEP 2: CACHED DATA & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Hàm nạp dữ liệu dùng chung hỗ trợ bộ nhớ đệm cache.
    Chuyển đổi chuỗi bytes thô thành đối tượng file-like qua io.BytesIO để tránh lỗi Pandas.
    """
    try:
        file_buffer = io.BytesIO(file_bytes)
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_buffer)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_buffer)
        else:
            st.error("Định dạng tệp không được hỗ trợ! Vui lòng tải lên tệp .csv hoặc .xlsx")
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp dữ liệu: {e}")
        return None

def highlight_fraud(row, target_col):
    """
    Hàm helper định dạng style cho dataframe: 
    Nếu phát hiện giá trị bằng 1 ở cột mục tiêu, dòng đó sẽ được tô màu đỏ và in đậm.
    """
    if row[target_col] == 1:
        return ['color: #D32F2F; font-weight: bold; background-color: #FFEBEE;'] * len(row)
    return [''] * len(row)

# Khai báo tập biến đặc trưng và biến mục tiêu đồng bộ với cấu trúc mô hình
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# -----------------------------------------------------------------------------
# STEP 3: SIDEBAR — VÙNG CẤU HÌNH BỀN VỮNG
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Upload file đầu vào
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu mẫu chứa các cột biến đầu vào từ X_1 đến X_14 và biến phân loại 'default'."
    )
    
    st.divider()
    
    st.subheader("🤖 Cấu hình Mô hình AI")
    model_choice = st.selectbox(
        "Chọn thuật toán",
        options=["Random Forest Classifier", "Decision Tree Classifier", "Logistic Regression"],
        index=0,
        help="Lựa chọn thuật toán học máy phân loại từ quy trình nghiên cứu."
    )
    
    # Hiển thị cấu hình tham số động theo lựa chọn mô hình
    if model_choice == "Random Forest Classifier":
        n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10)
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, step=1)
        criterion = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"])
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42)
        
    elif model_choice == "Decision Tree Classifier":
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=5, step=1)
        criterion = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"])
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42)
        
    elif model_choice == "Logistic Regression":
        penalty = st.selectbox("Phương thức chuẩn hóa (penalty)", options=["l2", "none"])
        C_val = st.slider("Hệ số nghịch đảo chuẩn hóa (C)", min_value=0.01, max_value=10.0
