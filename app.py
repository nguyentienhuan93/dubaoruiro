import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io  # <--- Đã thêm thư viện này để sửa lỗi đọc dữ liệu dạng bytes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------
# STEP 1: PAGE CONFIGURATION (Lệnh đầu tiên của ứng dụng Streamlit)
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
    Hàm nạp dữ liệu dùng chung. Nhận biến bytes (để hashable cho việc cache).
    Sử dụng io.BytesIO để chuyển dữ liệu bytes thô thành đối tượng file-like hợp lệ cho Pandas.
    """
    try:
        # Giải quyết lỗi: Bọc chuỗi bytes thô vào bộ đệm RAM ảo thành dạng File Object
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
        st.error(f"Lỗi hệ thống khi phân tích cấu trúc tệp dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng đầu vào (X) và biến mục tiêu (y)
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# -----------------------------------------------------------------------------
# STEP 3: SIDEBAR — VÙNG THU THẬP CẤU HÌNH & ĐIỀU KHIỂN
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Bộ tải tệp tin đầu vào
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện mẫu (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp chứa các biến từ X_1 đến X_14 cùng nhãn mục tiêu 'default'."
    )
    
    st.divider()
    
    st.subheader("🤖 Cấu hình Mô hình AI")
    # Lựa chọn thuật toán
    model_choice = st.selectbox(
        "Chọn thuật toán",
        options=["Random Forest Classifier", "Decision Tree Classifier", "Logistic Regression"],
        index=0,
        help="Chọn thuật toán học máy phân loại phù hợp."
    )
    
    # Hiển thị tham số động theo mô hình lựa chọn
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
        C_val = st.slider("Hệ số nghịch đảo chuẩn hóa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=2000, value=1000)
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42)

    st.divider()
    
    # Điểm duy nhất kích hoạt việc Huấn luyện mô hình
    train_clicked = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# STEP 4: HEADER — VÙNG ĐỊNH HƯỚNG TRẠNG THÁI ỨNG DỤNG
# -----------------------------------------------------------------------------
st.title("🛡️ Ứng Dụng Phát Hiện Giao Dịch Gian Lận")
st.caption("Giao diện phân tích rủi ro tín dụng và tự động phát hiện giao dịch gian lận (default) tích hợp từ mô hình Học máy.")

# Kiểm tra xử lý trạng thái rỗng nếu chưa tải dữ liệu lên
if uploaded_file is None:
    st.info("💡 Vui lòng tải tệp dữ liệu huấn luyện mẫu (`.csv` hoặc `.xlsx`) tại thanh Sidebar bên trái để bắt đầu khám phá ứng dụng.")
    st.stop()
else:
    # Truyền dữ liệu dạng .getvalue() để đảm bảo an toàn cho luồng hash bộ nhớ đệm
    df_raw = load_data(uploaded_file.getvalue(), uploaded_file.name)
    if df_raw is None:
        st.stop()
    st.caption(f"📁 Đang kết nối tệp dữ liệu: `{uploaded_file.name}`")

st.divider()

# -----------------------------------------------------------------------------
# PROCESS: KHỐI HUÂN LUYỆN MÔ HÌNH (Lưu tập trung vào st.session_state)
# -----------------------------------------------------------------------------
if train_clicked:
    # Xác thực kiểm tra cấu trúc schema cột
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ Lỗi dữ liệu đầu vào: Tệp thiếu các cột bắt buộc sau: {missing_cols}")
    else:
        with st.spinner("⏳ Hệ thống đang tiến hành huấn luyện mô hình... Vui lòng đợi."):
            X = df_raw[FEATURES]
            y = df_raw[TARGET]
            
            # Phân tách tập train/test mẫu
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
