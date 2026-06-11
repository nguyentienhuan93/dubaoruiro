import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io  # <--- Bộ thư viện xử lý đệm dữ liệu RAM để sửa lỗi nhận diện bytes
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
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận tại Agribank",
    page_icon="💖"
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
        # SỬA LỖI: Bọc dữ liệu nhị phân vào bộ đệm dòng dữ liệu hợp lệ
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
        C_val = st.slider("Hệ số nghịch đảo chuẩn hóa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=2000, value=1000)
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42)

    st.divider()
    
    # Điểm duy nhất kích hoạt huấn luyện mô hình
    train_clicked = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# STEP 4: HEADER — VÙNG ĐỊNH HƯỚNG & XỬ LÝ TRẠNG THÁI RỖNG
# -----------------------------------------------------------------------------
st.title("💖 Ứng Dụng Phát Hiện Giao Dịch Gian Lận tại Agribank 💖")
st.caption("Ứng dụng hỗ trợ tự động hóa việc thẩm định rủi ro tín dụng và phát hiện gian lận dựa trên quy trình huấn luyện AI.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu huấn luyện mẫu (.csv hoặc .xlsx) tại Sidebar bên trái để kích hoạt các chức năng phân tích.")
    st.stop()
else:
    # Lấy dữ liệu bytes thô đưa vào hàm bọc đệm cache
    df_raw = load_data(uploaded_file.getvalue(), uploaded_file.name)
    if df_raw is None:
        st.stop()
    st.caption(f"📁 Đang dùng tệp: `{uploaded_file.name}`")

st.divider()

# -----------------------------------------------------------------------------
# KHỐI HUẤN LUYỆN: Thực hiện fit và lưu trữ kết quả vào st.session_state
# -----------------------------------------------------------------------------
if train_clicked:
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc: {missing_cols}")
    else:
        with st.spinner("⏳ Đang tiến hành huấn luyện mô hình AI... Vui lòng đợi."):
            X = df_raw[FEATURES]
            y = df_raw[TARGET]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(random_state))
            
            if model_choice == "Random Forest Classifier":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=int(random_state))
            elif model_choice == "Decision Tree Classifier":
                model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=int(random_state))
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(penalty=penalty, C=C_val, max_iter=int(max_iter), random_state=int(random_state))
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Lưu trữ toàn cục 3 thành phần chính
            st.session_state['trained_model'] = model
            st.session_state['model_name'] = model_choice
            st.session_state['metrics'] = {
                'cm': confusion_matrix(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'accuracy': accuracy_score(y_test, y_pred),
                'y_test': y_test,
                'y_pred': y_pred
            }
        st.success(f"🎉 Huấn luyện thành công mô hình **{model_choice}**!")

# -----------------------------------------------------------------------------
# STEP 5: TABS LAYOUT — NỘI DUNG & KẾT QUẢ ĐẦU RA
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện", 
    "🔮 Sử dụng mô hình"
])

# ---- TAB 1: TỔNG QUAN DỮ LIỆU ----
with tab1:
    st.subheader("📋 Phân tích Thống kê Mô tả")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Số dòng (Rows)", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số cột (Columns)", f"{df_raw.shape[1]}")
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    col_m3.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
    
    st.write("**Xem trước dữ liệu thô (5 dòng đầu):**")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("**Bảng chỉ số thống kê mô tả tập dữ liệu đặc trưng (X & y):**")
    available_cols = [c for c in FEATURES + [TARGET] if c in df_raw.columns]
    if available_cols:
        st.dataframe(df_raw[available_cols].describe(), use_container_width=True)

# ---- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ----
with tab2:
    st.subheader("🎨 Phân phối đồ thị cấu trúc biến")
    
    if TARGET in df_raw.columns:
        target_counts = df_raw[TARGET].value_counts().reset_index()
        target_counts.columns = [TARGET, 'Số lượng']
        target_counts[TARGET] = target_counts[TARGET].astype(str)
        
        fig_target = px.bar(
            target_counts, x=TARGET, y='Số lượng',
            title=f"Tỷ lệ phân phối lớp mục tiêu rủi ro ({TARGET})",
            labels={TARGET: "Trạng thái (0: Bình thường, 1: Gian lận)"},
            color=TARGET, color_discrete_sequence=px.colors.qualitative.Set2,
            height=350
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.write("**Cấu hình hiển thị các biến đầu vào:**")
    selected_features = st.multiselect(
        "Lựa chọn các biến đặc trưng cần trực quan hóa:",
        options=FEATURES,
        default=FEATURES[:4]
    )
    
    if selected_features:
        for i in range(0, len(selected_features), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(selected_features):
                    feat = selected_features[i + j]
                    if feat in df_raw.columns:
                        fig_hist = px.histogram(
                            df_raw, x=feat,
                            title=f"Phân phối tần suất đặc trưng của cột {feat}",
                            marginal="box",
                            color_discrete_sequence=['#4A90E2'],
                            height=300
                        )
                        cols[j].plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Vui lòng tích chọn các biến đầu vào để hệ thống hiển thị biểu đồ phân phối.")

# ---- TAB 3: KẾT QUẢ HUÂN LUYỆN & KIỂM ĐỊNH ----
with tab3:
    st.subheader("📊 Đánh giá chất lượng phân loại mô hình")
    
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Hiện tại chưa ghi nhận dữ liệu huấn luyện. Vui lòng nhấn nút **🚀 Huấn luyện Mô hình** tại cấu hình Sidebar.")
    else:
        metrics = st.session_state['metrics']
        model_name = st.session_state['model_name']
        
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Độ chính xác (Accuracy Score)", f"{metrics['accuracy']:.4f}")
        col_r2.metric("Thuật toán đang chạy", model_name)
        col_r3.metric("Số lượng mẫu kiểm định (Test set)", f"{len(metrics['y_test'])} mẫu")
        
        st.write("### 🧮 Báo cáo chỉ số phân loại chi tiết (Classification Report)")
        report_df = pd.DataFrame(metrics['report']).transpose()
        st.dataframe(report_df.style.format(precision=4), use_container_width=True)
        
        st.write("### 🧩 Ma trận biểu đồ nhầm lẫn (Confusion Matrix)")
        cm = metrics['cm']
        cm_df = pd.DataFrame(cm, index=['Thực tế: 0', 'Thực tế: 1'], columns=['Dự báo: 0', 'Dự báo: 1'])
        
        col_cm1, col_cm2 = st.columns([1, 1])
        with col_cm1:
            st.dataframe(cm_df, use_container_width=True)
        with col_cm2:
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Hệ Thống Dự Báo", y="Nhãn Xác Thực Thực Tế", color="Số mẫu"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues',
                height=250
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ---- TAB 4: SỬ DỤNG MÔ HÌNH ----
with tab4:
    st.subheader("🔮 Thẩm định và dự báo rủi ro giao dịch trực tuyến")
    
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Tính năng dự báo yêu cầu mô hình phải được sinh ra từ việc huấn luyện trước. Vui lòng chạy mô hình tại Sidebar.")
    else:
        model = st.session_state['trained_model']
        
        predict_mode = st.radio(
            "Hình thức nạp thông tin đầu vào phục vụ dự báo:",
            options=["Chấm điểm trực tiếp bằng Form nhập", "Dự báo danh sách hàng loạt (Tải tệp tin X_new)"],
            horizontal=True
        )
        
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP
        if predict_mode == "Chấm điểm trực tiếp bằng Form nhập":
            st.write("✍️ *Nhập các thông số giao dịch cụ thể cần thẩm định:*")
            
            with st.form("single_prediction_form"):
                form_cols = st.columns(4)
                input_data = {}
                
                for idx, feat in enumerate(FEATURES):
                    col_idx = idx % 4
                    # Gợi ý mặc định lấy theo giá trị trung vị từ tập dữ liệu huấn luyện thô đã nạp
                    default_val = float(df_raw[feat].median()) if feat in df_raw.columns else 0.0
                    
                    with form_cols[col_idx]:
                        input_data[feat] = st.number_input(
                            f"Trường {feat}",
                            value=default_val,
                            format="%.6f"
                        )
                
                submit_pred = st.form_submit_button("🔍 Tiến hành Thẩm Định Hệ Thống", use_container_width=True)
                
                if submit_pred:
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    
                    st.write("---")
                    st.write("### 📊 Kết quả phân tích rủi ro độc lập:")
                    if prediction == 1:
                        st.error("⚠️ **CẢNH BÁO:** Phát hiện dấu hiệu rủi ro cao. Giao dịch được phân loại là **GIAN LẬN/NỢ XẤU** (default = 1)!")
                    else:
                        st.success("✅ **AN TOÀN:** Giao dịch vượt qua kiểm tra, được đánh giá là **BÌNH THƯỜNG** (default = 0).")
                        
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(input_df)[0]
                        fraud_prob = prob[1]
                        st.metric(label="Xác suất tính toán rủi ro gian lận", value=f"{fraud_prob * 100:.2f} %")
                        st.progress(float(fraud_prob))

        # CHẾ ĐỘ 2: BATCH PREDICTION (DỰ BÁO HÀNG LOẠT)
        elif predict_mode == "Dự báo danh sách hàng loạt (Tải tệp tin X_new)":
            st.write("📁 *Tải lên tệp chứa cấu trúc thông tin định dạng bao gồm đầy đủ các trường từ `X_1` đến `X_14`:*")
            
            new_file = st.file_uploader(
                "Tải tệp danh sách khách hàng mới cần chấm điểm", 
                type=["csv", "xlsx"],
                key="batch_prediction_uploader"
            )
            
            if new_file is not None:
                df_new = load_data(new_file.getvalue(), new_file.name)
                if df_new is not None:
                    missing_batch_cols = [col for col in FEATURES if col not in df_new.columns]
                    
                    if missing_batch_cols:
                        st.error(f"❌ Không thể phân tích dữ liệu hàng loạt. Tệp tải lên thiếu các cột bắt buộc sau: {missing_batch_cols}")
                    else:
                        X_new = df_new[FEATURES]
                        
                        # Chạy dự báo hàng loạt dữ liệu mới
                        batch_predictions = model.predict(X_new)
                        df_result = df_new.copy()
                        df_result['Predicted_Default'] = batch_predictions
                        
                        if hasattr(model, "predict_proba"):
                            df_result['Fraud_Probability'] = model.predict_proba(X_new)[:, 1]
                        
                        st.success("🎉 Đã hoàn thành quá trình chấm điểm tự động cho toàn bộ danh sách!")
                        
                        num_fraud = int((batch_predictions == 1).sum())
                        st.warning(f"🔎 Kết quả kiểm tra: Phát hiện tổng số **{num_fraud}** ca rủi ro trên tổng số **{len(df_result)}** giao dịch.")
                        
                        st.write("**Bảng chi tiết danh sách kết quả chấm điểm:**")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Kết xuất file CSV mã hóa utf-8-sig để tải xuống máy tính cá nhân
                        csv_data = df_result.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống bảng dữ liệu kết quả phân tích (.CSV)",
                            data=csv_data,
                            file_name="Ket_qua_du_bao_gian_lan_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
