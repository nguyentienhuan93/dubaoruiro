import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------
# STEP 1: PAGE CONFIGURATION (Lệnh đầu tiên của ứng dụng)
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
    Hàm nạp dữ liệu dùng chung. Nhận bytes để có thể băm (hashable) phục vụ việc cache.
    Hỗ trợ cả định dạng CSV và Excel dựa trên phần mở rộng tệp.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_bytes)
        else:
            st.error("Định dạng tệp không được hỗ trợ! Vui lòng tải lên tệp .csv hoặc .xlsx")
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp dữ liệu: {e}")
        return None

# Định nghĩa danh sách biến đầu vào chuẩn từ mô hình trong notebook
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# -----------------------------------------------------------------------------
# STEP 3: SIDEBAR — VÙNG CẤU HÌNH & ĐIỀU KHIỂN
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải tệp dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các biến đặc trưng từ X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.divider()
    
    st.subheader("🤖 Cấu hình Mô hình AI")
    # Lựa chọn mô hình dựa trên các mô hình xuất hiện trong notebook (Model 3 là RandomForest)
    model_choice = st.selectbox(
        "Chọn thuật toán",
        options=["Random Forest Classifier", "Decision Tree Classifier", "Logistic Regression"],
        index=0,
        help="Chọn thuật toán học máy phù hợp để huấn luyện phát hiện gian lận."
    )
    
    # Hiển thị tham số động theo từng loại mô hình
    if model_choice == "Random Forest Classifier":
        n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, help="Độ sâu lớn nhất của các cây quyết định.")
        criterion = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"], index=0, help="Hàm đo lường chất lượng phân tách.")
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
        
    elif model_choice == "Decision Tree Classifier":
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=5, help="Độ sâu lớn nhất của cây quyết định.")
        criterion = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"], index=0, help="Hàm đo lường chất lượng phân tách.")
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
        
    elif model_choice == "Logistic Regression":
        penalty = st.selectbox("Phương thức chuẩn hóa (penalty)", options=["l2", "none"], index=0)
        C_val = st.slider("Hệ số nghịch đảo chuẩn hóa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=2000, value=1000, step=100)
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)

    st.divider()
    
    # Nút bấm kích hoạt huấn luyện duy nhất
    train_clicked = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# STEP 4: HEADER — VÙNG ĐỊNH HƯỚNG ỨNG DỤNG
# -----------------------------------------------------------------------------
st.title("🛡️ Ứng Dụng Phát Hiện Giao Dịch Gian Lận")
st.caption("Ứng dụng hỗ trợ phân tích rủi ro tín dụng và tự động phát hiện các giao dịch có dấu hiệu gian lận (default) dựa trên mô hình Học máy.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu huấn luyện mẫu (.csv hoặc .xlsx) tại Sidebar bên trái để bắt đầu khám phá ứng dụng.")
    st.stop()
else:
    # Đọc dữ liệu thô qua hàm cache_data
    df_raw = load_data(uploaded_file.getvalue(), uploaded_file.name)
    if df_raw is None:
        st.stop()
    st.caption(f"📁 Đang sử dụng tệp dữ liệu: `{uploaded_file.name}`")

st.divider()

# -----------------------------------------------------------------------------
# PROCESS: KHỐI HUẤN LUYỆN MÔ HÌNH (Lưu vào st.session_state)
# -----------------------------------------------------------------------------
if train_clicked:
    # Kiểm tra tính hợp lệ của schema dữ liệu huấn luyện
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc: {missing_cols}")
    else:
        with st.spinner("⏳ Đang tiến hành huấn luyện mô hình học máy... Vui lòng đợi trong giây lát."):
            X = df_raw[FEATURES]
            y = df_raw[TARGET]
            
            # Tách tập dữ liệu huấn luyện/kiểm tra đồng bộ theo tỷ lệ của bài toán
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(random_state))
            
            # Khởi tạo mô hình theo lựa chọn tại Sidebar
            if model_choice == "Random Forest Classifier":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=int(random_state))
            elif model_choice == "Decision Tree Classifier":
                model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=int(random_state))
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(penalty=penalty, C=C_val, max_iter=int(max_iter), random_state=int(random_state))
            
            # Fit mô hình
            model.fit(X_train, y_train)
            
            # Dự đoán thu thập chỉ số kiểm định
            y_pred = model.predict(X_test)
            
            # Lưu trữ toàn bộ kết quả vào session_state để tái sử dụng xuyên suốt giữa các Tab
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
# STEP 5: TABS CONTROL LAYOUT — KHU VỰC HIỂN THỊ NỘI DUNG & KẾT QUẢ
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện", 
    "🔮 Sử dụng mô hình"
])

# ---- TAB 1: TỔNG QUAN DỮ LIỆU ----
with tab1:
    st.subheader("📋 Phân tích Thống kê Dữ liệu Thô")
    
    # Hiển thị số lượng dòng, cột và dung lượng tệp
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Số lượng dòng (Rows)", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng cột (Columns)", f"{df_raw.shape[1]}")
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    col_m3.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
    
    st.write("**Xem trước 5 dòng dữ liệu đầu tiên (Head):**")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("**Thống kê mô tả các biến đặc trưng đưa vào mô hình (X & y):**")
    available_model_cols = [c for c in FEATURES + [TARGET] if c in df_raw.columns]
    if available_model_cols:
        st.dataframe(df_raw[available_model_cols].describe(), use_container_width=True)
    else:
        st.warning("Không tìm thấy các biến đặc trưng X_1 đến X_14 trong dữ liệu tải lên.")

# ---- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ----
with tab2:
    st.subheader("🎨 Biểu đồ phân phối các biến")
    
    # Trực quan hóa biến mục tiêu trước (nếu có)
    if TARGET in df_raw.columns:
        target_counts = df_raw[TARGET].value_counts().reset_index()
        target_counts.columns = [TARGET, 'Số lượng']
        target_counts[TARGET] = target_counts[TARGET].astype(str)
        
        fig_target = px.bar(
            target_counts, x=TARGET, y='Số lượng',
            title=f"Phân phối biến mục tiêu ({TARGET})",
            labels={TARGET: "Trạng thái (0: Bình thường, 1: Gian lận/Nợ xấu)"},
            color=TARGET, color_discrete_sequence=px.colors.qualitative.Set2,
            height=350
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.write("**Tùy chọn hiển thị các biến đặc trưng:**")
    # Cho phép người dùng tùy chọn hiển thị các biến cụ thể, mặc định lấy 4 biến đầu
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng cần vẽ biểu đồ phân phối:",
        options=FEATURES,
        default=FEATURES[:4]
    )
    
    if selected_features:
        # Bố trí biểu đồ lưới dạng 2x2 hoặc tương đương bằng cách chia cặp cột
        for i in range(0, len(selected_features), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(selected_features):
                    feat = selected_features[i + j]
                    if feat in df_raw.columns:
                        fig_hist = px.histogram(
                            df_raw, x=feat,
                            title=f"Phân phối tần suất của biến {feat}",
                            marginal="box", # Thêm biểu đồ hộp xem điểm ngoại lai
                            color_discrete_sequence=['#4A90E2'],
                            height=300
                        )
                        cols[j].plotly_chart(fig_hist, use_container_width=True)
                    else:
                        cols[j].warning(f"Cột `{feat}` không có trong dữ liệu.")
    else:
        st.info("Vui lòng chọn ít nhất một biến đặc trưng để hiển thị biểu đồ phân phối.")

# ---- TAB 3: KẾT QUẢ HUÂN LUYỆN & KIỂM ĐỊNH ----
with tab3:
    st.subheader("📊 Kết quả kiểm định chất lượng mô hình phân loại")
    
    # Khơi tạo điểm điều phối trạng thái rỗng
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Bạn chưa thực hiện huấn luyện mô hình. Vui lòng thiết lập tham số và nhấn nút **🚀 Huấn luyện Mô hình** tại Sidebar.")
    else:
        metrics = st.session_state['metrics']
        model_name = st.session_state['model_name']
        
        # Chỉ tiêu vô hướng dạng st.metric
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Độ chính xác tổng thể (Accuracy)", f"{metrics['accuracy']:.4f}")
        col_r2.metric("Mô hình đang kiểm định", model_name)
        col_r3.metric("Kích thước tập Test", f"{len(metrics['y_test'])} mẫu")
        
        st.write("### 🧮 Bảng báo cáo phân loại chi tiết (Classification Report)")
        report_df = pd.DataFrame(metrics['report']).transpose()
        st.dataframe(report_df.style.format(precision=4), use_container_width=True)
        
        st.write("### 🧩 Ma trận nhầm lẫn (Confusion Matrix)")
        cm = metrics['cm']
        cm_df = pd.DataFrame(cm, index=['Thực tế: 0', 'Thực tế: 1'], columns=['Dự báo: 0', 'Dự báo: 1'])
        
        col_cm1, col_cm2 = st.columns([1, 1])
        with col_cm1:
            st.dataframe(cm_df, use_container_width=True)
        with col_cm2:
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Báo", y="Nhãn Thực Tế", color="Số lượng mẫu"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues',
                height=250
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ---- TAB 4: SỬ DỤNG MÔ HÌNH ----
with tab4:
    st.subheader("🔮 Dự báo rủi ro gian lận trên dữ liệu mới")
    
    if 'trained_model' not in st.session_state:
        st.info("ℹ️ Vui lòng huấn luyện mô hình học máy thành công trước khi sử dụng tính năng dự báo rủi ro này.")
    else:
        model = st.session_state['trained_model']
        
        predict_mode = st.radio(
            "Phương thức nhập dữ liệu dự báo:",
            options=["Nhập thông số trực tiếp", "Tải tệp danh sách khách hàng mới (Batch Prediction)"],
            horizontal=True
        )
        
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP QUA FORM INPUT
        if predict_mode == "Nhập thông số trực tiếp":
            st.write("✍️ *Vui lòng điền thông số các biến đặc trưng của khách hàng giao dịch:*")
            
            with st.form("single_prediction_form"):
                form_cols = st.columns(4)
                input_data = {}
                
                # Tạo tự động các ô nhập thông số dựa trên các cột đặc trưng từ X_1 đến X_14
                for idx, feat in enumerate(FEATURES):
                    col_idx = idx % 4
                    # Lấy giá trị trung vị mặc định dựa trên dữ liệu thô đã nạp để tối ưu trải nghiệm người dùng
                    default_val = float(df_raw[feat].median()) if feat in df_raw.columns else 0.0
                    min_val = float(df_raw[feat].min()) if feat in df_raw.columns else -100.0
                    max_val = float(df_raw[feat].max()) if feat in df_raw.columns else 100.0
                    
                    with form_cols[col_idx]:
                        input_data[feat] = st.number_input(
                            f"Biến {feat}",
                            value=default_val,
                            format="%.6f",
                            help=f"Giá trị thực tế phân phối từ {min_val:.2f} đến {max_val:.2f}"
                        )
                
                submit_pred = st.form_submit_button("🔍 Tiến hành Dự Báo Rủi Ro", use_container_width=True)
                
                if submit_pred:
                    # Chuyển đổi dữ liệu sang định dạng DataFrame tương thích đầu vào
                    input_df = pd.DataFrame([input_data])
                    
                    # Tiến hành dự đoán nhãn rủi ro
                    prediction = model.predict(input_df)[0]
                    
                    # Tính xác suất phân loại nếu thuật toán mô hình có hỗ trợ
                    has_proba = hasattr(model, "predict_proba")
                    if has_proba:
                        prob = model.predict_proba(input_df)[0]
                        fraud_prob = prob[1]
                    
                    st.write("---")
                    st.write("### 📊 Kết quả phân tích rủi ro:")
                    if prediction == 1:
                        st.error("⚠️ **CẢNH BÁO:** Hệ thống phát hiện giao dịch này có **RỦI RO GIAN LẬN CAO** (default = 1)!")
                    else:
                        st.success("✅ **AN TOÀN:** Giao dịch được thẩm định là **BÌNH THƯỜNG / AN TOÀN** (default = 0).")
                        
                    if has_proba:
                        st.metric(label="Xác suất xảy ra gian lận (Default Probability)", value=f"{fraud_prob * 100:.2f} %")
                        st.progress(float(fraud_prob))

        # CHẾ ĐỘ 2: BATCH PREDICTION TỪ FILE XLSX/CSV NGOÀI
        elif predict_mode == "Tải tệp danh sách khách hàng mới (Batch Prediction)":
            st.write("📁 *Vui lòng tải lên tệp chứa cấu trúc thông tin đặc trưng (yêu cầu đầy đủ các cột từ `X_1` đến `X_14`):*")
            
            new_file = st.file_uploader(
                "Tải danh sách khách hàng mới cần chấm điểm", 
                type=["csv", "xlsx"],
                key="batch_uploader"
            )
            
            if new_file is not None:
                df_new = load_data(new_file.getvalue(), new_file.name)
                if df_new is not None:
                    # Kiểm tra độ khớp cấu trúc Schema đầu vào
                    missing_batch_cols = [col for col in FEATURES if col not in df_new.columns]
                    
                    if missing_batch_cols:
                        st.error(f"❌ Không thể dự báo hành loạt. Tệp tải lên thiếu các cột biến đặc trưng bắt buộc sau: {missing_batch_cols}")
                    else:
                        X_new = df_new[FEATURES]
                        
                        # Dự đoán hàng loạt
                        batch_predictions = model.predict(X_new)
                        df_result = df_new.copy()
                        df_result['Predicted_Default'] = batch_predictions
                        
                        if hasattr(model, "predict_proba"):
                            batch_prob = model.predict_proba(X_new)[:, 1]
                            df_result['Fraud_Probability'] = batch_prob
                        
                        st.success("🎉 Đã hoàn thành xử lý chấm điểm và dự báo tự động cho toàn bộ danh sách khách hàng!")
                        
                        # Thống kê nhanh số ca rủi ro
                        num_fraud = int((batch_predictions == 1).sum())
                        st.warning(f"🔎 Phát hiện tổng số **{num_fraud}** trường hợp rủi ro gian lận trong tổng số **{len(df_result)}** giao dịch được chấm điểm.")
                        
                        st.write("**Bảng kết quả dự đoán chi tiết:**")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Nút xuất dữ liệu ra tệp CSV để người dùng lưu trữ về máy tính nội bộ
                        csv_data = df_result.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name="Ket_qua_du_bao_gian_lan_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
