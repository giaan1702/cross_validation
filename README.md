# Bài toán phân loại văn bản (Text Classification)

Dự án phân loại tiêu đề tin tức tiếng Việt theo **nhãn chủ đề** bằng mô hình ML cổ điển: **TF-IDF + LinearSVC/LogisticRegression/MultinomialNB** (scikit-learn).

## Dữ liệu

- `data_train.csv`: tập huấn luyện  
- `data_test.csv`: tập kiểm tra  

Mỗi file có 2 cột chính:

- `title`: văn bản (tiêu đề)
- `category`: nhãn lớp

## Cài đặt môi trường

Khuyến nghị dùng Python 3.10+.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install pandas scikit-learn matplotlib joblib
```

## Huấn luyện (Training)

Mở và chạy notebook `training.ipynb`.

Notebook sẽ:

- Đọc `data_train.csv`
- Dò tham số bằng `GridSearchCV` với `StratifiedKFold`
- Chọn mô hình tốt nhất theo **Macro F1**
- Lưu mô hình ra file `best_model.pkl`

Kết quả mẫu (từ notebook):

- Best CV macro-F1 khoảng ~0.83 với `TfidfVectorizer(ngram_range=(1,2)) + LinearSVC(C=1)`

## Đánh giá (Testing)

Mở và chạy notebook `testing.ipynb`.

Notebook sẽ:

- Đọc `data_test.csv`
- Load `best_model.pkl`
- Dự đoán và tính **Macro F1** + **Confusion Matrix**

## Cấu trúc thư mục

- `training.ipynb`: pipeline huấn luyện + chọn mô hình + lưu model
- `testing.ipynb`: pipeline test + đánh giá
- `data_train.csv`, `data_test.csv`: dữ liệu
- `best_model.pkl`: model đã huấn luyện (được tạo sau khi chạy training)

## Ghi chú

- Nếu bạn đổi tên cột dữ liệu, cập nhật `text_col` và `label_col` trong các notebook.
- `best_model.pkl` là file sinh ra sau khi train; nếu chưa có, hãy chạy `training.ipynb` trước khi chạy `testing.ipynb`.
