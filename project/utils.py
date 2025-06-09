import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc file đã xử lý
df = pd.read_csv(r'C:\Users\Pham Thanh\Downloads\Timesseries\cholera_processed.csv')

# Chuyển 'day' thành kiểu datetime nếu cần
df['day'] = pd.to_datetime(df['day'])

# Sắp xếp lại nếu cần
df = df.sort_values('day').reset_index(drop=True)

# Chuẩn hóa lại cột count nếu chưa có count_scaled
if 'count_scaled' not in df.columns:
    scaler = StandardScaler()
    df['count_scaled'] = scaler.fit_transform(df[['count']])

# Hàm tiền xử lý đầu vào
def preprocess_input(disease, year, month):
    # Lọc dữ liệu của tháng và năm đó
    df_month = df[(df['day'].dt.year == year) & (df['day'].dt.month == month)]
    df_month = df_month.sort_values('day')

    if len(df_month) == 0:
        raise ValueError(f"Không có dữ liệu cho tháng {month}/{year}.")

    # Lấy ngày cuối cùng có dữ liệu trong tháng
    last_day = df_month['day'].max()

    # Lấy dữ liệu 30 ngày liên tiếp tính từ last_day ngược về trước
    start_day = last_day - pd.Timedelta(days=29)

    df_30days = df[(df['day'] >= start_day) & (df['day'] <= last_day)].sort_values('day')

    if len(df_30days) < 30:
        raise ValueError(f"Không đủ dữ liệu 30 ngày liên tiếp tính từ ngày {last_day.date()}.")

    last_30 = df_30days['count_scaled'].values
    raw_counts = df_30days['count'].values

    input_tensor = torch.tensor(last_30, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, 30, 1)

    return input_tensor, raw_counts


# Hàm dự đoán nhãn phân lớp
def predict_trend(model, input_tensor):
    """
    Dự đoán nhãn và xác suất từ chuỗi thời gian đầu vào.
    """
    input_tensor = input_tensor.float()
    with torch.no_grad():
        output = model(input_tensor)  # (1, num_classes)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs.cpu().numpy()
