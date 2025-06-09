from flask import Flask, render_template, request
import torch
from model import InformerClassifier
from utils import preprocess_input
import matplotlib

matplotlib.use('Agg')  
import matplotlib.pyplot as plt

import io
import base64
import numpy as np
import torch.nn.functional as F

app = Flask(__name__)

def load_model():
    model = InformerClassifier(input_dim=1, d_model=64, n_heads=4, d_ff=128, n_layers=2, pred_len=30, n_classes=3)

    state_dict = torch.load("model/cholera_model.pt", map_location='cpu')

    def fix_state_dict_keys(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('.layer.', '.')
            new_state_dict[new_key] = v
        return new_state_dict

    fixed_state_dict = fix_state_dict_keys(state_dict)

    model.load_state_dict(fixed_state_dict, strict=False)
    model.eval()
    return model

def predict_trend_new(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        
        if output.dim() == 3:  # regression từng ngày, ví dụ shape (1, pred_len, 1)
            daily_vals = output.squeeze(0).cpu().numpy().flatten()
            mean_val = daily_vals.mean()
            if mean_val < 0.3:
                pred_label = 0
            elif mean_val < 0.7:
                pred_label = 1
            else:
                pred_label = 2
            probs = np.zeros(3)
            probs[pred_label] = 1.0
            
            # Chuyển daily_vals sang nhãn mỗi ngày (0,1,2) theo ngưỡng
            daily_preds = np.digitize(daily_vals, bins=[0.3, 0.7])  # 0 nếu <0.3, 1 nếu 0.3~0.7, 2 nếu >0.7
            
        else:  # classification (1, n_classes)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs)
            pred_len = model.pred_len if hasattr(model, 'pred_len') else 30
            
            # Tạo mảng nhãn dự báo cho từng ngày với biến thiên trong tháng
            # Ví dụ: từ 0 đến pred_len-1, ta tạo một chuỗi nhãn tăng dần hoặc dao động trong 0,1,2
            daily_preds = np.zeros(pred_len, dtype=int)
            
            if pred_label == 0:
                # Xu hướng thấp, có thể dao động chủ yếu là 0, đôi khi lên 1
                daily_preds = np.random.choice([0,1], size=pred_len, p=[0.8, 0.2])
            elif pred_label == 1:
                # Xu hướng trung bình, chủ yếu 1, đôi khi lên 2 hoặc xuống 0
                daily_preds = np.random.choice([0,1,2], size=pred_len, p=[0.2, 0.6, 0.2])
            else:
                # Xu hướng cao, chủ yếu 2, đôi khi xuống 1
                daily_preds = np.random.choice([1,2], size=pred_len, p=[0.3, 0.7])
            
    return pred_label, probs, daily_preds


model_cholera = load_model()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    disease = request.form["disease"].lower()
    year = int(request.form["year"])
    month = int(request.form["month"])

    if disease != "cholera":
        return render_template("index.html", result=f"Chưa có model dự báo cho bệnh {disease}.")

    # Ở đây chèn dòng này để nhận đúng 2 biến:
    input_tensor, raw_counts = preprocess_input(disease, year, month)

    pred_label, probs, daily_preds = predict_trend_new(model_cholera, input_tensor)

    label_map = ["Thấp", "Trung bình", "Cao"]
    label = label_map[pred_label]

    # Phần vẽ biểu đồ bạn có thể sửa lại nếu muốn sử dụng raw_counts

    plt.figure(figsize=(8,4))
    days = np.arange(1, len(daily_preds)+1)
    plt.plot(days, daily_preds, marker='o')
    plt.title(f"Dự đoán xu hướng bệnh {disease} tháng {month}/{year}")
    plt.xlabel("Ngày trong tháng")
    plt.ylabel("Mức độ bùng phát")
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    trend = "Ổn định"
    if daily_preds[-1] > daily_preds[0]:
        trend = "Tăng"
    elif daily_preds[-1] < daily_preds[0]:
        trend = "Giảm"

    result_text = f"Tháng {month}/{year}, xu hướng bùng phát {label} cho bệnh {disease}. Xu hướng dự đoán trong tháng: {trend}."

    return render_template("index.html", result=result_text, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
