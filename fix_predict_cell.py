"""Fix prediction cell - actual and preds are already in °C (denormalized)"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# The correct code for 5-sample predictions
# actual = numpy array of real °C values (test set targets)
# preds[name] = numpy array of predicted °C values (test set)
# Both are already denormalized

NEW_CODE = """\
# ============================================================
# Predict จริง 5 ตัวอย่าง จาก Test Set
# แสดงผล context 24 ชั่วโมงที่ผ่านมา + ค่าจริง vs ค่าที่แต่ละโมเดลทำนาย
# ============================================================

# เลือก 5 ตัวอย่าง (index) จาก test set แบบกระจาย
n_test = len(actual)
sample_indices = [int(n_test * r) for r in [0.05, 0.20, 0.40, 0.60, 0.80]]

# ดึง test X (normalized) สำหรับแสดง context
X_test_np, _ = data24["test"]
X_test_np = X_test_np.cpu().numpy()  # shape: (N, window, 1)

model_colors = {"LSTM": "#FF6B6B", "LSTM+Attention": "#4ECDC4",
                "Transformer": "#45B7D1", "N-BEATS": "#F7B801"}

fig, axes = plt.subplots(5, 1, figsize=(14, 18))
fig.suptitle("5 Sample Predictions — LSTM / LSTM+Attention / Transformer / N-BEATS",
             fontsize=14, fontweight="bold")

for ax_i, (ax, idx) in enumerate(zip(axes, sample_indices)):
    # Context: 24 ชั่วโมงที่ผ่านมา (denormalize กลับเป็น °C)
    context_norm = X_test_np[idx, :, 0]           # shape: (24,)
    context_real = context_norm * STD + MEAN        # °C จริง

    x_context = list(range(len(context_real)))      # 0..23
    x_pred    = len(context_real)                   # 24 (จุดทำนาย)

    # Plot context
    ax.plot(x_context, context_real, "k-o", ms=3, lw=1.2,
            label="Context (24h input)", alpha=0.65)

    # Plot actual ที่จุดทำนาย
    actual_val = actual[idx]   # °C จริง (denormalized แล้ว)
    ax.plot(x_pred, actual_val, "k*", ms=14, zorder=6,
            label=f"Actual: {actual_val:.2f}°C")

    # Plot prediction ของแต่ละโมเดล
    for mname, color in model_colors.items():
        if mname in preds and idx < len(preds[mname]):
            pred_val = preds[mname][idx]   # °C จริง (denormalized แล้ว)
            err = abs(pred_val - actual_val)
            ax.plot(x_pred, pred_val, "D", ms=10, color=color, zorder=5,
                    label=f"{mname}: {pred_val:.2f}°C (err={err:.2f})")

    # เส้นแบ่ง context กับ prediction
    ax.axvline(x=x_pred - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Sample {ax_i+1}  (test index={idx})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sample_predictions_5.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sample_predictions_5.png")
"""

with open('e:/LB6/lab6_timeseries_029.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 36 is our prediction code cell
nb['cells'][36]['source'] = NEW_CODE

with open('e:/LB6/lab6_timeseries_029.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed prediction cell 36!")
print(f"Total cells: {len(nb['cells'])}")
