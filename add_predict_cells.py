"""Add 5-sample prediction cell to LB6 notebook after cell 34 (actual vs pred plot)"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NEW_CELL_MD_SOURCE = """\
## 📌 หัวข้อที่ 14b: Predict จริง 5 ตัวอย่าง (5 Sample Predictions)

แสดงการพยากรณ์ของโมเดลทั้ง 4 บน **ตัวอย่างอิสระ 5 ชุด** จาก Test Set:
- แต่ละตัวอย่าง: ใช้ window 24 ชั่วโมงที่ผ่านมา → ทำนายอุณหภูมิชั่วโมงถัดไป
- แสดงค่าจริง (Actual) และค่าที่ทำนายได้จากแต่ละโมเดล
"""

NEW_CELL_CODE_SOURCE = """\
# ============================================================
# Predict จริง 5 ตัวอย่าง จาก Test Set
# ============================================================
import math

# เลือก 5 ตัวอย่าง (index) จาก test set แบบกระจาย
n_test = len(actual)
sample_indices = [int(n_test * r) for r in [0.05, 0.20, 0.40, 0.60, 0.80]]

fig, axes = plt.subplots(5, 1, figsize=(14, 18))
fig.suptitle("🔮 5 Sample Predictions — LSTM / LSTM+Attention / Transformer / N-BEATS",
             fontsize=14, fontweight="bold", y=1.01)

model_colors = {"LSTM": "#FF6B6B", "LSTM+Attention": "#4ECDC4",
                "Transformer": "#45B7D1", "N-BEATS": "#F7B801"}

for ax_i, (ax, idx) in enumerate(zip(axes, sample_indices)):
    # แสดง window 24 ชั่วโมงก่อนหน้า + จุดที่ทำนาย
    window_size = 24
    # Denormalize: value = norm * std + mean
    context_norm = actual[max(0, idx-window_size):idx]  # context ที่ผ่านมา
    context_real = [v * TRAIN_STD + TRAIN_MEAN for v in context_norm]
    actual_real  = actual[idx] * TRAIN_STD + TRAIN_MEAN

    x_context = list(range(len(context_real)))
    x_pred    = len(context_real)  # ตำแหน่งที่ทำนาย

    # Plot context (อุณหภูมิที่ผ่านมา)
    ax.plot(x_context, context_real, "k-o", ms=3, lw=1.2, label="Context (24h)", alpha=0.6)

    # Plot actual ที่จุดทำนาย
    ax.plot(x_pred, actual_real, "k*", ms=12, label=f"Actual: {actual_real:.2f}°C", zorder=5)

    # Plot prediction ของแต่ละโมเดล
    for mname, color in model_colors.items():
        if mname in preds and idx < len(preds[mname]):
            pred_norm = preds[mname][idx]
            pred_real = pred_norm * TRAIN_STD + TRAIN_MEAN
            ax.plot(x_pred, pred_real, "D", ms=9, color=color,
                    label=f"{mname}: {pred_real:.2f}°C", zorder=5)

    ax.axvline(x=x_pred - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Sample {ax_i+1} (Test idx={idx})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hour"); ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sample_predictions_5.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ บันทึกภาพ: sample_predictions_5.png")
"""

with open('e:/LB6/lab6_timeseries_029.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 34 (actual vs pred) and insert AFTER it
# First verify cell 34 is the right one
src34 = nb['cells'][34].get('source', '')
if isinstance(src34, list):
    src34 = ''.join(src34)

print(f"Cell 34 starts with: {src34[:60]}")

new_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": NEW_CELL_MD_SOURCE
}
new_code_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": NEW_CELL_CODE_SOURCE,
    "outputs": [],
    "execution_count": None
}

# Insert after cell 34
nb['cells'].insert(35, new_code_cell)
nb['cells'].insert(35, new_md_cell)

with open('e:/LB6/lab6_timeseries_029.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done! Total cells now: {len(nb['cells'])}")
print("Added: Cell 35 (markdown) + Cell 36 (code) — 5 sample predictions")
