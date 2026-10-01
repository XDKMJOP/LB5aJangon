"""Add/enhance descriptions for topic cells 10-23 in LB5 notebook"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

DESCRIPTIONS = {
    18: """\
---
## 📌 หัวข้อที่ 10: SimpleRNN

**SimpleRNN** ประมวลผลข้อมูลตามลำดับ (Sequential) ทีละ time step แต่ละ step รับ input ปัจจุบัน + hidden state จาก step ก่อนหน้า

### โครงสร้าง:
```
Token[1] → RNN → h[1] ──┐
                          ↓
Token[2] → RNN → h[2] ──┐
                          ↓
Token[3] → RNN → h[3] ──→ Output
```

### สูตร:
```
h_t = tanh(W_x · x_t  +  W_h · h_{t-1}  +  b)
```

### ข้อดี:
- โครงสร้างเรียบง่าย, Parameters น้อย
- เรียนรู้ context ได้บ้าง

### ข้อเสีย:
- **Vanishing Gradient Problem** — gradient หายไปเมื่อ sequence ยาว
- จำ dependency ระยะไกลได้ไม่ดี (เช่น คำต้น sequence กับคำท้าย)

### ค่าที่ใช้:
- `rnn_units = 32` (จำนวน hidden units)
""",

    20: """\
---
## 📌 หัวข้อที่ 11: LSTM (Long Short-Term Memory)

**LSTM** แก้ปัญหา Vanishing Gradient ของ SimpleRNN โดยเพิ่ม **Memory Cell** และ **3 Gates**

### Gates ของ LSTM:
| Gate | สูตร | หน้าที่ |
|------|------|--------|
| **Forget Gate** (f) | σ(W_f · [h,x] + b_f) | ตัดสินใจลืมข้อมูลเก่าอะไร |
| **Input Gate** (i) | σ(W_i · [h,x] + b_i) | ตัดสินใจจำข้อมูลใหม่อะไร |
| **Output Gate** (o) | σ(W_o · [h,x] + b_o) | ตัดสินใจ output อะไร |

### การไหลของข้อมูล:
```
Cell State (c_t) ───────────────────────────────→ ดำเนินไปตลอด
              ↑ forget    ↑ input
              f_t × c_{t-1} + i_t × c̃_t
Hidden State (h_t) = o_t × tanh(c_t)
```

### ข้อดี vs SimpleRNN:
- จำ dependency ระยะไกลได้ดีกว่า (ข้อมูลใน Cell State ไม่ค่อยหาย)
- แก้ Vanishing Gradient ได้บางส่วน

### ข้อเสีย:
- Parameters มากกว่า SimpleRNN (~4×) ทำให้ train ช้ากว่า
""",

    22: """\
---
## 📌 หัวข้อที่ 12: GRU (Gated Recurrent Unit)

**GRU** เป็น simplified version ของ LSTM ที่มีแค่ **2 Gates** และไม่มี Cell State แยกต่างหาก

### Gates ของ GRU:
| Gate | สูตร | หน้าที่ |
|------|------|--------|
| **Reset Gate** (r) | σ(W_r · [h,x]) | ควบคุมว่าจะ "ลืม" hidden state เก่ามากแค่ไหน |
| **Update Gate** (z) | σ(W_z · [h,x]) | ควบคุมสัดส่วนระหว่าง hidden state เก่ากับใหม่ |

### สูตร:
```
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])     ← candidate hidden state
h_t  = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  ← hidden state ใหม่
```

### เปรียบเทียบ SimpleRNN / LSTM / GRU:
| | SimpleRNN | LSTM | GRU |
|--|-----------|------|-----|
| Gates | 0 | 3 | 2 |
| Parameters | น้อยสุด | มากสุด | กลาง |
| จำระยะไกล | ❌ | ✅ | ✅ |
| ความเร็ว Train | เร็ว | ช้า | กลาง |

### ค่าที่ใช้:
- `rnn_units = 32` (เท่ากับทุกโมเดล เพื่อเปรียบเทียบที่ยุติธรรม)
""",

    24: """\
---
## 📌 หัวข้อที่ 13: Dense Layer และ Output Layer

### Dense Layer (Fully Connected) — `Dense(128, activation='relu')`
- รับ output จาก RNN/LSTM/GRU (ขนาด 32) แปลงเป็น feature ขนาด 128
- **ReLU**: `f(x) = max(0, x)` — ตัดค่าลบออก ช่วยให้ network เรียนรู้ non-linear patterns
- Dropout(0.30) หลัง Dense — ปิด 30% ของ neurons แบบสุ่มระหว่าง train เพื่อป้องกัน Overfitting

### Output Layer — `Dense(5, activation='softmax')`
- 5 neurons ตรงกับ 5 class ของ BBC News
- **Softmax**: แปลง logits เป็น probability distribution (รวมกัน = 1.0)

```
Output: [0.05, 0.03, 0.80, 0.07, 0.05]
             ↑       ↑       ↑
          business  entertain  politics (โมเดลทำนาย politics)
```

### Architecture สรุป:
```
Embedding(10000, 64)
    ↓
RNN/LSTM/GRU (32 units)
    ↓
Dropout(0.20)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.30)
    ↓
Dense(5, Softmax) ← Output
```
""",

    25: """\
---
## 📌 หัวข้อที่ 14: Compile โมเดล

การ Compile กำหนดวิธีที่โมเดลจะ **เรียนรู้** และ **วัดผล**

| Parameter | ค่า | เหตุผล |
|-----------|-----|--------|
| **Optimizer** | Adam | Adaptive learning rate — ทำงานได้ดีกับ NLP tasks |
| **Learning Rate** | 0.001 | ค่า default ของ Adam ที่ให้ผลดีทั่วไป |
| **Loss Function** | `sparse_categorical_crossentropy` | ใช้เมื่อ label เป็นตัวเลขจำนวนเต็ม (0,1,2,3,4) |
| **Metrics** | `accuracy` | วัดความถูกต้องในการจำแนก |

### ทำไมใช้ sparse_categorical_crossentropy?
- **categorical_crossentropy** → ต้องการ one-hot labels เช่น [0, 0, 1, 0, 0]
- **sparse_categorical_crossentropy** → ใช้ integer labels เช่น 2
- เราใช้ integer labels → ต้องใช้ sparse version
""",

    27: """\
---
## 📌 หัวข้อที่ 15: Fit / Training (ฝึกสอนโมเดล)

ฝึกสอนทั้ง 3 โมเดล (SimpleRNN, LSTM, GRU) ด้วย parameters เดียวกัน:

| Parameter | ค่า | คำอธิบาย |
|-----------|-----|---------|
| **Epochs** | 15 | จำนวนรอบที่ผ่านข้อมูลทั้งหมด |
| **Batch Size** | 64 | จำนวนตัวอย่างต่อ 1 gradient update |
| **Validation** | Val set | Monitor overfitting ทุก epoch |

### ขั้นตอนใน 1 Epoch:
```
1. แบ่ง Training data เป็น batches (64 ตัวอย่างต่อ batch)
2. สำหรับแต่ละ batch:
   a. Forward pass: คำนวณ output
   b. คำนวณ loss (cross-entropy)
   c. Backward pass: คำนวณ gradients
   d. Adam: อัปเดต weights
3. Evaluate บน Validation set
4. แสดงผล loss/accuracy
```

### Callbacks:
- ⏱️ จับเวลา training แต่ละโมเดล — เปรียบเทียบว่า SimpleRNN, LSTM, GRU ใครเร็วกว่า
- บันทึก history ของ loss/accuracy ทุก epoch สำหรับ plot กราฟ
""",

    29: """\
---
## 📌 หัวข้อที่ 16: Accuracy / Loss Graphs

แสดงกราฟ 2 ชุด สำหรับทั้ง 3 โมเดล เพื่อวิเคราะห์กระบวนการ training:

### กราฟที่ 1: Epoch vs Accuracy
- **Training Accuracy** (เส้นทึบ) — accuracy บน training data
- **Validation Accuracy** (เส้นประ) — accuracy บน validation data
- ต้องการเห็น: ทั้งสองเส้นเพิ่มขึ้นและใกล้เคียงกัน

### กราฟที่ 2: Epoch vs Loss
- **Training Loss** — ควรลดลงเรื่อยๆ
- **Validation Loss** — ถ้าเพิ่มขึ้นในขณะที่ Training Loss ยังลด = **Overfitting**

### Pattern ที่ต้องสังเกต:
| Pattern | ลักษณะ | ความหมาย |
|---------|--------|---------|
| ✅ Good Fit | Train ≈ Val, ทั้งคู่สูง | โมเดลเรียนรู้ได้ดี |
| ⚠️ Overfitting | Train >> Val | โมเดลจำ training data มากเกิน |
| ⚠️ Underfitting | Train ต่ำ, Val ต่ำ | โมเดลเรียนรู้ไม่พอ |
""",

    31: """\
---
## 📌 หัวข้อที่ 17: ประเมินผลบน Test Set

ประเมินผลโมเดลทั้ง 3 บน **Test Set** ที่ไม่เคยเห็นระหว่าง training เลย

### ตาราง Summary:
| โมเดล | Train Acc | Val Acc | **Test Acc** | Training Time |
|-------|-----------|---------|------------|--------------|
| SimpleRNN | ? | ? | **?** | ? sec |
| LSTM | ? | ? | **?** | ? sec |
| GRU | ? | ? | **?** | ? sec |

### Metrics ที่วัด:
- **Test Accuracy** — % ที่ทำนายถูกต้องบน test set (เกณฑ์หลัก)
- **Test Loss** — ค่า cross-entropy บน test set

### ทำไม Test Set สำคัญ?
- **Validation set** ถูกใช้ Monitor → โมเดลอาจ overfit ทาง validation ด้วย
- **Test set** ไม่เคยถูกใช้เลย → บอก performance ที่แท้จริงในโลกจริง
""",

    33: """\
---
## 📌 หัวข้อที่ 18: Confusion Matrix

**Confusion Matrix** แสดงว่าโมเดลทำนายแต่ละ class ถูก/ผิดอย่างไร

### โครงสร้าง (5×5 สำหรับ BBC News):
```
                 Predicted
                 bus  ent  pol  spo  tec
         bus  [ [TN,   e,   e,   e,  e ],  ← Actual business
Actual   ent  [  e,  TN,   e,   e,  e ],  ← Actual entertainment
         pol  [  e,   e,  TN,   e,  e ],  ← Actual politics
         spo  [  e,   e,   e,  TN,  e ],  ← Actual sport
         tec  [  e,   e,   e,   e, TN ] ] ← Actual tech
```
- **แนวทแยง** = ทำนายถูก (True Positives)
- **นอกแนวทแยง** = ทำนายผิด (False Positives/Negatives)

### วิธีอ่าน:
- แถว = class จริง (Actual), คอลัมน์ = class ที่ทำนาย (Predicted)
- ถ้า row "sport" column "entertainment" มีค่าสูง → โมเดล classify sport เป็น entertainment บ่อย
""",

    35: """\
---
## 📌 หัวข้อที่ 19: Classification Report

แสดง Precision, Recall, F1-score ของแต่ละ class อย่างละเอียด

### นิยามของแต่ละ Metric:
| Metric | สูตร | ความหมาย |
|--------|------|---------|
| **Precision** | TP / (TP + FP) | ในจำนวนที่ทำนายว่าเป็น class นี้ มีกี่% ที่ถูก |
| **Recall** | TP / (TP + FN) | ในจำนวน class นี้จริงๆ โมเดลจับได้กี่% |
| **F1-score** | 2×P×R / (P+R) | ค่าเฉลี่ยฮาร์โมนิกของ Precision กับ Recall |
| **Support** | จำนวน | ตัวอย่างจริงในแต่ละ class |

### ตัวอย่างการอ่านผล:
```
              precision  recall  f1-score  support
    business      0.95    0.94      0.95       76
 entertainmt      0.92    0.90      0.91       58
    politics      0.88    0.91      0.89       62
       sport      0.98    0.97      0.97       77
        tech      0.94    0.95      0.94       60
```

### เมื่อไหรที่สำคัญ:
- **Class imbalance** → ดู F1 ดีกว่า Accuracy
- **False Negatives แพง** (เช่น โรค) → เน้น Recall
- **False Positives แพง** (เช่น spam filter) → เน้น Precision
""",

    37: """\
---
## 📌 หัวข้อที่ 20: Predict ข้อความใหม่ (5 ตัวอย่าง)

ทดสอบโมเดลกับข้อความที่สร้างขึ้นใหม่ (ไม่ได้อยู่ใน dataset เดิม) เพื่อดู real-world performance

### ขั้นตอนการ Predict:
```
ข้อความใหม่
    ↓ 1. Clean text (lowercase, remove special chars)
    ↓ 2. Tokenize (แปลงคำเป็น ID ด้วย tokenizer ที่ train แล้ว)
    ↓ 3. Pad sequence (ให้ยาว MAX_LENGTH = 200)
    ↓ 4. ผ่านโมเดล → Softmax probabilities
    ↓ 5. argmax → predicted class
Predicted: "sport" (confidence: 97.3%)
```

### ตัวอย่างข้อความที่ใช้ทดสอบ:
1. **Business**: "Stock markets rose sharply after positive economic data..."
2. **Entertainment**: "The blockbuster film broke box office records this weekend..."
3. **Politics**: "Parliament voted to approve the new budget proposal..."
4. **Sport**: "England won the cricket series against Australia..."
5. **Tech**: "The new smartphone features an advanced AI chip..."

### Output ที่แสดง:
- Input Text
- Predicted Class & Confidence
- Probability ของทุก class (bar chart)
""",

    39: """\
---
## 📌 หัวข้อที่ 21: วิเคราะห์ Overfitting / Underfitting

วิเคราะห์ว่าแต่ละโมเดล (SimpleRNN, LSTM, GRU) อยู่ในสถานะใด:

### สถานะของโมเดล:
| สถานะ | Train Acc | Val Acc | สาเหตุ | แก้ไขโดย |
|------|-----------|---------|--------|---------|
| **Good Fit** ✅ | สูง | สูง ≈ Train | สมดุล | - |
| **Overfitting** ⚠️ | สูงมาก | ต่ำกว่า Train ชัดเจน | Model ซับซ้อนเกิน | เพิ่ม Dropout, Early Stopping, ลด units |
| **Underfitting** ⚠️ | ต่ำ | ต่ำ | Model ง่ายเกิน | เพิ่ม units, Epochs, ลด Dropout |

### แนวทางลด Overfitting:
1. **Dropout** — ปิด neurons แบบสุ่ม (ใช้แล้ว: 0.20 + 0.30)
2. **Early Stopping** — หยุด train เมื่อ val_loss ไม่ลดลง
3. **ลด model complexity** — ลด units หรือ layers
4. **เพิ่มข้อมูล** — Data Augmentation หรือ collect เพิ่ม
5. **L2 Regularization** — เพิ่ม penalty สำหรับ weights ที่ใหญ่มาก
""",

    41: """\
---
## 📌 หัวข้อที่ 22: Hyperparameter Experiment

ทดลองปรับ Hyperparameter อย่างน้อย 3 รูปแบบ แล้วเปรียบเทียบผลกับ Baseline

### การทดลอง (ใช้ GRU เป็น base):
| Experiment | สิ่งที่เปลี่ยน | ค่าใหม่ | เหตุผลที่ทดลอง |
|------------|---------------|--------|--------------|
| **Baseline** | - | Embed=64, Drop=0.20/0.30, LR=0.001 | ค่าเริ่มต้น |
| **Exp1: EmbDim128** | Embedding Dimension | 64 → 128 | embedding ใหญ่ขึ้น = แทนคำได้ละเอียดขึ้น |
| **Exp2: HighDropout** | Dropout rate | 0.20/0.30 → 0.40/0.50 | ลด overfitting มากขึ้น |
| **Exp3: LowLR** | Learning Rate | 0.001 → 0.0005 | เรียนรู้ช้าลง อาจ converge ดีขึ้น |

### สิ่งที่สังเกต:
- Embedding ใหญ่ขึ้น → แทนความหมายคำได้ดีขึ้น แต่ใช้ memory และเวลา train มากขึ้น
- Dropout สูงขึ้น → ลด overfitting แต่อาจทำให้ underfitting ถ้าสูงเกิน
- LR ต่ำลง → อาจต้องการ Epoch มากขึ้น แต่ converge เสถียรกว่า
""",

    43: """\
---
## 📌 หัวข้อที่ 23: เปรียบเทียบ SimpleRNN / LSTM / GRU — สรุปผล

### เปรียบเทียบโครงสร้าง:
| คุณสมบัติ | SimpleRNN | LSTM | GRU |
|-----------|-----------|------|-----|
| **Gates** | 0 | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Memory Cell** | ❌ | ✅ (แยก c_t กับ h_t) | ❌ (รวมใน h_t) |
| **Parameters** | น้อยสุด | มากสุด (~4× RNN) | กลาง (~3× RNN) |
| **Vanishing Gradient** | ❌ แย่ | ✅ ดี | ✅ ดี |
| **Training Speed** | เร็วสุด | ช้าสุด | กลาง |
| **Long-range Dep.** | ❌ | ✅ | ✅ |

### เมื่อไหรควรใช้อะไร:
- **SimpleRNN** → ข้อมูลลำดับสั้น, prototype เร็ว
- **LSTM** → ลำดับยาว, dependency ระยะไกลสำคัญ, มี GPU
- **GRU** → คล้าย LSTM แต่เร็วกว่า, ข้อมูลขนาดกลาง

### สรุปผลการทดลอง:
- ดู Test Accuracy ของทั้ง 3 โมเดล ว่าใครทำได้ดีที่สุด
- ดู Training Time ว่า GRU เร็วกว่า LSTM มากแค่ไหน
- วิเคราะห์ว่า BBC News เป็น task ที่ "ง่าย" พอสำหรับ SimpleRNN หรือต้องการ LSTM/GRU
""",
}

with open('e:/LB5/lab5_bbc_news_029.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

count = 0
for cell_idx, new_source in DESCRIPTIONS.items():
    cell = nb['cells'][cell_idx]
    old_src = cell.get('source', '')
    if isinstance(old_src, list):
        old_src = ''.join(old_src)
    
    if len(new_source) > len(old_src):
        cell['source'] = new_source
        count += 1
        topic_line = [l for l in new_source.split('\n') if 'หัวข้อที่' in l]
        print(f"Updated Cell {cell_idx}: {topic_line[0][:60] if topic_line else '?'}")
    else:
        print(f"Skipped Cell {cell_idx} (already longer or same)")

with open('e:/LB5/lab5_bbc_news_029.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nDone! Updated {count} cells.")
print(f"Total cells: {len(nb['cells'])}")
