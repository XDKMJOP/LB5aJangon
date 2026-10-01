# ============================================================
# ติดตั้ง Libraries ที่จำเป็น
# ============================================================

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# 🚀 SPEED OPTIMIZATION: ใช้ทรัพยากร CPU/GPU/RAM ให้เต็มที่
# ============================================================

# 1) ตั้งค่า GPU - ให้ใช้ memory แบบ dynamic (ไม่จองทั้งหมดตั้งแต่แรก)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU พร้อมใช้งาน: {len(gpus)} ตัว")
    for gpu in gpus:
        print(f"   - {gpu.name}")
else:
    print("⚠️ ไม่พบ GPU → ใช้ CPU")

# 2) ตั้งค่า CPU Threading - ใช้ทุก core ที่มี
num_cores = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
print(f"✅ CPU Cores: {num_cores} (ใช้ทั้งหมด)")

# 3) เปิด XLA (Accelerated Linear Algebra) Compiler - เร่งการคำนวณ
tf.config.optimizer.set_jit(True)
print("✅ XLA JIT Compiler: เปิดใช้งาน")

# 4) แสดงข้อมูลระบบ
print(f"\n📊 System Info:")
print(f"   TensorFlow version: {tf.__version__}")
print(f"   Python version: {os.sys.version}")
print(f"   RAM available: {os.popen('wmic OS get FreePhysicalMemory').read().strip().split()[-1] if os.name == 'nt' else 'N/A'} KB")


# ============================================================
# โหลด BBC News Dataset
# ============================================================

# ลองโหลดจาก HuggingFace datasets ก่อน
try:
    from datasets import load_dataset
    print("📥 กำลังโหลด BBC News จาก HuggingFace...")
    ds = load_dataset("SetFit/bbc-news")
    
    # รวม train + test เพื่อแบ่งเอง
    all_texts = ds['train']['text'] + ds['test']['text']
    all_labels_str = ds['train']['label_text'] + ds['test']['label_text']
    
    df = pd.DataFrame({'text': all_texts, 'category': all_labels_str})
    print("✅ โหลดจาก HuggingFace สำเร็จ!")
    
except Exception as e:
    print(f"⚠️ HuggingFace ไม่สำเร็จ: {e}")
    print("📥 ลองโหลดจาก URL สำรอง...")
    
    try:
        url = "https://raw.githubusercontent.com/sugarplum7/bbc-news/master/bbc-text.csv"
        df = pd.read_csv(url)
        print("✅ โหลดจาก GitHub สำเร็จ!")
    except:
        # Fallback: ใช้ URL อื่น
        url = "https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv"
        df = pd.read_csv(url)
        print("✅ โหลดจาก Google Storage สำเร็จ!")

# ตรวจสอบ columns - ปรับชื่อให้เป็นมาตรฐาน
if 'category' not in df.columns:
    # อาจมีชื่อ column ต่างกัน
    possible_cat_cols = [c for c in df.columns if c.lower() in ['category', 'label', 'label_text', 'class']]
    possible_text_cols = [c for c in df.columns if c.lower() in ['text', 'content', 'article']]
    if possible_cat_cols:
        df = df.rename(columns={possible_cat_cols[0]: 'category'})
    if possible_text_cols:
        df = df.rename(columns={possible_text_cols[0]: 'text'})

# สร้าง label mapping (แปลงจากชื่อหมวดเป็นตัวเลข)
# เรียงลำดับตัวอักษร: business=0, entertainment=1, politics=2, sport=3, tech=4
label_names = sorted(df['category'].unique())
label_to_id = {name: i for i, name in enumerate(label_names)}
id_to_label = {i: name for name, i in label_to_id.items()}

df['label'] = df['category'].map(label_to_id)

# ============================================================
# แสดงข้อมูลทั่วไปของ Dataset
# ============================================================
print("=" * 60)
print("📊 BBC News Dataset Summary")
print("=" * 60)
print(f"  Dataset Name       : BBC News")
print(f"  Number of Classes  : {len(label_names)}")
print(f"  Number of Samples  : {len(df)}")
print(f"  Classes            : {label_names}")
print(f"  Label Mapping      : {label_to_id}")
print(f"  Columns            : {list(df.columns)}")
print("=" * 60)
df.head()


# ============================================================
# แสดงจำนวนข้อมูลแต่ละ Class
# ============================================================
class_counts = df['category'].value_counts()
print("📊 จำนวนข้อมูลแต่ละ Class:")
print("-" * 40)
for cat, count in class_counts.items():
    print(f"  {cat:15s} : {count:5d} ({count/len(df)*100:.1f}%)")
print("-" * 40)
print(f"  {'Total':15s} : {len(df):5d}")

# แสดงกราฟ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('จำนวนข้อมูลแต่ละ Class', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontweight='bold')

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=[0.02]*5)
axes[1].set_title('สัดส่วนข้อมูลแต่ละ Class', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# ============================================================
# แสดงตัวอย่างข้อมูล 5 ตัวอย่าง (1 ตัวอย่างจากแต่ละ class)
# ============================================================
print("📝 ตัวอย่างข้อมูลจากแต่ละ Class:")
print("=" * 80)

for label_name in label_names:
    sample = df[df['category'] == label_name].iloc[0]
    text_preview = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
    print(f"\n🏷️  Class: {label_name.upper()} (Label: {label_to_id[label_name]})")
    print(f"   Text : {text_preview}")
    print("-" * 80)

# แสดงสถิติความยาวของ text
text_lengths = df['text'].apply(lambda x: len(x.split()))
print(f"\n📏 สถิติความยาว Text (จำนวนคำ):")
print(f"   Min    : {text_lengths.min()}")
print(f"   Max    : {text_lengths.max()}")
print(f"   Mean   : {text_lengths.mean():.1f}")
print(f"   Median : {text_lengths.median():.1f}")

# Histogram ความยาว
plt.figure(figsize=(10, 4))
plt.hist(text_lengths, bins=50, color='#45B7D1', edgecolor='black', alpha=0.7)
plt.axvline(x=200, color='red', linestyle='--', label='MAX_LENGTH=200')
plt.title('Distribution of Text Length (จำนวนคำ)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Text Cleaning: ทำความสะอาดข้อความ
# ============================================================

def clean_text(text):
    """
    ฟังก์ชันทำความสะอาดข้อความ:
    1. แปลงเป็นตัวพิมพ์เล็ก (lowercase)
    2. ลบ HTML tags
    3. ลบอักขระพิเศษ เก็บเฉพาะตัวอักษรและตัวเลข
    4. ลบ whitespace ที่ซ้ำ
    """
    text = text.lower()                           # แปลงเป็นตัวเล็ก
    text = re.sub(r'<[^>]+>', '', text)           # ลบ HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)   # ลบอักขระพิเศษ
    text = re.sub(r'\s+', ' ', text).strip()      # ลบช่องว่างซ้ำ
    return text

# ทำความสะอาดข้อความทั้งหมด
df['cleaned_text'] = df['text'].apply(clean_text)

# แสดงตัวอย่างก่อน-หลัง cleaning
print("🧹 ตัวอย่าง Text Cleaning:")
print("=" * 80)
for i in range(3):
    print(f"\n📌 ตัวอย่างที่ {i+1}:")
    print(f"  Before: {df['text'].iloc[i][:150]}...")
    print(f"  After : {df['cleaned_text'].iloc[i][:150]}...")
    print("-" * 80)


# ============================================================
# กำหนด Hyperparameters สำหรับ Tokenization
# ============================================================

# จำกัดจำนวนคำที่ใช้ใน Vocabulary (เลือกคำที่พบบ่อยที่สุด 10,000 คำ)
VOCAB_SIZE = 10000

# กำหนดความยาวสูงสุดของ Sequence (ตัด/เพิ่ม padding ให้ทุก sequence มีความยาวเท่ากัน)
MAX_LENGTH = 200

# กำหนดขนาด Embedding Dimension (แต่ละคำจะถูกแทนด้วย vector ขนาดนี้)
EMBEDDING_DIM = 64

# Batch size สำหรับ training (ยิ่งใหญ่ยิ่งเร็ว แต่ใช้ RAM มาก)
BATCH_SIZE = 64

# จำนวน Epochs
EPOCHS = 1

print(f"📋 Hyperparameters:")
print(f"   VOCAB_SIZE     = {VOCAB_SIZE}")
print(f"   MAX_LENGTH     = {MAX_LENGTH}")
print(f"   EMBEDDING_DIM  = {EMBEDDING_DIM}")
print(f"   BATCH_SIZE     = {BATCH_SIZE}")
print(f"   EPOCHS         = {EPOCHS}")


# ============================================================
# Train / Validation / Test Split (ต้องแบ่งก่อน Tokenization!)
# ============================================================
# ⚠️ สำคัญ: ต้องแบ่งข้อมูลก่อน fit tokenizer เพื่อป้องกัน data leakage
# Training 70% / Validation 15% / Testing 15%

texts = df['cleaned_text'].astype(str).tolist()
labels = np.array(df['label'].tolist(), dtype=int)

# แบ่งครั้งที่ 1: Train (70%) vs Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, random_state=42, stratify=labels
)

# แบ่งครั้งที่ 2: Temp → Validation (15%) + Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"📊 Data Split:")
print(f"   Training   : {len(X_train):,} samples ({len(X_train)/len(texts)*100:.1f}%)")
print(f"   Validation : {len(X_val):,} samples ({len(X_val)/len(texts)*100:.1f}%)")
print(f"   Testing    : {len(X_test):,} samples ({len(X_test)/len(texts)*100:.1f}%)")
print(f"   Total      : {len(texts):,} samples")

# ============================================================
# Tokenization: สร้าง Vocabulary และแปลง Text เป็น Sequence
# ============================================================
# ใช้ Keras Tokenizer - fit เฉพาะ training data เท่านั้น!
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE,    # จำกัดจำนวนคำใน vocabulary
    oov_token="<OOV>"        # token สำหรับคำที่ไม่รู้จัก
)

# fit tokenizer บน training data เท่านั้น (ป้องกัน data leakage)
tokenizer.fit_on_texts(X_train)

# แปลง text เป็น sequence ของตัวเลข
train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)

# แสดงตัวอย่าง tokenization
print(f"\n🔤 Vocabulary Info:")
print(f"   Total unique words : {len(tokenizer.word_index):,}")
print(f"   Vocabulary size    : {VOCAB_SIZE}")
print(f"   OOV token          : '<OOV>' (ID: {tokenizer.word_index.get('<OOV>', 'N/A')})")

print(f"\n📝 ตัวอย่าง Tokenization:")
for i in range(2):
    print(f"   Text    : {X_train[i][:80]}...")
    print(f"   Tokens  : {train_sequences[i][:15]}...")
    print()

# ============================================================
# Padding: ทำให้ทุก sequence มีความยาวเท่ากัน
# ============================================================
train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences,
    maxlen=MAX_LENGTH,      # ความยาวสูงสุด
    padding='post',         # เติม 0 ต่อท้าย
    truncating='post'       # ตัดส่วนท้ายถ้ายาวเกิน
)
val_padded = tf.keras.preprocessing.sequence.pad_sequences(
    val_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post'
)
test_padded = tf.keras.preprocessing.sequence.pad_sequences(
    test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post'
)

print(f"\n📐 Shape after Padding:")
print(f"   Train : {train_padded.shape}")
print(f"   Val   : {val_padded.shape}")
print(f"   Test  : {test_padded.shape}")

print(f"\n📝 ตัวอย่าง Padded Sequence:")
print(f"   {train_padded[0][:20]}... (แสดง 20 ตัวแรก)")


# ============================================================
# 🚀 สร้าง tf.data.Dataset เพื่อเร่งความเร็ว Training
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE  # ให้ TensorFlow กำหนดค่าที่เหมาะสมอัตโนมัติ

# Training dataset: shuffle + batch + cache + prefetch
train_dataset = tf.data.Dataset.from_tensor_slices((train_padded, y_train))
train_dataset = (train_dataset
    .shuffle(buffer_size=len(train_padded))  # สุ่มข้อมูลทุก epoch
    .batch(BATCH_SIZE)                        # แบ่งเป็น batch
    .cache()                                  # เก็บใน RAM
    .prefetch(AUTOTUNE)                       # โหลดล่วงหน้า
)

# Validation dataset: batch + cache + prefetch (ไม่ต้อง shuffle)
val_dataset = tf.data.Dataset.from_tensor_slices((val_padded, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

# Test dataset: batch + prefetch
test_dataset = tf.data.Dataset.from_tensor_slices((test_padded, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("✅ tf.data Pipeline สร้างเสร็จ!")
print(f"   Train batches : {len(train_dataset)}")
print(f"   Val batches   : {len(val_dataset)}")
print(f"   Test batches  : {len(test_dataset)}")
print(f"   Batch size    : {BATCH_SIZE}")
print(f"\n💡 Optimizations: shuffle + batch + cache + prefetch")


# ============================================================
# สร้างฟังก์ชันสำหรับ build model (ใช้ได้กับทั้ง SimpleRNN, LSTM, GRU)
# ============================================================
# จำนวน classes สำหรับ BBC News
NUM_CLASSES = len(label_names)  # 5 classes

def build_model(rnn_type='SimpleRNN', rnn_units=32, dense_units=128,
                vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM,
                max_len=MAX_LENGTH, num_classes=NUM_CLASSES,
                dropout1=0.20, dropout2=0.30, name=None):
    """
    สร้าง RNN model ตาม architecture ที่กำหนด (%4 = 1):
    
    Embedding → RNN Layer (32) → Dropout(0.20) → Dense(128, ReLU) → Dropout(0.30) → Dense(5, Softmax)
    
    Parameters:
    - rnn_type: 'SimpleRNN', 'LSTM', หรือ 'GRU'
    - rnn_units: จำนวน units ใน Recurrent Layer
    - dense_units: จำนวน units ใน Dense Layer
    """
    model = keras.Sequential(name=name or rnn_type)
    
    # 1) Embedding Layer: แปลง token IDs เป็น dense vectors
    model.add(layers.Embedding(
        input_dim=vocab_size,     # ขนาด vocabulary
        output_dim=embed_dim,     # ขนาด embedding vector
        input_length=max_len,     # ความยาว input sequence
        name='embedding'
    ))
    
    # 2) Recurrent Layer: ประมวลผล sequence
    if rnn_type == 'SimpleRNN':
        model.add(layers.SimpleRNN(rnn_units, name='simple_rnn'))
    elif rnn_type == 'LSTM':
        # LSTM: มี memory cell + 3 gates (forget, input, output)
        model.add(layers.LSTM(rnn_units, name='lstm'))
    elif rnn_type == 'GRU':
        # GRU: มี 2 gates (reset, update) — ง่ายกว่า LSTM
        model.add(layers.GRU(rnn_units, name='gru'))
    
    # 3) Dropout Layer 1: ป้องกัน overfitting (ปิด 20% ของ neurons)
    model.add(layers.Dropout(dropout1, name='dropout_1'))
    
    # 4) Dense Layer: fully connected layer สำหรับเรียนรู้ features
    model.add(layers.Dense(dense_units, activation='relu', name='dense'))
    
    # 5) Dropout Layer 2: ป้องกัน overfitting เพิ่มเติม (ปิด 30% ของ neurons)
    model.add(layers.Dropout(dropout2, name='dropout_2'))
    
    # 6) Output Layer: Softmax สำหรับ multi-class classification (5 classes)
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

# ============================================================
# สร้าง SimpleRNN Model
# ============================================================
model_rnn = build_model(rnn_type='SimpleRNN', rnn_units=32, dense_units=128, name='SimpleRNN_Model')
print("=" * 60)
print("📌 SimpleRNN Model Summary")
print("=" * 60)
model_rnn.summary()


# ============================================================
# สร้าง LSTM Model
# ============================================================
model_lstm = build_model(rnn_type='LSTM', rnn_units=32, dense_units=128, name='LSTM_Model')
print("=" * 60)
print("📌 LSTM Model Summary")
print("=" * 60)
model_lstm.summary()


# ============================================================
# สร้าง GRU Model
# ============================================================
model_gru = build_model(rnn_type='GRU', rnn_units=32, dense_units=128, name='GRU_Model')
print("=" * 60)
print("📌 GRU Model Summary")
print("=" * 60)
model_gru.summary()

# ============================================================
# เปรียบเทียบจำนวน Parameters ของทั้ง 3 โมเดล
# ============================================================
print("\n" + "=" * 60)
print("📊 เปรียบเทียบ Parameters")
print("=" * 60)
for name, model in [('SimpleRNN', model_rnn), ('LSTM', model_lstm), ('GRU', model_gru)]:
    trainable = model.count_params()
    print(f"  {name:12s} : {trainable:,} parameters")


# ============================================================
# Compile ทั้ง 3 โมเดล
# ============================================================
# ใช้ค่า Optimizer, Loss, Metric เหมือนกันทุกโมเดล เพื่อเปรียบเทียบได้อย่างยุติธรรม

LEARNING_RATE = 0.001

for name, model in [('SimpleRNN', model_rnn), ('LSTM', model_lstm), ('GRU', model_gru)]:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),  # Optimizer: Adam
        loss='sparse_categorical_crossentropy',  # Loss สำหรับ multi-class (label เป็น int)
        metrics=['accuracy']                     # วัดค่า accuracy
    )
    print(f"✅ {name} compiled — Optimizer: Adam(lr={LEARNING_RATE}), "
          f"Loss: sparse_categorical_crossentropy")

print(f"\n📋 Compile Settings:")
print(f"   Optimizer      : Adam")
print(f"   Learning Rate  : {LEARNING_RATE}")
print(f"   Loss Function  : sparse_categorical_crossentropy")
print(f"   Metrics        : accuracy")


# ============================================================
# 🚀 Train ทั้ง 3 โมเดล พร้อมจับเวลา
# ============================================================

# เก็บ history และเวลา training ของแต่ละโมเดล
histories = {}
train_times = {}

models_to_train = [
    ('SimpleRNN', model_rnn),
    ('LSTM', model_lstm),
    ('GRU', model_gru)
]

for name, model in models_to_train:
    print(f"\n{'='*60}")
    print(f"🏋️ Training {name}...")
    print(f"{'='*60}")
    
    # จับเวลา training
    start_time = time.time()
    
    # ฝึกสอนโมเดล
    history = model.fit(
        train_dataset,                    # ใช้ tf.data pipeline ที่ optimize แล้ว
        validation_data=val_dataset,      # ใช้ validation set
        epochs=EPOCHS,                    # จำนวนรอบ
        verbose=1                         # แสดง progress bar
    )
    
    # คำนวณเวลา
    elapsed = time.time() - start_time
    
    # เก็บผลลัพธ์
    histories[name] = history
    train_times[name] = elapsed
    
    print(f"\n⏱️ {name} Training Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

# ============================================================
# สรุปเวลา Training
# ============================================================
print(f"\n{'='*60}")
print(f"⏱️ สรุปเวลา Training")
print(f"{'='*60}")
for name, t in train_times.items():
    print(f"  {name:12s} : {t:.2f} sec ({t/60:.2f} min)")


# ============================================================
# 📈 วาดกราฟ Accuracy / Loss ของทั้ง 3 โมเดล
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
model_names = ['SimpleRNN', 'LSTM', 'GRU']
colors = {'train': '#FF6B6B', 'val': '#4ECDC4'}

for i, name in enumerate(model_names):
    h = histories[name].history
    epochs_range = range(1, len(h['accuracy']) + 1)
    
    # Accuracy Graph
    axes[0][i].plot(epochs_range, h['accuracy'], 'o-', color=colors['train'],
                    label='Training Accuracy', linewidth=2, markersize=4)
    axes[0][i].plot(epochs_range, h['val_accuracy'], 's-', color=colors['val'],
                    label='Validation Accuracy', linewidth=2, markersize=4)
    axes[0][i].set_title(f'{name} — Accuracy', fontsize=14, fontweight='bold')
    axes[0][i].set_xlabel('Epoch')
    axes[0][i].set_ylabel('Accuracy')
    axes[0][i].legend()
    axes[0][i].grid(True, alpha=0.3)
    axes[0][i].set_ylim([0, 1.05])
    
    # Loss Graph
    axes[1][i].plot(epochs_range, h['loss'], 'o-', color=colors['train'],
                    label='Training Loss', linewidth=2, markersize=4)
    axes[1][i].plot(epochs_range, h['val_loss'], 's-', color=colors['val'],
                    label='Validation Loss', linewidth=2, markersize=4)
    axes[1][i].set_title(f'{name} — Loss', fontsize=14, fontweight='bold')
    axes[1][i].set_xlabel('Epoch')
    axes[1][i].set_ylabel('Loss')
    axes[1][i].legend()
    axes[1][i].grid(True, alpha=0.3)

plt.suptitle('📊 Training & Validation — Accuracy / Loss', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# กราฟเปรียบเทียบทุกโมเดลในกราฟเดียว
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_colors = {'SimpleRNN': '#FF6B6B', 'LSTM': '#4ECDC4', 'GRU': '#45B7D1'}

for name in model_names:
    h = histories[name].history
    epochs_range = range(1, len(h['accuracy']) + 1)
    
    # Validation Accuracy
    axes[0].plot(epochs_range, h['val_accuracy'], 'o-', color=model_colors[name],
                 label=name, linewidth=2, markersize=4)
    
    # Validation Loss
    axes[1].plot(epochs_range, h['val_loss'], 'o-', color=model_colors[name],
                 label=name, linewidth=2, markersize=4)

axes[0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.suptitle('📊 Model Comparison — Validation Accuracy & Loss', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# ============================================================
# 📊 Evaluate ทั้ง 3 โมเดลบน Test Set
# ============================================================
results = {}

print("=" * 70)
print("📊 Evaluation Results on Test Set")
print("=" * 70)

for name, model in models_to_train:
    # Evaluate บน test set
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    
    h = histories[name].history
    train_acc = h['accuracy'][-1]
    val_acc = h['val_accuracy'][-1]
    train_loss = h['loss'][-1]
    val_loss = h['val_loss'][-1]
    
    results[name] = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'params': model.count_params(),
        'time': train_times[name]
    }
    
    print(f"\n🏷️ {name}:")
    print(f"   Training Accuracy   : {train_acc*100:.2f}%")
    print(f"   Validation Accuracy : {val_acc*100:.2f}%")
    print(f"   Test Accuracy       : {test_acc*100:.2f}%")
    print(f"   Training Loss       : {train_loss:.4f}")
    print(f"   Validation Loss     : {val_loss:.4f}")
    print(f"   Test Loss           : {test_loss:.4f}")
    print(f"   Parameters          : {model.count_params():,}")
    print(f"   Training Time       : {train_times[name]:.2f} sec")

# ============================================================
# สร้างตารางเปรียบเทียบ
# ============================================================
print(f"\n{'='*70}")
print("📋 Summary Table")
print(f"{'='*70}")
print(f"{'Model':<12} {'Params':>10} {'Time (s)':>10} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10}")
print("-" * 70)
for name in model_names:
    r = results[name]
    print(f"{name:<12} {r['params']:>10,} {r['time']:>10.2f} "
          f"{r['train_acc']*100:>9.2f}% {r['val_acc']*100:>9.2f}% {r['test_acc']*100:>9.2f}%")


# ============================================================
# 📊 Confusion Matrix สำหรับทั้ง 3 โมเดล
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for i, (name, model) in enumerate(models_to_train):
    # ทำนายผลบน test set
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # สร้าง confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # วาด heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=label_names, yticklabels=label_names)
    axes[i].set_title(f'{name} — Confusion Matrix', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.suptitle('📊 Confusion Matrix — Test Set', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# อธิบาย Confusion Matrix ของโมเดลที่ดีที่สุด
# ============================================================
# เลือกโมเดลที่ test accuracy สูงที่สุด
best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_model = dict(models_to_train)[best_model_name]

y_pred_probs = best_model.predict(test_dataset, verbose=0)
y_pred_best = np.argmax(y_pred_probs, axis=1)
cm_best = confusion_matrix(y_test, y_pred_best)

print(f"\n📌 วิเคราะห์ Confusion Matrix ของ {best_model_name} (โมเดลที่ดีที่สุด):")
print("=" * 60)
for i, class_name in enumerate(label_names):
    tp = cm_best[i, i]
    fp = cm_best[:, i].sum() - tp
    fn = cm_best[i, :].sum() - tp
    tn = cm_best.sum() - tp - fp - fn
    print(f"\n  🏷️ Class: {class_name}")
    print(f"     True Positive  (TP) = {tp:3d}  (ทำนายถูก ว่าเป็น {class_name})")
    print(f"     True Negative  (TN) = {tn:3d}  (ทำนายถูก ว่าไม่ใช่ {class_name})")
    print(f"     False Positive (FP) = {fp:3d}  (ทำนายผิด ว่าเป็น {class_name} แต่จริงๆ ไม่ใช่)")
    print(f"     False Negative (FN) = {fn:3d}  (ทำนายผิด ว่าไม่ใช่ {class_name} แต่จริงๆ ใช่)")


# ============================================================
# 📊 Classification Report สำหรับทั้ง 3 โมเดล
# ============================================================

for name, model in models_to_train:
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print(f"\n{'='*60}")
    print(f"📋 Classification Report — {name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))


# ============================================================
# 🔮 Predict ข้อความใหม่ 5 ข้อความ
# ============================================================

# สร้างข้อความใหม่ 5 ข้อความ (ครอบคลุมทุก class)
new_texts = [
    "The stock market saw a significant decline today as investors worried about inflation and rising interest rates affecting the economy.",
    "The new superhero movie broke box office records this weekend with millions of fans flocking to cinemas worldwide.",
    "The prime minister announced new policies on healthcare reform during the parliamentary debate yesterday.",
    "Manchester United secured a dramatic victory in the Champions League final with a last minute goal.",
    "Apple unveiled its latest artificial intelligence chip that promises to revolutionize smartphone computing power."
]

# ข้อความที่คาดหวัง (ground truth)
expected_classes = ['business', 'entertainment', 'politics', 'sport', 'tech']

def predict_texts(model, model_name, texts):
    """ทำนาย class ของข้อความใหม่"""
    # 1) ทำความสะอาดข้อความ
    cleaned = [clean_text(t) for t in texts]
    
    # 2) แปลงเป็น sequence
    sequences = tokenizer.texts_to_sequences(cleaned)
    
    # 3) Padding
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=MAX_LENGTH, padding='post', truncating='post'
    )
    
    # 4) ทำนาย
    predictions = model.predict(padded, verbose=0)
    
    print(f"\n{'='*70}")
    print(f"🔮 Predictions — {model_name}")
    print(f"{'='*70}")
    
    for i, (text, pred, expected) in enumerate(zip(texts, predictions, expected_classes)):
        pred_class_id = np.argmax(pred)
        pred_class = id_to_label[pred_class_id]
        confidence = pred[pred_class_id] * 100
        status = "✅" if pred_class == expected else "❌"
        
        print(f"\n  📝 Text {i+1}: {text[:80]}...")
        print(f"     Expected  : {expected}")
        print(f"     Predicted : {pred_class} {status}")
        print(f"     Confidence: {confidence:.2f}%")
        print(f"     Probs     : {', '.join([f'{label_names[j]}:{pred[j]*100:.1f}%' for j in range(len(label_names))])}")

# ใช้โมเดลที่ดีที่สุดในการ predict
predict_texts(best_model, best_model_name, new_texts)


# ============================================================
# 📊 วิเคราะห์ Overfitting / Underfitting
# ============================================================

print("=" * 70)
print("📊 Overfitting / Underfitting Analysis")
print("=" * 70)

for name in model_names:
    r = results[name]
    gap = r['train_acc'] - r['val_acc']
    loss_gap = r['val_loss'] - r['train_loss']
    
    print(f"\n🏷️ {name}:")
    print(f"   Training Accuracy   = {r['train_acc']*100:.2f}%")
    print(f"   Validation Accuracy = {r['val_acc']*100:.2f}%")
    print(f"   Gap (Train - Val)   = {gap*100:.2f}%")
    print(f"   Training Loss       = {r['train_loss']:.4f}")
    print(f"   Validation Loss     = {r['val_loss']:.4f}")
    print(f"   Loss Gap (Val-Train)= {loss_gap:.4f}")
    
    if gap > 0.10:
        print(f"   ⚠️ สถานะ: OVERFITTING — Train Acc สูงกว่า Val Acc มากกว่า 10%")
        print(f"   💡 แนวทางปรับปรุง: เพิ่ม Dropout, ใช้ Early Stopping, ลด model complexity")
    elif r['val_acc'] < 0.70:
        print(f"   ❌ สถานะ: UNDERFITTING — Validation Accuracy ต่ำกว่า 70%")
        print(f"   💡 แนวทางปรับปรุง: เพิ่ม epochs, เพิ่ม units, ใช้ LSTM/GRU")
    else:
        print(f"   ✅ สถานะ: GOOD FIT — โมเดลเรียนรู้ได้ดี")

# วาดกราฟ gap analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name in model_names:
    h = histories[name].history
    epochs_range = range(1, len(h['accuracy']) + 1)
    gap = [t - v for t, v in zip(h['accuracy'], h['val_accuracy'])]
    loss_gap = [v - t for t, v in zip(h['loss'], h['val_loss'])]
    
    axes[0].plot(epochs_range, gap, 'o-', label=name, linewidth=2, markersize=4)
    axes[1].plot(epochs_range, loss_gap, 'o-', label=name, linewidth=2, markersize=4)

axes[0].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold (10%)')
axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[0].set_title('Accuracy Gap (Train - Val)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Gap')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[1].set_title('Loss Gap (Val - Train)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Gap')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('📊 Overfitting Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# ============================================================
# 🔬 Hyperparameter Experiment (อย่างน้อย 3 การทดลอง)
# ============================================================

# ใช้ GRU เป็น base model สำหรับทดลอง
experiments = {
    'Baseline': {
        'rnn_units': 32, 'dense_units': 128, 'embed_dim': 64,
        'dropout1': 0.20, 'dropout2': 0.30, 'lr': 0.001, 'epochs': 1
    },
    'Exp1_EmbDim128': {
        'rnn_units': 32, 'dense_units': 128, 'embed_dim': 128,  # เปลี่ยน embedding dim
        'dropout1': 0.20, 'dropout2': 0.30, 'lr': 0.001, 'epochs': 1
    },
    'Exp2_HighDropout': {
        'rnn_units': 32, 'dense_units': 128, 'embed_dim': 64,
        'dropout1': 0.40, 'dropout2': 0.50, 'lr': 0.001, 'epochs': 1  # เพิ่ม dropout
    },
    'Exp3_LowLR': {
        'rnn_units': 32, 'dense_units': 128, 'embed_dim': 64,
        'dropout1': 0.20, 'dropout2': 0.30, 'lr': 0.0005, 'epochs': 1  # ลด learning rate
    },
}

exp_results = {}

for exp_name, params in experiments.items():
    print(f"\n{'='*60}")
    print(f"🔬 {exp_name}: units={params['rnn_units']}, embed={params['embed_dim']}, "
          f"dropout={params['dropout1']}/{params['dropout2']}, lr={params['lr']}")
    print(f"{'='*60}")
    
    # สร้างโมเดลใหม่
    exp_model = build_model(
        rnn_type='GRU',
        rnn_units=params['rnn_units'],
        dense_units=params['dense_units'],
        embed_dim=params['embed_dim'],
        dropout1=params['dropout1'],
        dropout2=params['dropout2'],
        name=f'GRU_{exp_name}'
    )
    
    # Compile
    exp_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    start_time = time.time()
    exp_history = exp_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params['epochs'],
        verbose=0  # ซ่อน progress bar เพื่อความสะอาด
    )
    exp_time = time.time() - start_time
    
    # Evaluate
    test_loss, test_acc = exp_model.evaluate(test_dataset, verbose=0)
    
    exp_results[exp_name] = {
        'params': params,
        'test_acc': test_acc,
        'val_acc': exp_history.history['val_accuracy'][-1],
        'train_acc': exp_history.history['accuracy'][-1],
        'test_loss': test_loss,
        'time': exp_time,
        'model_params': exp_model.count_params()
    }
    
    print(f"   Train Acc  : {exp_results[exp_name]['train_acc']*100:.2f}%")
    print(f"   Val Acc    : {exp_results[exp_name]['val_acc']*100:.2f}%")
    print(f"   Test Acc   : {test_acc*100:.2f}%")
    print(f"   Time       : {exp_time:.2f} sec")
    
    # ลบ model เพื่อประหยัด memory
    del exp_model
    tf.keras.backend.clear_session()

# ============================================================
# ตารางเปรียบเทียบ Hyperparameter Experiment
# ============================================================
print(f"\n{'='*90}")
print("📊 Hyperparameter Experiment Results")
print(f"{'='*90}")
print(f"{'Experiment':<20} {'RNN':>5} {'Embed':>6} {'Drop':>10} {'LR':>8} "
      f"{'Train%':>8} {'Val%':>8} {'Test%':>8} {'Time(s)':>8}")
print("-" * 90)
for exp_name, r in exp_results.items():
    p = r['params']
    print(f"{exp_name:<20} {p['rnn_units']:>5} {p['embed_dim']:>6} "
          f"{p['dropout1']:.2f}/{p['dropout2']:.2f} {p['lr']:>8.4f} "
          f"{r['train_acc']*100:>7.2f}% {r['val_acc']*100:>7.2f}% "
          f"{r['test_acc']*100:>7.2f}% {r['time']:>7.2f}")

# วิเคราะห์ผล
print(f"\n📝 วิเคราะห์ผล Hyperparameter Experiment:")
best_exp = max(exp_results, key=lambda x: exp_results[x]['test_acc'])
print(f"   🏆 การทดลองที่ดีที่สุด: {best_exp} (Test Acc: {exp_results[best_exp]['test_acc']*100:.2f}%)")
print(f"\n   1) Embedding Dim 128 vs 64:")
print(f"      - Embedding ที่ใหญ่กว่าช่วยให้แต่ละคำมี representation ที่ละเอียดขึ้น")
print(f"   2) High Dropout (0.40/0.50):")
print(f"      - Dropout สูงช่วยลด overfitting แต่อาจทำให้เรียนรู้ช้าลง")
print(f"   3) Low Learning Rate (0.0005):")
print(f"      - LR ต่ำทำให้การเรียนรู้ละเอียดขึ้น แต่ต้องใช้ epochs มากขึ้น")


# ============================================================
# 📊 เปรียบเทียบ SimpleRNN / LSTM / GRU — Final Summary
# ============================================================

print("=" * 80)
print("📊 FINAL COMPARISON: SimpleRNN vs LSTM vs GRU")
print("=" * 80)

# สร้าง DataFrame สรุปผล
comparison_data = []
for name in model_names:
    r = results[name]
    comparison_data.append({
        'Model': name,
        'Parameters': f"{r['params']:,}",
        'Training Time (sec)': f"{r['time']:.2f}",
        'Training Accuracy': f"{r['train_acc']*100:.2f}%",
        'Validation Accuracy': f"{r['val_acc']*100:.2f}%",
        'Test Accuracy': f"{r['test_acc']*100:.2f}%",
        'Test Loss': f"{r['test_loss']:.4f}",
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# ============================================================
# กราฟเปรียบเทียบ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Test Accuracy
accs = [results[n]['test_acc']*100 for n in model_names]
bars1 = axes[0].bar(model_names, accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                     edgecolor='black', linewidth=0.5)
axes[0].set_title('Test Accuracy (%)', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 100])
for bar, acc in zip(bars1, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

# 2) Parameters
params = [results[n]['params'] for n in model_names]
bars2 = axes[1].bar(model_names, params, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                     edgecolor='black', linewidth=0.5)
axes[1].set_title('Number of Parameters', fontsize=14, fontweight='bold')
for bar, p in zip(bars2, params):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                 f'{p:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3) Training Time
times = [results[n]['time'] for n in model_names]
bars3 = axes[2].bar(model_names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                     edgecolor='black', linewidth=0.5)
axes[2].set_title('Training Time (seconds)', fontsize=14, fontweight='bold')
for bar, t in zip(bars3, times):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')

plt.suptitle('📊 SimpleRNN vs LSTM vs GRU — Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# 📝 สรุปวิเคราะห์ผล
# ============================================================
best_name = max(results, key=lambda x: results[x]['test_acc'])
fastest_name = min(results, key=lambda x: results[x]['time'])
least_params_name = min(results, key=lambda x: results[x]['params'])

print(f"\n{'='*80}")
print("📝 สรุปวิเคราะห์ผล")
print(f"{'='*80}")
print(f"\n🏆 โมเดลที่ Test Accuracy สูงสุด  : {best_name} ({results[best_name]['test_acc']*100:.2f}%)")
print(f"⚡ โมเดลที่ Train เร็วสุด          : {fastest_name} ({results[fastest_name]['time']:.2f} sec)")
print(f"📦 โมเดลที่ Parameters น้อยสุด     : {least_params_name} ({results[least_params_name]['params']:,})")

print("\n📌 วิเคราะห์ความแตกต่าง:")
print("\n1. SimpleRNN:")
print("   - Parameters น้อยที่สุด → train เร็วที่สุด")
print("   - แต่ไม่มี gate mechanism → จำข้อมูลระยะยาวไม่ดี")
print("   - เกิด Vanishing Gradient → accuracy อาจต่ำกว่า LSTM/GRU")
print("\n2. LSTM:")
print("   - Parameters มากที่สุด (~4x ของ SimpleRNN) → train ช้าสุด")
print("   - มี Memory Cell + 3 Gates → จำข้อมูลระยะยาวได้ดีมาก")
print("   - เหมาะกับ sequence ที่ยาวและ dependency ซับซ้อน")
print("\n3. GRU:")
print("   - Parameters ปานกลาง (~3x ของ SimpleRNN) → train เร็วกว่า LSTM")
print("   - มี 2 Gates → จำข้อมูลระยะยาวได้ดี (แม้จะน้อยกว่า LSTM เล็กน้อย)")
print("   - มักให้ผลดีเทียบเท่า LSTM แต่ใช้ทรัพยากรน้อยกว่า")
print("\n4. สำหรับ BBC News Dataset:")
print("   - บทความข่าวมีความยาวปานกลาง → LSTM/GRU ได้เปรียบ SimpleRNN")
print(f"   - โมเดลที่เหมาะสมที่สุดคือ {best_name}")
print(f"   - เพราะให้ Test Accuracy สูงสุดที่ {results[best_name]['test_acc']*100:.2f}%")


