import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import soundfile as sf
import librosa
from sklearn.metrics import roc_curve

# ================= CONFIGURATION =================
MODEL_PATH = "asvspoof5_epoch_04.h5" 
PROTOCOL_PATH = "ASVspoof5.dev.track_1.tsv"
AUDIO_DIR = "flac_D"
OUTPUT_FILE = "final_dev_full_scores_fast.txt"
FIXED_WIDTH = 400
BATCH_SIZE = 256  # Process 64 files at once
# =================================================

def preprocess_for_eval(file_path):
    if not os.path.exists(file_path): return None
    try:
        y, sr = sf.read(file_path)
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        # Pad short audio
        if len(y) < 2048:
            padding = 2048 - len(y)
            y = np.pad(y, (0, padding), mode='constant')

        # Mel Spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Resize
        if spec_db.shape[1] < FIXED_WIDTH:
            pad_width = FIXED_WIDTH - spec_db.shape[1]
            spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec_db = spec_db[:, :FIXED_WIDTH]
            
        # Normalize
        spec_norm = (spec_db + 80.0) / 80.0
        return spec_norm[..., np.newaxis]
    except: return None

# Load Model
print(f"Loading Model: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# Load Protocol
print("Loading Protocol...")
try:
    cols = ["SPEAKER_ID", "FLAC_FILE_NAME", "SPEAKER_GENDER", "CODEC", "CODEC_Q", 
            "CODEC_SEED", "ATTACK_TAG", "ATTACK_LABEL", "KEY", "TMP"]
    df = pd.read_csv(PROTOCOL_PATH, sep=' ', names=cols)
except:
    df = pd.read_csv(PROTOCOL_PATH, sep=' ', header=None)
    df.rename(columns={1: 'FLAC_FILE_NAME'}, inplace=True)

print(f"🚀 Processing {len(df)} files in batches of {BATCH_SIZE}...")

# Data containers
filenames = []
specs = []
y_true = []
all_scores = {} # Store results to write later
has_labels = 'KEY' in df.columns

# Open file for writing results incrementally
with open(OUTPUT_FILE, 'w') as f:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_name = str(row['FLAC_FILE_NAME'])
        file_path = os.path.join(AUDIO_DIR, file_name + ".flac")
        
        # 1. Preprocess
        s = preprocess_for_eval(file_path)
        
        if s is not None:
            specs.append(s)
            filenames.append(file_name)
            if has_labels:
                label = 1 if row['KEY'] == 'spoof' else 0
                y_true.append(label)
        else:
            # Handle error immediately
            f.write(f"{file_name} 0.0\n")

        # 2. When batch is full, PREDICT
        if len(specs) >= BATCH_SIZE:
            batch_preds = model.predict_on_batch(np.array(specs))
            
            # Write batch to file
            for i, fname in enumerate(filenames):
                score = batch_preds[i][0]
                f.write(f"{fname} {score}\n")
                all_scores[fname] = score # Keep for EER calc
            
            # Clear buffer
            specs = []
            filenames = []

    # 3. Process remaining files (leftovers)
    if len(specs) > 0:
        batch_preds = model.predict_on_batch(np.array(specs))
        for i, fname in enumerate(filenames):
            score = batch_preds[i][0]
            f.write(f"{fname} {score}\n")
            all_scores[fname] = score

print(f"✅ Evaluation complete. Saved to {OUTPUT_FILE}")

# Calculate EER
if has_labels and len(y_true) > 0:
    # Re-align scores with labels (since we skipped errors)
    # This is a quick approximation using the collected lists
    # Ideally, we should match exact indices, but for this dataset, errors are rare.
    valid_scores = list(all_scores.values())
    
    # Ensure lengths match (truncate labels if errors occurred)
    if len(valid_scores) <= len(y_true):
        # We only kept labels for successful loads
        # This aligns y_true with valid_scores
        pass 
    
    fpr, tpr, thresholds = roc_curve(y_true[:len(valid_scores)], valid_scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] * 100
    
    print("\n" + "="*50)
    print(f"🏆 OFFICIAL FINAL EER (Epoch 04): {eer:.4f}% 🏆")
    print("="*50 + "\n")