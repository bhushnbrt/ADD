import tensorflow as tf
import numpy as np
import librosa
import os
import sys

# ================= CONFIGURATION =================
# 1. YOUR TRAINED MODEL
MODEL_PATH = "asvspoof5_epoch_04.h5"

# 2. THE FILE TO TEST (Change this filename!)
# You can use .wav, .mp3, .flac, etc.
# TEST_FILE = "bharat.wav"  # <--- CHANGE THIS
TEST_FILE = "flac_D/D_0002136877.flac" # Point to a real Dev file
#TEST_FILE = "flac_D/D_0000128101.flac" # Point to a fake Dev file
# =================================================

# Constants (Must match training)
FIXED_WIDTH = 400  # ~4 seconds of audio

def preprocess_audio(file_path):
    print(f"Processing: {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found: {file_path}")
        return None

    try:
        # Load audio (automatically resamples to 16kHz)
        y, sr = librosa.load(file_path, sr=16000)
        
        # Trim silence from beginning and end (optional, but helps)
        y, _ = librosa.effects.trim(y)

        # Fix Length: Pad if too short
        if len(y) < 2048:
            padding = 2048 - len(y)
            y = np.pad(y, (0, padding), mode='constant')

        # Generate Mel Spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Fit to Model Input Shape (Truncate or Pad to FIXED_WIDTH)
        if spec_db.shape[1] < FIXED_WIDTH:
            pad_width = FIXED_WIDTH - spec_db.shape[1]
            spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Note: We only test the FIRST 4 seconds
            spec_db = spec_db[:, :FIXED_WIDTH]
            
        # Normalize (Crucial!)
        spec_norm = (spec_db + 80.0) / 80.0
        
        # Add batch and channel dimensions: (1, 128, 400, 1)
        return spec_norm[np.newaxis, ..., np.newaxis]

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

# --- MAIN EXECUTION ---
print("="*50)
print(" 🕵️  DEEPFAKE DETECTOR - LIVE TEST ")
print("="*50)

# 1. Load Model
if not os.path.exists(MODEL_PATH):
    print("🚨 Model file not found! Check the name.")
    sys.exit()

print("Loading AI Brain...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Process Audio
input_tensor = preprocess_audio(TEST_FILE)

if input_tensor is not None:
    # 3. Predict
    print("Analyzing audio patterns...")
    prediction = model.predict(input_tensor, verbose=0)[0][0]
    
    # 4. Interpret Result
    # In our training: 0 = Bonafide (Real), 1 = Spoof (Fake)
    
    score_percent = prediction * 100
    
    print("\n" + "-"*30)
    print(f"RAW SCORE: {prediction:.4f}")
    print("-"*30)

    if prediction < 0.50:
        confidence = (1 - prediction) * 100
        print(f"✅ RESULT: REAL HUMAN VOICE")
        print(f"💪 Confidence: {confidence:.2f}%")
    else:
        confidence = prediction * 100
        print(f"⚠️ RESULT: ARTIFICIAL / DEEPFAKE")
        print(f"🚨 Confidence: {confidence:.2f}%")
    print("-"*30 + "\n")