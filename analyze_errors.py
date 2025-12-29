import pandas as pd
import os

# ================= CONFIGURATION =================
PROTOCOL_PATH = "ASVspoof5.dev.track_1.tsv"
SCORES_FILE = "final_dev_full_scores_fast.txt"
# =================================================

print("🕵️  Starting Forensic Analysis...")

# 1. Load the Truth (Protocol)
print(f"Loading Truth from {PROTOCOL_PATH}...")
try:
    # Try reading with headers
    cols = ["SPEAKER_ID", "FLAC_FILE_NAME", "SPEAKER_GENDER", "CODEC", "CODEC_Q", 
            "CODEC_SEED", "ATTACK_TAG", "ATTACK_LABEL", "KEY", "TMP"]
    df_truth = pd.read_csv(PROTOCOL_PATH, sep=' ', names=cols)
except:
    # Fallback for no headers
    df_truth = pd.read_csv(PROTOCOL_PATH, sep=' ', header=None)
    df_truth.rename(columns={1: 'FLAC_FILE_NAME', 8: 'KEY'}, inplace=True)

# Keep only what we need: Filename and Key (bonafide/spoof)
df_truth = df_truth[['FLAC_FILE_NAME', 'KEY']]
df_truth['FLAC_FILE_NAME'] = df_truth['FLAC_FILE_NAME'].astype(str)

# 2. Load the Predictions (Scores)
print(f"Loading Scores from {SCORES_FILE}...")
# The scores file is "filename score"
df_scores = pd.read_csv(SCORES_FILE, sep=' ', names=['FLAC_FILE_NAME', 'SCORE'])
df_scores['FLAC_FILE_NAME'] = df_scores['FLAC_FILE_NAME'].astype(str)

# 3. Merge them
print("Merging data...")
df = pd.merge(df_truth, df_scores, on='FLAC_FILE_NAME')

print(f"Successfully matched {len(df)} files.")

# ================= ANALYSIS =================

# --- CASE 1: FALSE POSITIVES (False Alarms) ---
# Truth = 'bonafide' (Real), but Score is HIGH (Model thinks Fake)
false_positives = df[df['KEY'] == 'bonafide'].copy()
# Sort by score descending (Highest confidence fakes)
worst_fp = false_positives.sort_values(by='SCORE', ascending=False).head(5)

print("\n" + "="*60)
print("🚨 TOP 5 FALSE ALARMS (Real Humans flagged as Deepfakes)")
print("These files are likely noisy, short, or have weird microphones.")
print("="*60)
for _, row in worst_fp.iterrows():
    print(f"File: {row['FLAC_FILE_NAME']}.flac  |  Model Confidence: {row['SCORE']*100:.2f}% Fake")

# --- CASE 2: FALSE NEGATIVES (Missed Attacks) ---
# Truth = 'spoof' (Fake), but Score is LOW (Model thinks Real)
false_negatives = df[df['KEY'] == 'spoof'].copy()
# Sort by score ascending (Lowest confidence fakes -> Model thought they were very Real)
worst_fn = false_negatives.sort_values(by='SCORE', ascending=True).head(5)

print("\n" + "="*60)
print("⚠️ TOP 5 MISSED ATTACKS (Deepfakes that tricked the AI)")
print("These are the 'Super-Deepfakes' your model cannot detect.")
print("="*60)
for _, row in worst_fn.iterrows():
    print(f"File: {row['FLAC_FILE_NAME']}.flac  |  Model Confidence: {(1-row['SCORE'])*100:.2f}% Real")

print("\nDone. Copy these filenames and listen to them in your folder!")