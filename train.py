import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
DATASET_DIR = "flac_T" 
PROTOCOL_PATH = "ASVspoof5.train.tsv"
BATCH_SIZE = 64  
FIXED_WIDTH = 400
# =================================================

class ASVspoofGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, base_dir):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.base_dir = base_dir

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_specs = []
        for file_name in batch_x:
            file_path = os.path.join(self.base_dir, file_name + ".flac")
            spec = self.process_audio(file_path)
            batch_specs.append(spec)

        return np.array(batch_specs)[..., np.newaxis], np.array(batch_y)

    def process_audio(self, file_path):
        if not os.path.exists(file_path):
            return np.zeros((128, FIXED_WIDTH))
        try:
            y, sr = sf.read(file_path)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            
            # --- NORMALIZATION LOGIC ---
            spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
            spec_db = librosa.power_to_db(spec, ref=np.max) 
            
            if spec_db.shape[1] < FIXED_WIDTH:
                pad_width = FIXED_WIDTH - spec_db.shape[1]
                spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                spec_db = spec_db[:, :FIXED_WIDTH]
            
            # Map -80dB...0dB to 0.0...1.0
            spec_norm = (spec_db + 80.0) / 80.0
            return spec_norm
            # ---------------------------
        except:
            return np.zeros((128, FIXED_WIDTH))

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # 1. Load Protocol
    print("Loading full protocol...")
    cols = ["SPEAKER_ID", "FLAC_FILE_NAME", "SPEAKER_GENDER", "CODEC", "CODEC_Q", 
            "CODEC_SEED", "ATTACK_TAG", "ATTACK_LABEL", "KEY", "TMP"]
    df = pd.read_csv(PROTOCOL_PATH, sep=' ', names=cols)
    df['target'] = df['KEY'].apply(lambda x: 1 if x == 'spoof' else 0)
    
    # 2. CREATE BALANCED SUBSET
    df_real = df[df['target'] == 0]
    df_spoof = df[df['target'] == 1]
    
    n_samples = len(df_real) 
    df_spoof_balanced = df_spoof.sample(n=n_samples, random_state=42)
    df_balanced = pd.concat([df_real, df_spoof_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_train, X_val, y_train, y_val = train_test_split(
        df_balanced['FLAC_FILE_NAME'].values, 
        df_balanced['target'].values, 
        test_size=0.2, 
        random_state=42
    )

    print(f"Training on {len(X_train)} files (Balanced)")

    train_gen = ASVspoofGenerator(X_train, y_train, BATCH_SIZE, DATASET_DIR)
    val_gen = ASVspoofGenerator(X_val, y_val, BATCH_SIZE, DATASET_DIR)

    # 3. MODEL (With Batch Norm)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(128, 400, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 4. CALLBACK: SAVE EVERY EPOCH
    # This creates: asvspoof5_epoch_01.h5, asvspoof5_epoch_02.h5, etc.
    checkpoint = ModelCheckpoint(
        "asvspoof5_epoch_{epoch:02d}.h5", 
        monitor="val_loss",
        save_best_only=False, # Save EVERY file
        verbose=1
    )

    print("Starting Training (Saving SEPARATE files for each epoch)...")
    
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=8, 
        callbacks=[checkpoint],
        workers=4, 
        use_multiprocessing=False
    )
    
    print("Training Complete. Check your folder for 'asvspoof5_epoch_XX.h5' files.")