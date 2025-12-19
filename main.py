# ============================================================
# INSTAGRAM EXPORT → ML DATASET → INTEREST PREDICTION MODEL
# Predicts interest based on liked posts and liked comments
# ============================================================
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# CONFIG
# ============================================================
EXPORT_DIR = Path("instagram_export")  # <<< INPUT FOLDER
OUTPUT_DIR = Path("output")
RAW_DIR = OUTPUT_DIR / "raw"
EMB_DIR = OUTPUT_DIR / "embeddings"
MODEL_DIR = OUTPUT_DIR / "model"

for d in [RAW_DIR, EMB_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILS
# ============================================================
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load {path}: {e}")
        return None

def ts_to_dt(ts):
    try:
        return datetime.fromtimestamp(ts)
    except:
        return None

# ============================================================
# STEP 1: EXTRACT DATA (LIKED POSTS & LIKED COMMENTS)
# ============================================================
def extract_from_export(export_dir):
    print("📥 Reading Instagram export folder...")
    rows = []
    pid = 0
    
    # ---------- LIKED POSTS ----------
    liked_posts = safe_load_json(export_dir / "likes" / "liked_posts.json")
    if liked_posts:
        print("   Processing liked_posts.json...")
        for it in liked_posts.get("likes_media_likes", []):
            d = it["string_list_data"][0]
            rows.append({
                "post_id": pid,
                "caption": d.get("value", ""),
                "timestamp": ts_to_dt(d.get("timestamp", 0)),
                "interaction_type": "liked_post",
                "interest": 1
            })
            pid += 1
    
    # ---------- LIKED COMMENTS ----------
    liked_comments = safe_load_json(export_dir / "likes" / "liked_comments.json")
    if liked_comments:
        print("   Processing liked_comments.json...")
        for it in liked_comments.get("likes_comment_likes", []):
            d = it["string_list_data"][0]
            rows.append({
                "post_id": pid,
                "caption": d.get("value", ""),
                "timestamp": ts_to_dt(d.get("timestamp", 0)),
                "interaction_type": "liked_comment",
                "interest": 1
            })
            pid += 1
    
    if not rows:
        raise RuntimeError("❌ No usable data found in export folder. Check that liked_posts.json and liked_comments.json exist in instagram_export/likes/")
    
    df = pd.DataFrame(rows)
    print(f"✅ Extracted {len(df)} interaction rows")
    print(f"   - Liked posts: {(df['interaction_type'] == 'liked_post').sum()}")
    print(f"   - Liked comments: {(df['interaction_type'] == 'liked_comment').sum()}")
    return df

# ============================================================
# STEP 2: NEGATIVE SAMPLES
# ============================================================
def add_negative_samples(df):
    """Create synthetic negative samples by shuffling captions"""
    neg = df.copy()
    neg["interest"] = 0
    neg["interaction_type"] = "none"
    # Shuffle captions to create "not interested" examples
    neg["caption"] = np.random.permutation(neg["caption"].values)
    
    combined = pd.concat([df, neg], ignore_index=True)
    print(f"✅ Created dataset with {len(combined)} samples")
    print(f"   - Positive (interested): {(combined['interest'] == 1).sum()}")
    print(f"   - Negative (not interested): {(combined['interest'] == 0).sum()}")
    return combined

# ============================================================
# STEP 3: EMBEDDINGS
# ============================================================
class TextEmbedder:
    def __init__(self):
        print("🧠 Loading text embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def encode(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
def build_features(df, embeddings):
    """Create features from text embeddings and metadata"""
    # Time-based features
    df["hour"] = df["timestamp"].dt.hour.fillna(0)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0)
    
    # Text features
    df["caption_len"] = df["caption"].str.len().fillna(0)
    df["hashtags"] = df["caption"].str.count("#").fillna(0)
    df["mentions"] = df["caption"].str.count("@").fillna(0)
    df["word_count"] = df["caption"].str.split().str.len().fillna(0)
    
    # Combine metadata features
    meta = df[["hour", "day_of_week", "caption_len", "hashtags", "mentions", "word_count"]].values
    
    # Combine text embeddings + metadata
    X = np.concatenate([embeddings, meta], axis=1)
    y = df["interest"].values
    
    print(f"✅ Built feature matrix: {X.shape}")
    return X, y

# ============================================================
# STEP 5: MODEL
# ============================================================
class InterestModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X, y):
        """Train the interest prediction model"""
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        Xtr = self.scaler.fit_transform(Xtr)
        Xte = self.scaler.transform(Xte)
        
        # Train model
        print("\n🔧 Training model...")
        self.model.fit(Xtr, ytr)
        
        # Evaluate
        preds = self.model.predict(Xte)
        print("\n📊 Model Performance:")
        print(f"   Accuracy: {accuracy_score(yte, preds):.3f}")
        print(f"   F1 Score: {f1_score(yte, preds):.3f}")
        print("\n" + classification_report(yte, preds, target_names=["Not Interested", "Interested"]))
    
    def predict(self, X):
        """Predict interest for new posts"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]  # Return probability of interest
    
    def save(self):
        """Save trained model and scaler"""
        pickle.dump(self.model, open(MODEL_DIR / "model.pkl", "wb"))
        pickle.dump(self.scaler, open(MODEL_DIR / "scaler.pkl", "wb"))
        print(f"\n💾 Model saved to {MODEL_DIR}")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_interest(caption_text, timestamp=None):
    """
    Predict interest level for a new post
    
    Args:
        caption_text: Post caption text
        timestamp: Optional timestamp (defaults to now)
    
    Returns:
        float: Interest probability (0-1)
    """
    # Load model
    model = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
    scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))
    embedder = TextEmbedder()
    
    # Prepare data
    if timestamp is None:
        timestamp = datetime.now()
    
    df = pd.DataFrame([{
        "caption": caption_text,
        "timestamp": timestamp
    }])
    
    # Extract features
    embeddings = embedder.encode(df["caption"].tolist())
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["caption_len"] = df["caption"].str.len()
    df["hashtags"] = df["caption"].str.count("#")
    df["mentions"] = df["caption"].str.count("@")
    df["word_count"] = df["caption"].str.split().str.len()
    
    meta = df[["hour", "day_of_week", "caption_len", "hashtags", "mentions", "word_count"]].values
    X = np.concatenate([embeddings, meta], axis=1)
    
    # Predict
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    
    return prob

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("INSTAGRAM INTEREST PREDICTION MODEL")
    print("Training on: Liked Posts + Liked Comments")
    print("=" * 60)
    
    # Step 1: Extract data
    df = extract_from_export(EXPORT_DIR)
    
    # Step 2: Add negative samples
    df = add_negative_samples(df)
    
    # Save raw data
    df.to_csv(RAW_DIR / "interactions.csv", index=False)
    print(f"💾 Saved raw data to {RAW_DIR / 'interactions.csv'}")
    
    # Step 3: Generate embeddings
    embedder = TextEmbedder()
    embeddings = embedder.encode(df["caption"].tolist())
    np.save(EMB_DIR / "text_embeddings.npy", embeddings)
    print(f"💾 Saved embeddings to {EMB_DIR / 'text_embeddings.npy'}")
    
    # Step 4: Build features
    X, y = build_features(df, embeddings)
    
    # Step 5: Train model
    model = InterestModel()
    model.train(X, y)
    model.save()
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can now use predict_interest() to predict interest for new posts")
    print("Example:")
    print('  prob = predict_interest("Check out this amazing sunset! #nature")')
    print('  print(f"Interest probability: {prob:.2%}")')

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()