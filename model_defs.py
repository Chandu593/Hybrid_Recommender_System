
import os
import ast
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline

os.environ["TF_USE_LEGACY_KERAS"] = "1"
warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("HybridRec")

CFG = dict(
    movies_path="/content/tmdb_5000_movies.csv",
    credits_path="/content/tmdb_5000_credits.csv",
    ratings_path="/content/ratings.csv",
    links_path="/content/links.csv",
    reviews_path="/content/tmdb_reviews.csv",
    cache_dir="/app/models",
    ratings_sample=8_000_000,
    test_size=0.2,
    min_ratings_user=20,
    neg_ratio=4,
    sbert_model="all-MiniLM-L6-v2",
    faiss_n_candidates=1000,
    embed_dim=64,
    proj_dim=384,
    n_heads=4,
    n_layers=2,
    max_seq_len=30,
    transformer_epochs=3,
    transformer_lr=1e-3,
    transformer_batch=128,
    grad_clip=1.0,
    ncf_embed_dim=32,
    ncf_epochs=5,
    ncf_lr=1e-3,
    ncf_batch=2048,
    deep_epochs=5,
    deep_batch=2048,
    ranker_n_est=200,
    ranker_leaves=63,
    ranker_lr=0.05,
    eval_k=10,
    eval_users=2000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    force_retrain=False,
)

class ContentModel:
    def __init__(self, movies: pd.DataFrame) -> None:
        self.movies     = movies.reset_index(drop=True)
        self.movies     = self.movies.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
        self.sbert      = SentenceTransformer(CFG["sbert_model"])
        self.embeddings: Optional[np.ndarray] = None
        self.index:      Optional[faiss.Index] = None
        self.tmdb_to_idx: Dict[int, int] = {
            int(row["id"]): i for i, row in self.movies.iterrows()
        }

    def train(self) -> None:
        emb_path = Path(CFG["cache_dir"]) / "sbert_embeddings.pkl"
        if emb_path.exists() and not CFG["force_retrain"]:
            print("  Loading cached SBERT embeddings …")
            self.embeddings = joblib.load(emb_path)
        else:
            print("  Encoding movies with SBERT …")
            self.embeddings = self.sbert.encode(
                self.movies["soup"].tolist(),
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            joblib.dump(self.embeddings, emb_path)
            print(f"  Embeddings cached → {emb_path}")

        dim        = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"  FAISS index: {self.index.ntotal} vectors, dim={dim}")

    def get_embedding(self, tmdb_id: int) -> np.ndarray:
        idx = self.tmdb_to_idx.get(int(tmdb_id))
        if idx is None:
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)
        return self.embeddings[idx]

    def retrieve_candidates(self, query_vec: np.ndarray, k: int = None) -> List[int]:
        k = k or CFG["faiss_n_candidates"]
        q = query_vec.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 1e-9:
            q /= norm
        _, indices = self.index.search(q, k)
        return [int(self.movies.iloc[i]["id"]) for i in indices[0] if i >= 0]


class SentimentModel:
    def build(
        self,
        movies: pd.DataFrame,
        reviews: Optional[pd.DataFrame],
    ) -> Dict[int, float]:
        print("Building sentiment features …")

        va   = movies[["id", "vote_average"]].dropna()
        va   = va[va["vote_average"] > 0]
        vmin = va["vote_average"].min()
        vmax = va["vote_average"].max()
        feats: Dict[int, float] = {
            int(row["id"]): 2 * (row["vote_average"] - vmin) / (vmax - vmin + 1e-9) - 1
            for _, row in va.iterrows()
        }

        if (reviews is not None
                and "content"  in reviews.columns
                and "tmdbId"   in reviews.columns):
            print("  Running DistilBERT sentiment on reviews …")
            pipe     = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=None, device=-1, batch_size=32,
            )
            rev_dict = reviews.groupby("tmdbId")["content"].apply(list).to_dict()
            for tmdb_id, texts in tqdm(rev_dict.items(), desc="  Sentiment"):
                texts = [str(t)[:512] for t in texts[:20]]
                try:
                    results = pipe(texts, truncation=True, max_length=512)
                    vals = []
                    for res in results:
                        top = max(res, key=lambda x: x["score"])
                        vals.append(
                            top["score"] if top["label"] == "POSITIVE" else -top["score"]
                        )
                    feats[int(tmdb_id)] = float(np.mean(vals))
                except Exception:
                    pass

        print(f"  Sentiment map: {len(feats):,} entries")
        return feats

def _pad_seq(seq: List[int], max_len: int) -> List[int]:
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class UserTransformer(nn.Module):
    def __init__(self, n_items: int) -> None:
        super().__init__()
        ed  = CFG["embed_dim"]
        pd_ = CFG["proj_dim"]
        self.item_emb = nn.Embedding(n_items + 1, ed, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ed, nhead=CFG["n_heads"], batch_first=True,
            dim_feedforward=ed * 4, dropout=0.1, activation="relu",
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=CFG["n_layers"])
        self.user_proj  = nn.Linear(ed, pd_)
        self.target_proj = nn.Linear(ed, pd_)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, T = seq.shape
        if T == 0:
            zero_emb = torch.zeros(B, CFG["embed_dim"], device=seq.device)
            return self.user_proj(zero_emb)

        x    = self.item_emb(seq)
        mask = (seq == 0)

        if mask.all():
            zero_emb = torch.zeros(B, CFG["embed_dim"], device=seq.device, dtype=x.dtype)
            return self.user_proj(zero_emb)

        out = self.encoder(x, src_key_padding_mask=mask)

        non_pad  = (~mask).float().unsqueeze(-1)
        sum_out  = (out * non_pad).sum(dim=1)
        count    = non_pad.sum(dim=1).clamp(min=1)
        mean_out = sum_out / count

        return self.user_proj(mean_out)

    def target_embed(self, items: torch.Tensor) -> torch.Tensor:
        return self.target_proj(self.item_emb(items))


class NCF(nn.Module):
    def __init__(self, n_users: int, n_items: int) -> None:
        super().__init__()
        ed = CFG["ncf_embed_dim"]
        self.user_emb = nn.Embedding(n_users, ed)
        self.item_emb = nn.Embedding(n_items + 1, ed, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(ed * 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)).squeeze(1)

    def bpr_loss(self, u: torch.Tensor, pos_i: torch.Tensor, neg_i: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(self.score(u, pos_i) - self.score(u, neg_i)).mean()

def build_deep_model(n_users: int, n_items: int, text_dim: int) -> Model:
    u_in    = Input(shape=(1,),        name="user")
    i_in    = Input(shape=(1,),        name="item")
    text_in = Input(shape=(text_dim,), name="text")
    sent_in = Input(shape=(1,),        name="sent")

    u = Flatten()(Embedding(n_users,     32, name="user_emb")(u_in))
    i = Flatten()(Embedding(n_items + 1, 32, name="item_emb")(i_in))

    x   = Concatenate()([u, i, text_in, sent_in])
    x   = Dense(256, activation="relu")(x)
    x   = Dropout(0.3)(x)
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(0.2)(x)
    x   = Dense(64,  activation="relu")(x)
    out = Dense(1)(x)

    model = Model([u_in, i_in, text_in, sent_in], out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

class Ranker:
    FEATURE_NAMES = ["ncf", "deep", "content_sim", "popularity", "vote_avg", "sentiment"]

    def __init__(self) -> None:
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=CFG["ranker_n_est"],
            num_leaves=CFG["ranker_leaves"],
            learning_rate=CFG["ranker_lr"],
            min_child_samples=5,
            n_jobs=-1,
            verbose=-1,
            label_gain=[0, 1, 3, 7, 15, 31],
        )

    def train(self, X: np.ndarray, y: np.ndarray, group_sizes: List[int]) -> None:
        print(f"  Training LambdaRank on {len(y):,} samples …")
        y_int = np.clip(np.round(y).astype(np.int32), 0, 5)
        self.model.fit(X, y_int, group=group_sizes, feature_name=self.FEATURE_NAMES)
        print("  LambdaRank trained")

        print("" + "=" * 40)
        print("EXPLAINABLE AI: FEATURE IMPORTANCE (GAIN)")
        print("=" * 40)
        importance = self.model.feature_importances_
        total_gain = np.sum(importance)
        for name, imp in zip(self.FEATURE_NAMES, importance):
            print(f"  {name:<15}: {imp/total_gain * 100:.1f}% contribution")
        print("=" * 40 + "")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)