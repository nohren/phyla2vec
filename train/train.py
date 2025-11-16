import numpy as np

import biom
#import unifrac
import h5py
import tempfile
#from skbio import TreeNode

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

BIOM_PATH = "feature-table-5000.biom"
TREE_PATH = "tree.nwk"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RAREFY_DEPTH = 5000
MODEL_READS = 1024
MAX_SAMPLES = 5000

SEQ_LEN = 150
EMBED_DIM = 128
BATCH_SIZE = 16
LR = 1e-4
NUM_EPOCHS = 20
RAREFY_INTERVAL = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUC_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}
VOCAB_SIZE = len(NUC_TO_ID)


def encode_sequence(seq, max_len=SEQ_LEN):
    seq = seq.upper()
    ids = [NUC_TO_ID.get(ch, NUC_TO_ID["N"]) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [NUC_TO_ID["N"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.uint8)


def filter_samples_by_depth(table, depth):

    def sample_filter(vals, sid, md):
        return vals.sum() >= depth

    return table.filter(sample_filter, axis="sample", inplace=False)


def subsample_samples(table, max_samples, seed=0):

    sample_ids = list(table.ids(axis="sample"))
    n = len(sample_ids)
    if n <= max_samples:
        return table, sample_ids

    rng = np.random.default_rng(seed)
    chosen = rng.choice(sample_ids, size=max_samples, replace=False)
    chosen = chosen.tolist()

    sub = table.filter(chosen, axis="sample", inplace=False)
    return sub, chosen


def prepare_epoch_data(full_table, depth, seq_len, model_reads, max_samples, seed=0):

    filtered = filter_samples_by_depth(full_table, depth)

    filtered_sub, chosen_ids = subsample_samples(filtered, max_samples, seed=seed)

    rare = filtered_sub.subsample(depth, axis="sample")

    sample_ids = list(rare.ids(axis="sample"))
    obs_ids = list(rare.ids(axis="observation"))

    num_samples = len(sample_ids)
    num_obs = len(obs_ids)

    print(f"  Rarefied samples: {num_samples}, observations: {num_obs}")

    obs_tokens = np.stack(
        [encode_sequence(seq, max_len=seq_len) for seq in obs_ids],
        axis=0
    )

    model_reads = min(depth, model_reads)
    sample_tokens = np.zeros((num_samples, model_reads, seq_len), dtype=np.uint8)

    for j, sid in enumerate(sample_ids):
        counts = np.asarray(rare.data(id=sid, axis="sample")).flatten()
        counts_int = counts.astype(np.int64)

        total = int(counts_int.sum())
        if total != depth:
            raise ValueError(
                f"Rarefied sample {sid} sums to {total}, expected {depth}"
            )

        expanded_full = np.repeat(np.arange(num_obs, dtype=np.int64), counts_int)

        if model_reads < depth:
            chosen_idx = np.random.choice(
                depth, size=model_reads, replace=False
            )
            expanded = expanded_full[chosen_idx]
        else:
            expanded = expanded_full

        np.random.shuffle(expanded)
        sample_tokens[j] = obs_tokens[expanded]

    return sample_tokens, rare, sample_ids, obs_ids

class SampleSequenceDataset(Dataset):

    def __init__(self, sample_tokens):
        self.sample_tokens = sample_tokens

    def __len__(self):
        return self.sample_tokens.shape[0]

    def __getitem__(self, idx):
        seqs = self.sample_tokens[idx]
        return {
            "idx": idx,
            "seq": torch.from_numpy(seqs.astype(np.int64)),
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class UniFracEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        seq_len=SEQ_LEN,
        n_heads=4,
        dim_feedforward=256,
        num_layers_nuc=2,
        num_layers_seq=2,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.pos_enc_nuc = PositionalEncoding(embed_dim, max_len=seq_len)

        nuc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.nuc_encoder = nn.TransformerEncoder(
            nuc_layer, num_layers=num_layers_nuc
        )

        self.pos_enc_seq = PositionalEncoding(embed_dim, max_len=MODEL_READS)

        seq_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(
            seq_layer, num_layers=num_layers_seq
        )

    def forward(self, seq_ids):
        B, R, L = seq_ids.shape
        assert L == self.seq_len, "Unexpected sequence length"

        x = seq_ids.view(B * R, L)
        x = self.embed(x)
        x = self.pos_enc_nuc(x)
        x = self.nuc_encoder(x)

        seq_emb = x.mean(dim=1)
        seq_emb = seq_emb.view(B, R, -1)

        y = self.pos_enc_seq(seq_emb)
        y = self.seq_encoder(y)
        sample_emb = y.mean(dim=1)

        return sample_emb


class MLPUniFracEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        seq_len=SEQ_LEN,
        hidden_dim=256,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)

        mlp_layers = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, embed_dim))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, seq_ids):

        B, R, L = seq_ids.shape
        assert L == self.seq_len, "Unexpected sequence length"

        x = self.embed(seq_ids)
        sample_emb = x.mean(dim=(1, 2))

        sample_emb = self.mlp(sample_emb)

        return sample_emb


class UniFracVAE(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        seq_len=SEQ_LEN,
        latent_dim=32,
        n_heads=4,
        dim_feedforward=256,
        num_layers_nuc=2,
        num_layers_seq=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder_backbone = UniFracEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            num_layers_nuc=num_layers_nuc,
            num_layers_seq=num_layers_seq,
            dropout=dropout,
        )

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size),
        )

    def encode(self, seq_ids):

        sample_emb = self.encoder_backbone(seq_ids)
        mu = self.fc_mu(sample_emb)
        logvar = self.fc_logvar(sample_emb)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, seq_ids):
        B, R, L = seq_ids.shape

        with torch.no_grad():
            one_hot = F.one_hot(seq_ids, num_classes=self.vocab_size).float()
            target_freqs = one_hot.mean(dim=(1, 2))

        mu, logvar = self.encode(seq_ids)
        z = self.reparameterize(mu, logvar)

        recon_logits = self.decode(z)

        return recon_logits, mu, logvar, target_freqs

def vae_loss(recon_logits, target_freqs, mu, logvar, beta=1.0):

    recon_loss = F.binary_cross_entropy_with_logits(
        recon_logits,
        target_freqs,
        reduction="sum",
    )

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    B = recon_logits.size(0)
    recon_loss = recon_loss / B
    kl = kl / B

    return recon_loss + beta * kl, recon_loss, kl


def pairwise_euclidean_dist(x):

    sq_norms = (x ** 2).sum(dim=1, keepdim=True)   
    sq_dist = sq_norms + sq_norms.t() - 2.0 * (x @ x.t())
    sq_dist = torch.clamp(sq_dist, min=0.0)
    dist = torch.sqrt(sq_dist + 1e-8)
    return dist


def compute_unweighted_unifrac_matrix_from_files(biom_path, tree_path):
    dm = unifrac.unweighted(biom_path, tree_path)
    dm_mat = dm.data.astype(np.float32)
    dm_ids = list(dm.ids)
    print(f"  UniFrac distance matrix shape: {dm_mat.shape}")
    return dm_mat, dm_ids


def distance_matching_loss_batch(
    embeddings,
    batch_indices,
    unifrac_dm,
    dataset_to_dm_idx,
):
    device = embeddings.device
    B = embeddings.size(0)

    pred_dist = pairwise_euclidean_dist(embeddings)

    batch_indices_np = batch_indices.cpu().numpy()
    batch_dm_idx = dataset_to_dm_idx[batch_indices_np]

    target = unifrac_dm[np.ix_(batch_dm_idx, batch_dm_idx)]
    target_dist = torch.from_numpy(target).to(device)

    loss = torch.mean((pred_dist - target_dist) ** 2)
    return loss

def main():
    full_table = biom.load_table(BIOM_PATH)

    print("Computing full unweighted UniFrac distance matrix from files ...")
    unifrac_dm_full, dm_ids = compute_unweighted_unifrac_matrix_from_files(
        BIOM_PATH,
        TREE_PATH,
    )
    id_to_dm_idx = {sid: i for i, sid in enumerate(dm_ids)}

    model = UniFracEncoder().to(DEVICE)
    #model = MLPUniFracEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    sample_tokens = None
    rare_table = None
    sample_ids = None
    obs_ids = None
    dataset = None
    dataloader = None
    dataset_to_dm_idx = None

    for epoch in range(NUM_EPOCHS):
        if epoch % RAREFY_INTERVAL == 0:
            print(f"\n[Epoch {epoch+1}] Re-rarefying ...")
            sample_tokens, rare_table, sample_ids, obs_ids = prepare_epoch_data(
                full_table,
                RAREFY_DEPTH,
                SEQ_LEN,
                MODEL_READS,
                MAX_SAMPLES,
                seed=epoch,
            )

            dataset = SampleSequenceDataset(sample_tokens)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            )
            print(f"  Num samples after filtering+subsample: {len(dataset)}")

            try:
                dataset_to_dm_idx = np.array(
                    [id_to_dm_idx[sid] for sid in sample_ids],
                    dtype=np.int64,
                )
            except KeyError as e:
                missing = str(e)
                raise RuntimeError(
                    f"Sample ID {missing} from rarefied table not found in UniFrac "
                    f"distance matrix IDs. Check that BIOM and tree.nwk match."
                )

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            seq = batch["seq"].to(DEVICE)   
            idx = batch["idx"].to(DEVICE)   

            optimizer.zero_grad()
            emb = model(seq)

            loss = distance_matching_loss_batch(
                emb,
                idx,
                unifrac_dm_full,
                dataset_to_dm_idx,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.6f}")
        save_path = f"/data/nicklas/scratch/unifrac_mlp_encoder_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "sample_ids": sample_ids,
        }, save_path)
        print(f"Saved checkpoint to {save_path}")


    print("Training finished.")

def main_vae():

    full_table = biom.load_table(BIOM_PATH)

    model = UniFracVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    sample_tokens = None
    rare_table = None
    sample_ids = None
    obs_ids = None
    dataset = None
    dataloader = None

    for epoch in range(NUM_EPOCHS):

        if epoch % RAREFY_INTERVAL == 0:
            print(f"\n[Epoch {epoch+1}] Re-rarefying for VAE ...")
            sample_tokens, rare_table, sample_ids, obs_ids = prepare_epoch_data(
                full_table,
                RAREFY_DEPTH,
                SEQ_LEN,
                MODEL_READS,
                MAX_SAMPLES,
                seed=epoch,
            )

            dataset = SampleSequenceDataset(sample_tokens)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            )
            print(f"  Num samples after filtering+subsample: {len(dataset)}")

        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for batch in dataloader:
            seq = batch["seq"].to(DEVICE)

            optimizer.zero_grad()
            recon_logits, mu, logvar, target_freqs = model(seq)

            target_freqs = target_freqs.to(DEVICE)

            loss, recon_loss, kl_loss = vae_loss(
                recon_logits, target_freqs, mu, logvar, beta=1.0
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        avg_recon = epoch_recon / max(1, num_batches)
        avg_kl = epoch_kl / max(1, num_batches)
        print(
            f"VAE Epoch {epoch+1}/{NUM_EPOCHS} "
            f"- Total: {avg_loss:.6f}, Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}"
        )

        save_path = f"/data/nicklas/scratch/unifrac_vae_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
            "sample_ids": sample_ids,
        }, save_path)
        print(f"Saved VAE checkpoint to {save_path}")

    print("VAE training finished.")



if __name__ == "__main__":
    #main()
    main_vae()
