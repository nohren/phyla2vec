import numpy as np

import biom
import unifrac
import h5py
import tempfile
from skbio import TreeNode

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F


########################################
# Config
########################################

# Use your 5000-sample subset + its tree
BIOM_PATH = "train_filtered/train.biom"
TREE_PATH = "train_filtered/tree.nwk"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RAREFY_DEPTH = 5000                # target reads per sample (for model/UniFrac alignment)
MODEL_READS = 1024                 # reads per sample fed to the model
MAX_SAMPLES = None                 # cap number of samples per epoch

SEQ_LEN = 150                      # 150bp sequences
EMBED_DIM = 128
BATCH_SIZE = 8                     # adjust for memory / speed
LR = 1e-4
NUM_EPOCHS = 20
RAREFY_INTERVAL = 5                # re-rarefy every 5 epochs

BETA_KL = 1e-4
USE_UNIFRAC_LOSS = True   # set True if we want UniFrac loss in VAE
UNIFRAC_WEIGHT = 50.0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


########################################
# Nucleotide encoding
########################################

NUC_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,  # catch-all
}
VOCAB_SIZE = len(NUC_TO_ID)


def encode_sequence(seq, max_len=SEQ_LEN):
    """
    Encode a nucleotide string (feature ID) into fixed-length int IDs.
    Pads/truncates to max_len.
    """
    seq = seq.upper()
    ids = [NUC_TO_ID.get(ch, NUC_TO_ID["N"]) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [NUC_TO_ID["N"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.uint8)


########################################
# Data prep: rarefaction and subsampling
########################################

def filter_samples_by_depth(table, depth):
    """
    Filter samples to those with total count >= depth.
    """
    def sample_filter(vals, sid, md):
        return vals.sum() >= depth

    return table.filter(sample_filter, axis="sample", inplace=False)


def subsample_samples(table, max_samples=None, seed=0):
    """
    Optionally randomly select up to max_samples samples from the table.

    If max_samples is None or <= 0, use all samples (no subsampling).
    """
    sample_ids = list(table.ids(axis="sample"))
    n = len(sample_ids)

    # No subsampling: use all samples
    if (max_samples is None) or (max_samples <= 0) or (n <= max_samples):
        return table, sample_ids

    rng = np.random.default_rng(seed)
    chosen = rng.choice(sample_ids, size=max_samples, replace=False)
    chosen = chosen.tolist()

    sub = table.filter(chosen, axis="sample", inplace=False)
    return sub, chosen



def prepare_epoch_data(full_table, depth, seq_len, model_reads, max_samples=None, seed=0):
    """
    For one "rarefaction epoch":

      1. Filter samples with >= depth total reads.
      2. Subsample up to max_samples samples.
      3. Rarefy each sample to 'depth' using BIOM's subsample (without replacement).
      4. Build:
         - rare_table: rarefied BIOM table for these samples
         - sample_ids: list of sample IDs
         - obs_ids: list of observation IDs (150bp feature IDs)
         - sample_tokens: (num_samples, model_reads, seq_len) nucleotide token arrays

    Returns:
      sample_tokens: np.ndarray, shape (num_samples, model_reads, seq_len), uint8
      rare_table: biom.Table (rarefied)
      sample_ids: list of sample IDs (order matches sample_tokens rows)
      obs_ids: list of observation IDs (order matches columns in rare_table)
    """

    # 1) Filter samples with enough total counts
    filtered = filter_samples_by_depth(full_table, depth)

    # 2) Subsample samples to cap memory/compute
    filtered_sub, chosen_ids = subsample_samples(filtered, max_samples, seed=seed)

    # 3) Rarefy to 'depth' per sample (without replacement)
    rare = filtered_sub.subsample(depth, axis="sample")

    sample_ids = list(rare.ids(axis="sample"))
    obs_ids = list(rare.ids(axis="observation"))  # 150bp feature IDs

    num_samples = len(sample_ids)
    num_obs = len(obs_ids)

    print(f"  Rarefied samples: {num_samples}, observations: {num_obs}")

    # Encode each 150bp feature sequence once
    obs_tokens = np.stack(
        [encode_sequence(seq, max_len=seq_len) for seq in obs_ids],
        axis=0
    )  # (num_obs, seq_len)

    # We'll feed only `model_reads` reads per sample to the model
    model_reads = min(depth, model_reads)
    sample_tokens = np.zeros((num_samples, model_reads, seq_len), dtype=np.uint8)

    for j, sid in enumerate(sample_ids):
        # Rarefied counts over observations for this sample (sparse or dense)
        counts = np.asarray(rare.data(id=sid, axis="sample")).flatten()
        counts_int = counts.astype(np.int64)

        total = int(counts_int.sum())
        if total != depth:
            raise ValueError(
                f"Rarefied sample {sid} sums to {total}, expected {depth}"
            )

        # Expand counts to explicit indices: e.g., [3,1,0,2] -> [0,0,0,1,3,3]
        expanded_full = np.repeat(np.arange(num_obs, dtype=np.int64), counts_int)
        # expanded_full has length == depth

        # For the model, sample up to `model_reads` reads without replacement
        if model_reads < depth:
            chosen_idx = np.random.choice(
                depth, size=model_reads, replace=False
            )
            expanded = expanded_full[chosen_idx]
        else:
            expanded = expanded_full

        np.random.shuffle(expanded)  # randomize read order for the model
        sample_tokens[j] = obs_tokens[expanded]

    return sample_tokens, rare, sample_ids, obs_ids


########################################
# Dataset
########################################

class SampleSequenceDataset(Dataset):
    """
    Each item is a single sample:
      seq: (num_reads=MODEL_READS, seq_len=SEQ_LEN) int64 tokens
      idx: sample index (for indexing sample_ids)
    """

    def __init__(self, sample_tokens):
        # sample_tokens: (N_samples, model_reads, seq_len), uint8
        self.sample_tokens = sample_tokens

    def __len__(self):
        return self.sample_tokens.shape[0]

    def __getitem__(self, idx):
        seqs = self.sample_tokens[idx]  # (R, L)
        return {
            "idx": idx,
            "seq": torch.from_numpy(seqs.astype(np.int64)),  # (R, L)
        }


########################################
# Model: two-level Transformer encoder
########################################

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
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class AttnPool1D(nn.Module):
    """
    Input:  x (B, N, D)
    Output: pooled (B, D)

    Learns a scalar weight per position N, then softmax → weighted sum.
    
    https://aclanthology.org/N16-1174.pdf "word attention"
    - name is a misnomer, there's no self attention here
    - its an mlp ranker really that scores each position invidivually. That being said each position is already infused with relationships to others due upstream transformer
    - after mlp then softmax through the logits to get weights, then weighted sum over positions to get pooled output.  Tanh for -> [-1,1] bounded activations
    - seems to work better than mean pooling, gives the model more expressivity in pooling.
    """
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1) # # (B, N, 1) weights
        )

    def forward(self, x):
        # x: (B, N, D) i.e (B*R, L, D) 
        weights = self.proj(x).squeeze(-1) # (B, N)
        # learns a weighted average over N positions
        weights = torch.softmax(weights, dim=1) # softmax over weight logits
        # apply learned weights and sum over N positions to get (B, D)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled

class UniFracEncoder(nn.Module):
    """
    Level 1: Transformer over nucleotides within each 150bp sequence.
    Level 2: Transformer over sequences (reads) within each sample.
    """

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

        # Embedding for nucleotides
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding over nucleotide positions
        self.pos_enc_nuc = PositionalEncoding(embed_dim, max_len=seq_len)

        # Transformer over nucleotides (per sequence)
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

        # Positional encoding over sequences (reads) within a sample
        self.pos_enc_seq = PositionalEncoding(embed_dim, max_len=MODEL_READS)

        # Transformer over sequence embeddings (per sample)
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
        self.nuc_pool = AttnPool1D(embed_dim)  # over nucleotide positions L
        self.seq_pool = AttnPool1D(embed_dim)  # over reads R

    def forward(self, seq_ids):
        """
        seq_ids: (B, R, L)
          B = batch size
          R = number of reads per sample (MODEL_READS)
          L = nucleotide positions (SEQ_LEN)

        Returns:
          sample_emb: (B, EMBED_DIM)
        """
        B, R, L = seq_ids.shape
        assert L == self.seq_len, "Unexpected sequence length"

        # -------- Level 1: nucleotide-level transformer per sequence --------
        x = seq_ids.view(B * R, L)       # (B*R, L)
        x = self.embed(x)                # (B*R, L, D)
        x = self.pos_enc_nuc(x)          # (B*R, L, D)
        x = self.nuc_encoder(x)          # (B*R, L, D)

        # # Mean over nucleotides -> per-sequence embedding
        # seq_emb = x.mean(dim=1)          # (B*R, D)
        
        # Learned pooling over nucleotides -> per-sequence embedding
        seq_emb = self.nuc_pool(x)        # (B*R, D)
        seq_emb = seq_emb.view(B, R, -1)   # (B, R, D)

        # -------- Level 2: sequence-level transformer per sample --------
        # y = self.pos_enc_seq(seq_emb)    # (B, R, D) reads are permutation invariant
        y = self.seq_encoder(seq_emb)          # (B, R, D)
        
        # # Mean over Reads -> per-sequence embedding
        # sample_emb = y.mean(dim=1)       # (B, D)
        
        # Learned pooling over reads -> per-sample embedding
        sample_emb = self.seq_pool(y)       # (B, D)

        return sample_emb
    
class MLPUniFracEncoder(nn.Module):
    """
    Simpler encoder:
      - Embed nucleotides
      - Average over reads and positions -> (B, D)
      - Pass through a small MLP -> (B, D)
    """

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
        mlp_layers.append(nn.Linear(in_dim, embed_dim))  # final to EMBED_DIM

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, seq_ids):
        """
        seq_ids: (B, R, L) of int64 tokens

        Returns:
          sample_emb: (B, EMBED_DIM)
        """
        B, R, L = seq_ids.shape
        assert L == self.seq_len, "Unexpected sequence length"

        # Embed tokens: (B, R, L, D)
        x = self.embed(seq_ids)

        # Global mean pooling over reads + positions -> (B, D)
        sample_emb = x.mean(dim=(1, 2))

        # MLP to get final embedding
        sample_emb = self.mlp(sample_emb)  # (B, D)

        return sample_emb


class UniFracVAE(nn.Module):
    """
    VAE where the encoder is exactly your UniFracEncoder stack.

    Encoder: seq_ids -> sample_emb (UniFracEncoder) -> mu, logvar
    Decoder: z -> nucleotide frequency logits per sample (length VOCAB_SIZE)

    Reconstruction target: per-sample nucleotide frequency vector derived
    from the input seq_ids.
    """

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

        # Latent heads
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        # Simple decoder MLP: z -> vocab_size logits
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size),
        )

    def encode(self, seq_ids):
        """
        seq_ids: (B, R, L)
        Returns: mu, logvar  each (B, latent_dim)
        """
        sample_emb = self.encoder_backbone(seq_ids)
        mu = self.fc_mu(sample_emb)
        logvar = self.fc_logvar(sample_emb)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Standard Gaussian reparam: z = mu + eps * exp(0.5*logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        z: (B, latent_dim)
        Returns:
          recon_logits: (B, vocab_size)
        """
        return self.decoder(z)

    def forward(self, seq_ids):
        """
        seq_ids: (B, R, L)

        Returns:
          recon_logits: (B, vocab_size)
          mu, logvar: (B, latent_dim)
          target_freqs: (B, vocab_size)  # reconstruction target
        """
        B, R, L = seq_ids.shape

        with torch.no_grad():
            # One-hot: (B, R, L, vocab_size)
            one_hot = F.one_hot(seq_ids, num_classes=self.vocab_size).float()
            # Average over reads and positions -> (B, vocab_size)
            target_freqs = one_hot.mean(dim=(1, 2))

        # ----- Encoder -----
        mu, logvar = self.encode(seq_ids)
        z = self.reparameterize(mu, logvar)

        # ----- Decoder -----
        recon_logits = self.decode(z)  # (B, vocab_size)

        return recon_logits, mu, logvar, target_freqs

def vae_loss(recon_logits, target_freqs, mu, logvar, beta=1.0):
    """
    recon_logits: (B, vocab_size), raw logits
    target_freqs: (B, vocab_size), in [0,1] (nucleotide frequencies)
    mu, logvar: (B, latent_dim)

    Uses:
      - Reconstruction: BCEWithLogitsLoss on nucleotide frequencies
      - KL divergence to N(0, I), weighted by beta
    """
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_logits,
        target_freqs,
        reduction="sum",
    )
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    B = recon_logits.size(0)
    recon_loss = recon_loss / B
    kl = kl / B

    total = recon_loss + beta * kl
    return total, recon_loss, kl





########################################
# UniFrac distance helpers
########################################

def pairwise_euclidean_dist(x):
    """
    x: (B, D)
    returns: (B, B) Euclidean distance matrix
    """
    sq_norms = (x ** 2).sum(dim=1, keepdim=True)   # (B, 1)
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
    
    # # NEW: normalize embeddings to unit norm so loss focuses on geometry
    # embeddings = F.normalize(embeddings, p=2, dim=1)  # (B, D)

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
    print("Device is ", DEVICE)
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
        # Re-rarefy every RAREFY_INTERVAL epochs
        if epoch % RAREFY_INTERVAL == 0:
            print(f"\n[Epoch {epoch+1}] Re-rarefying ...")
            sample_tokens, rare_table, sample_ids, obs_ids = prepare_epoch_data(
                full_table,
                RAREFY_DEPTH,
                SEQ_LEN,
                MODEL_READS,
                MAX_SAMPLES,
                seed=epoch,   # change seed each time for different subset
            )

            dataset = SampleSequenceDataset(sample_tokens)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            )
            print(f"  Num samples after filtering+subsample: {len(dataset)}")

            # Map this epoch's sample_ids to rows in the global UniFrac DM
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
            seq = batch["seq"].to(DEVICE)   # (B, R, L)
            idx = batch["idx"].to(DEVICE)   # (B,)
            
            # print("seq.device:", seq.device)
            # print("CUDA mem allocated (MB):", torch.cuda.memory_allocated() / 1e6)

            optimizer.zero_grad()
            emb = model(seq)                # (B, D)

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
        # Save model checkpoint
        save_path = f"pres_transformer_encoder_epoch_{epoch+1}.pt"
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
    # Load full BIOM table once (for counts + sequences)
    full_table = biom.load_table(BIOM_PATH)

    # Optional: compute UniFrac DM if we want UniFrac loss
    unifrac_dm_full = None
    id_to_dm_idx = None
    if USE_UNIFRAC_LOSS:
        print("Computing full unweighted UniFrac distance matrix from files for VAE ...")
        unifrac_dm_full, dm_ids = compute_unweighted_unifrac_matrix_from_files(
            BIOM_PATH,
            TREE_PATH,
        )
        id_to_dm_idx = {sid: i for i, sid in enumerate(dm_ids)}

    model = UniFracVAE().to(DEVICE)
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

            if USE_UNIFRAC_LOSS:
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
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_unifrac = 0.0
        num_batches = 0

        for batch in dataloader:
            seq = batch["seq"].to(DEVICE)   # (B, R, L)
            idx = batch["idx"].to(DEVICE)   # (B,)

            optimizer.zero_grad()
            recon_logits, mu, logvar, target_freqs = model(seq)

            target_freqs = target_freqs.to(DEVICE)

            # Base VAE loss with KL weighted by BETA_KL
            vae_total, recon_loss, kl_loss = vae_loss(
                recon_logits, target_freqs, mu, logvar, beta=BETA_KL
            )

            total_loss = vae_total

            # Optional UniFrac distance-matching term on the latent means mu
            unifrac_loss = torch.tensor(0.0, device=DEVICE)
            if USE_UNIFRAC_LOSS and unifrac_dm_full is not None and dataset_to_dm_idx is not None:
                unifrac_loss = distance_matching_loss_batch(
                    embeddings=mu,           # (B, latent_dim)
                    batch_indices=idx,       # indices into this epoch's dataset
                    unifrac_dm=unifrac_dm_full,
                    dataset_to_dm_idx=dataset_to_dm_idx,
                )
                total_loss = total_loss + UNIFRAC_WEIGHT * unifrac_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_unifrac += unifrac_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        avg_recon = epoch_recon / max(1, num_batches)
        avg_kl = epoch_kl / max(1, num_batches)
        avg_unifrac = epoch_unifrac / max(1, num_batches) if USE_UNIFRAC_LOSS else 0.0

        print(
            f"VAE Epoch {epoch+1}/{NUM_EPOCHS} "
            f"- Total: {avg_loss:.6f}, Recon: {avg_recon:.6f}, "
            f"KL (beta={BETA_KL}): {avg_kl:.6f}, "
            f"UniFrac: {avg_unifrac:.6f}"
        )

        save_path = f"/data/nicklas/scratch/pres_unifrac_vae_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
            "unifrac_loss": avg_unifrac,
            "sample_ids": sample_ids,
        }, save_path)
        print(f"Saved VAE checkpoint to {save_path}")


    print("VAE training finished.")



if __name__ == "__main__":
    main()
    #main_vae()
