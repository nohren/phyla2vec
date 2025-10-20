# Phyla2Vec

This project develops a phylogeny-aware encoding and diffusion framework for microbiome data, aimed at unifying 16S rRNA samples collected with different primer regions and collection protocols. Current primer-based methods (e.g., V4 vs. V3–V4) produce datasets that are incompatible across labs, limiting the effective size of training data for health classification tasks.
Our approach learns UniFrac-aligned embeddings of microbiome samples that are robust to primer and collection variance, then trains a conditional diffusion model to generate biologically consistent synthetic samples in this latent space. The result is a primer-agnostic generative pipeline capable of augmenting microbiome datasets, improving classifier generalization, and supporting downstream biological discovery.

Core components:

📘 Encoding: Transformer-based sample embedding aligned to UniFrac phylogenetic distances

🌱 Diffusion: Conditional flow-matching diffusion for synthetic microbiome sample generation

🔬 Datasets: American Gut Project (Qiita 10317, Deblur V4), with ongoing expansion to additional primer datasets (V1–V3, V3–V4) via UCSD Knight Lab collaboration

⚙️ Preprocessing: BIOM → (sample_id, sequence, nucleotide) triplet decomposition for fine-grained modeling

📊 Evaluation: Distance correlation with UniFrac, downstream classifier accuracy on real + synthetic data

Goal:
Create uniform, biologically grounded representations of 16S microbiome data that enable cross-primer compatibility and high-fidelity synthetic data generation.

## Env setup

### Setup

```bash
conda env create -f environment.yml
conda activate phyla2vec
# install the right PyTorch for your machine:
# macOS:
pip install torch torchvision torchaudio
# Windows/Linux with NVIDIA:
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```

## Data

### How to get the assembled data for this project?

```bash
https://drive.google.com/drive/folders/1ZoZcNtGAhEce-K6ldiekuRoMCkAK3RBy?usp=sharing
```

### Data fetching magic for redbiom:

1. look for suitable samples by context they were created with. Check sample count and use that primer. For instance Deblur_2021.09-Illumina-16S-V4-150nt-ac8c0b has 253301 and Deblur_2021.09-Illumina-16S-V3-150nt-ac8c0b has 1385 samples across all studies. The the intersection of qiita study 10317 (AGP) with these ctx will yield less samples.

Deblur_2021.09-Illumina-16S-V3-150nt-ac8c0b does not have AGP data.

```bash
# Show Deblur 16S contexts trimmed to 150nt; this is for V4, can change for other primers
redbiom summarize contexts \
  | grep -Ei 'Deblur.*16S.*.*150nt'
```

2. Pick one context and save as a shell variable

```bash
export CTX="Deblur_2021.09-Illumina-16S-V3V4-150nt-ac8c0b"
```

3. Get the biom samples... expect this step to take 20 to 30 min.

```
redbiom search metadata 'where qiita_study_id==10317' | redbiom fetch samples --context $CTX --output v3.biom

```

Inspecting data use biom from the package
