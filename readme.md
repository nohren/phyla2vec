# Phyla2Vec

This project develops a phylogeny-aware encoding and diffusion framework for microbiome data, aimed at unifying 16S rRNA samples collected with different primer regions and collection protocols. Current primer-based methods (e.g., V4 vs. V3–V4) produce datasets that are incompatible across labs, limiting the effective size of training data for health classification tasks.
Our approach learns UniFrac-aligned embeddings of microbiome samples, i.e a relational manifold of the microbiome, that is robust to primer and collection variance, then trains a conditional diffusion model to generate biologically consistent synthetic samples in this latent space. The result is a primer-agnostic generative pipeline capable of augmenting microbiome datasets, improving classifier generalization, and supporting downstream biological discovery.

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

### How to get the assembled datasets for this project?

```bash
https://drive.google.com/drive/folders/1ZoZcNtGAhEce-K6ldiekuRoMCkAK3RBy?usp=sharing
```

### Data fetching and cleaning steps:

1. look for suitable samples by the context they were created i.e Deblur._16S._.\*150nt. Check sample count and primer. If all looks good use that context.

```bash
# Show Deblur 16S contexts trimmed to 150nt; this is for V4, can change for other primers
redbiom summarize contexts \
  | grep -Ei 'Deblur.*16S.*.*150nt'
```

2. Pick one context and save as a shell variable

```bash
export CTX="Deblur_2021.09-Illumina-16S-V4-150nt-ac8c0b"
```

3. Get the biom samples... expect this step to take 20 to 30 min for large samples.

- use redbiom flags to filter any technical replicates or blank samples

```bash
redbiom search metadata 'where qiita_study_id==10317' | \
redbiom fetch samples --context $CTX --output v4.biom
```

4. Inspect biom table

```bash
biom summarize-table -i v4.biom -o v4_summary.txt
```

5. Assemble second dataset

- Repeat steps 1 and 2 for all non V4 contexts getting stool samples from other studies.
- use redbiom flags to filter any technical replicates or blank samples

```bash
redbiom search metadata 'where (body_site=="stool") and (qiita_study_id!=10317)' | \
redbiom fetch samples --context $CTX --output other_stool.biom
```

6. If still replicates or blanks, filter manually. technical replicates in metadata host_id. if multiple sample has host_id, keep only one with highest total observation count.
   Maybe flag in redbiom for technical replicates and one for blanks?

7. Filtering features in dataset - Run each dataset though greengenes 2 https://github.com/biocore/q2-greengenes2 filter function to clean it.

8. Preprocess dataset into model input format

## Training

Starting from the first epoch and continuing every $f=5$ epochs:

- conduct rarefaction on each sample via biom-format `biom.util.generate_subsamples`. This evens the depth per sample before unifrac calculation. It dilutes anomalies from differing sequencing depths.
- Select n = 5000 random reads without replacement from each sample.
- compute unifrac using unifrac binaries, distance during training or precompute.
  https://github.com/biocore/unifrac-binaries

## Testing encoder

- Perccrustes analysis between corresponding unifrac and embedding sample distances in geometric space. Want high correlation.
