# Phyla2Vec

This project develops a phylogeny-aware encoding and diffusion framework for microbiome data, aimed at unifying 16S rRNA samples collected with different primer regions and collection protocols. Current primer-based methods (e.g., V4 vs. V3–V4) produce datasets that are incompatible across labs, limiting the effective size of training data for health classification tasks.
Our approach learns UniFrac-aligned embeddings of microbiome samples, i.e a microbiome manifold, that is robust to primer and collection variance, then trains a conditional diffusion model to generate biologically consistent synthetic samples in this latent space. The result is a primer-agnostic generative pipeline capable of augmenting microbiome datasets, improving classifier generalization, and supporting downstream biological discovery.

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

Stool samples represent the gut microbiome. How many stool samples exist in all contexts?

Input:

```bash
redbiom search metadata "where sample_type in ('Stool','stool')"| wc -l
```

Output:

```bash
48244
```

The American Gut Project (study number 10317) is the largest collection of stool samples in redbiom. What sample types exist in AGP and their counts?

Input:

```bash
redbiom search metadata "where qiita_study_id == 10317" | redbiom summarize samples --category sample_type
```

Output:

```bash
Stool	29369
control blank	4987
Mouth	2587
Blood (skin prick)	1166
Forehead	833
skin of cheek	400
Left Hand	397
Right Hand	322
Nares	202
Vaginal mucus	149
Torso	126
LabControl test	112
Mucus	98
not provided	81
Axilla	59
Ear wax	59
Tears	55
Left leg	53
not applicable	32
Hair	21
Right leg	9
control positive	3
```

What are all the contexts that exist for 150bp?

```bash
redbiom summarize contexts \
  | grep -Ei 'Deblur.*16S.*.*150nt'
```

- output a text file of all 150nt contexts

```bash
redbiom summarize contexts \
  | grep -Ei 'Deblur.*16S.*150nt' \
  | awk '{print $1}' > ctx_list.txt
```

1. Let's assemble stool datasets by context

New way (multiple contexts):

```bash
chmod +x run_stool.sh
./run_stool.sh
```

Old way (single context):

```bash
export ctx="Deblur_2021.09-Illumina-16S-V4-150nt-ac8c0b"
```

```bash
redbiom search metadata "where sample_type in ('Stool','stool')" \
| redbiom fetch samples --context "$ctx" --output test.biom
```

Inspect biom table

```bash
biom summarize-table -i v4.biom -o v4_summary.txt
```

5. Clean table from blanks, negs, technical replicates and insufficient depth if any exist.

```bash
python data_cleaning/clean_biom.py \
 --table data/v4_stool/v4_stool.biom \
 --ambiguities data/v4_stool/v4_stool.biom.ambiguities \
 --out data/v4_stool/v4_stool_cleaned.biom \
 --clip-count 5000
```

**beware**!!! of running commands that require stdin. They will hang forever even if you think they are doing something.

```bash
#command causes redbiom to hang as select samples-from-metadata waits for stdin input.
#This is not well documented.  No error is thrown.
redbiom select samples-from-metadata |redbiom search samples --context $ctx
```

if you want metadata for all samples in a context:

```bash
redbiom search metadata "where sample_type in ('Stool','stool')" \
| redbiom fetch sample-metadata --context $ctx --output example_meta.txt
```

6. Filtering features in dataset - Run each dataset though greengenes 2 https://github.com/biocore/q2-greengenes2 filter function to clean it.

7. Preprocess dataset into model input format

8. Get dataset metadata for downstream analysis

## Training

Starting from the first epoch and continuing every $f=5$ epochs:

- conduct rarefaction on each sample via biom-format `biom.util.generate_subsamples`. This evens the depth per sample before unifrac calculation. It dilutes anomalies from differing sequencing depths.
- Select n = 5000 random reads without replacement from each sample.

```python
  sample_1_reads = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10]

  # rarefy without replacement to 5 reads. No duplicate reads allowed.

  # allowed
  sample1_rarefy = {r3, r7, r1, r9, r5}

  # not allowed
  sample1_rarefy = {r3, r3, r1, r9, r5}

```

- compute unifrac using unifrac binaries, distance during training or precompute.
  https://github.com/biocore/unifrac-binaries

## Testing encoder

- Perccrustes analysis between corresponding unifrac and embedding sample distances in geometric space. Want high correlation.
