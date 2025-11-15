from biom import load_table
import numpy as np
from sklearn.model_selection import train_test_split

class BiomDataset:
    def __init__(self, biom_fp, label_dict):
        self.table = load_table(biom_fp)
        self.sample_ids = np.array(self.table.ids(axis='sample'))
        self.otu_ids = self.table.ids(axis='observation')

        # samples x features
        self.counts = self.table.matrix_data.toarray().T

        # align labels with sample_ids
        self.y = np.array([label_dict[sid] for sid in self.sample_ids])

    def to_numpy(self):
        """Return (X, y, sample_ids) for use in torch or kNN."""
        return self.counts, self.y, self.sample_ids

    def train_test_split(self, test_size=0.25, random_state=42, stratify=True):
        y = self.y
        stratify_y = y if stratify else None

        train_ids, test_ids, y_train, y_test = train_test_split(
            self.sample_ids,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y
        )

        id_to_idx = {sid: i for i, sid in enumerate(self.sample_ids)}
        train_idx = np.array([id_to_idx[sid] for sid in train_ids])
        test_idx  = np.array([id_to_idx[sid] for sid in test_ids])

        X_train = self.counts[train_idx]
        X_test  = self.counts[test_idx]

        return (X_train, y_train, train_ids), (X_test, y_test, test_ids)
