# Dataset 

This directory contains scripts and processed data for experiments. All datasets are managed through a unified download script to ensure environment and data consistency.

---

## 📂 Directory Structure

After extraction, each dataset (e.g., `amazon_movie`) follows the specific hierarchy shown below:

```text
data/
├── download.sh              # CLI tool for dataset management
├── README.md                # Dataset documentation (this file)
└── [dataset_name]/          # Extracted dataset folder
    ├── cf_data/             # Collaborative Filtering interaction data
    │   ├── test.txt         # Interaction data for testing
    │   ├── train.txt        # Interaction data for training
    │   └── valid.txt        # Interaction data for validation
    ├── item_info/           # Item metadata and embeddings
    │   └── item_cf_embeds_large3_array.npy
    └── preprocessed/        # Preprocessed matrices for model input
        ├── aug_user_interact_matrix.npz  # Augmented user-item interaction matrix 
        ├── item_cf_embeds_augmented.npy  # Augmented item language representation
        └── user_interact_matrix.npz      # Original interaction matrix
```
- **aug_user_interact_matrix.npz**: Augmented user-item one-hot matrix (shape: |num_user| x |num_item|)
- **item_cf_embeds_augmented.npy**: Augmented item language representations (shape: |num_item| x |language_model_dimension|)
- **user_interact_matrix.npz**: Original user-item one-hot matrix (shape: |num_user| x |num_item|)

## ⚙️ Custom Dataset Support
If you wish to use your own custom dataset, you only need to provide the raw interaction files in the `cf_data/` directory. 

* **Requirement**: Only `train.txt`, `valid.txt`, and `test.txt` under `cf_data/` are mandatory.
* **Automation**: You **do not need** to manually build `.npz` or `.npy` files. The preprocessing pipeline will automatically create all artifacts in `item_info/` and `preprocessed/` using the data from `cf_data/`.
