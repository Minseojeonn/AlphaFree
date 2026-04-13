# AlphaFree
This repository provides the official implementation of **AlphaFree**, which has been accepted to ACM The Web Conference 2026. 
* **AlphaFree: Recommendation Free from Users, IDs, and GNNs** </br>
Minseo Jeon, Junwoo Jung, Daewon Gwak, and Jinhong Jung</br>
ACM Web Conference 2026 (WWW '26)

## ⚙️ Prerequisites
You should install the required packages with a conda environment by typing the following command in your terminal:
```bash
conda create -n alphafree python=3.9
conda activate alphafree

# Install with appropriate pytorch-cuda version depending on your GPU/driver.
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

The experiments were conducted on a machine equipped with an RTX 4090 GPU (24GB VRAM), <br> running PyTorch 1.13 and CUDA 11.7.

Before running the code, you need to compile the evaluator module written in C++ and wrap it for use in Python.  <br>
To do so, execute the following command.
```bash
pushd models/base
python setup.py build_ext --inplace
popd
```

If you encounter an error that the version of libstdcxx-ng mismatches, run the following to fix it, then try again.
```bash
conda install -c conda-forge libstdcxx-ng=13.1.0
```

## 📚 Datasets
The statistics of datasets used in the paper are summarized as follows. 
| Datasets | Movie | Book | Video | Baby | Steam | Beauty | Health |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **#Users** | 26,073 | 71,306 | 94,762 | 150,777 | 334,730 | 729,576 | 796,054 |
| **#Items** | 12,464 | 40,523 | 25,612 | 36,013 | 15,068 | 207,649 | 184,346 |
| **#Inter.** | 875,906 | 2,206,865 | 814,586 | 1,241,083 | 4,216,781 | 6,624,441 | 7,176,552 |

### ⬇️ Dataset downloads
The dataset names used in the paper correspond to the following $DATASET_NAME arguments in the download script:
* Movie → amazon_movie
* Book → amazon_book_2014
* Video → amazon_video
* Baby → amazon_baby
* Steam → steam
* Beauty → amazon_beauty_personal
* Health → amazon_health

You can download the data for each dataset, including the pre-trained weights of AlphaFree, using the following command:
```bash
cd ./data
chmod +x download.sh  
./download.sh --dataset $DATASET_NAME
```

For example:
```
./download.sh --dataset amazon_movie
```
To download multiple datasets at once:
```
./download.sh --dataset steam,amazon_movie,amazon_video
```

Please refer to `data/README.md` for details on which datasets are downloaded and how they are stored.


## 🚀 Usage of AlphaFree

![overview](./assets/overview.png)

`AlphaFree` consists of three phases: Preprocessing, Training, and Inference.
* In the preprocessing phase, language representations (LRs) are generated from item titles and used to construct augmented interaction and representation views. 
* In the training phase, `AlphaFree` learns item embeddings via contrastive learning with original and augmented views. 
* In the inference phase, only the original-view encoder is used to produce embeddings for recommendation without additional augmentation or similarity computation.

In this repository, we assume that the inputs for each phase are already downloaded. For practical convenience, we introduce the stages in the order of computational cost:
> Inference → Training → Preprocessing

### 🔎 Inference phase
You can perform the inference phase using the command below. Using the pre-trained weights (`./data/pretrained/`) on the given dataset, our model is evaluated on the test split, and the performance is reported in terms of Recall@20 and NDCG@20.
```bash
python main.py --phase inference --dataset $DATASET_NAME
```

### 🏋️ Training phase
You can train `AlphaFree` from scratch using the preprocessed data (`./data/{DATASET_NAME}/`) of the given dataset and the validated hyperparameters (`./configs/`) for each dataset by typing the following command in your terminal:
```bash
python main.py --phase train --dataset $DATASET_NAME
```

**Note:** If you would like to train with different hyperparameters, please modify the configuration file (`./configs/`) accordingly.

### 🧩 Preprocessing phase
Given the raw data (i.e., user–item interactions and item titles) of a dataset (`./data/{DATASET_NAME}/item_info/item_title.json`), the preprocessing phase generates LRs using a language model and performs augmentation. This phase can be executed using the following command:

```bash
python main.py --phase preprocessing --dataset $DATASET_NAME
```

**Note1:** The LR generation is not supported for the amazon_book_2014 and amazon_movie datasets, since we reuse the LRs provided by [AlphaRec](https://github.com/LehengTHU/AlphaRec) for both datasets. <br> 
**Note2:** For other datasets, we used LLaMA-3-8B. If you would like to use a different LM, please modify `data.py` (see 280 line) accordingly. <br>
**Note3:** If you would like to apply `AlphaFree` to your own dataset, please prepare the raw data according to the required format and run the preprocessing phase first.

The dimensionality of the generated LRs varies depending on the language model. The dimension $d_{LR}$ used for each dataset is as follows.
| Datasets | **Movie**  |**Book**   | **Video**  | **Baby**   | **Steam**  | **Beauty** | **Health** |
|:-------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| $d_{LR}$      | 3072 | 3072 | 4096 | 4096 |4096 |4096 | 4096 |


### ✅ Demo

You can run the demo of AlphaFree via `demo.py`. This CLI-based demo takes as input a list of items that a user has consumed (or selected) from the Amazon Video dataset and generates recommendation results using a pre-trained AlphaFree model. To clearly indicate that we use only the original-view encoder for inference, we provide a separate model implementation located at:
```
./models/AlphaFreeRecDemo.py
```
You can run the demo as follows:
```bash
python demo.py 
```
**Note**: Before running the demo, please download the Amazon Video dataset in advance.

<img src="./assets/demo.gif" alt="AlphaFree demo" width="800" />

## 📈 Experimental Results of `AlphaFree`

### Performance for top-k item recommendation
The reported results in the paper are as follows:

| **Recall@20** | **Movie**  |**Book**   | **Video**  | **Baby**   | **Steam**  | **Beauty** | **Health** |
|-------------|--------|--------|--------|--------|--------|--------|--------|
| MF-BPR      | 0.0580 | 0.0436 | 0.0177 | 0.0150 | 0.1610 | 0.0063 | 0.0091 |
| FISM-BPR    | 0.0861 | 0.0623 | 0.0392 | 0.0150 | 0.1801 | 0.0079 | 0.0104 |
| LightGCN    | 0.0860 | 0.0712 | 0.0732 | 0.0359 | 0.2013 | 0.0201 | 0.0250 |
| XSimGCL     | 0.0967 | 0.0818 | 0.0897 | 0.0390 | 0.2245 | 0.0253 | 0.0299 |
| RLMRec      | 0.1046 | 0.0905 | o.o.t. | o.o.t. | o.o.t. | o.o.t. | o.o.t. |
| AlphaRec    | 0.1219 | 0.0991 | 0.1088 | 0.0391 | 0.2360 | o.o.m. | o.o.m. |
| **AlphaFree** | **0.1267** | **0.1014** | **0.1111** | **0.0412** | **0.2402** | **0.0361** | **0.0325** |
| **% inc. over best LR** | **3.94%** | **2.32%** | **2.11%** | **5.37%** | **1.78%** | **-** | **-** |
| **% inc. over best non-LR** | **31.00%** | **23.96%** | **23.81%** | **5.58%** | **6.99%** | **42.69%** | **8.70%** |

| **NDCG@20** | **Movie**  |**Book**   | **Video**  | **Baby**   | **Steam**  | **Beauty** | **Health** |
|-------------|--------|--------|--------|--------|--------|--------|--------|
| MF-BPR      | 0.0533 | 0.0387 | 0.0095 | 0.0074 | 0.1537 | 0.0032 | 0.0048 |
| FISM-BPR    | 0.0801 | 0.0540 | 0.0220 | 0.0076 | 0.1431 | 0.0044 | 0.0060 |
| LightGCN    | 0.0754 | 0.0597 | 0.0409 | 0.0185 | 0.1518 | 0.0110 | 0.0142 |
| XSimGCL     | 0.0866 | 0.0690 | 0.0500 | 0.0203 | 0.1664 | 0.0140 | 0.0170 |
| RLMRec      | 0.0942 | 0.0741 | o.o.t. | o.o.t. | o.o.t. | o.o.t. | o.o.t. |
| AlphaRec    | 0.1141 | 0.0829 | 0.0605 | 0.0210 | 0.1884 | o.o.m. | o.o.m. |
| **AlphaFree** | **0.1194** | **0.0861** | **0.0615** | **0.0219** | **0.1938** | **0.0200** | **0.0184** |
|  **% inc. over best LR** | **4.65%** | **3.86%** | **1.65%** | **5.71%** | **2.89%** | **-** | **-** |
| **% inc. over best non-LR**  | **37.90%** | **24.78%** | **23.07%** | **9.26%** | **16.49%** | **42.86%** | **8.24%** |

* o.o.t. : Out of Time
* o.o.m. : Out of Memory (VRAM)

### Logs in training
You can find the training logs of our model in the `./log` directory. The test performance of the pre-trained AlphaFree that produced these logs (based on `./log`) is summarized below.  
| **AlphaFree** | **Movie**  |**Book**   | **Video**  | **Baby**   | **Steam**  | **Beauty** | **Health** |
|-------------|--------|--------|--------|--------|--------|--------|--------|
| **Recall@20** | 0.1263 | 0.1023 | 0.1114 | 0.0414 | 0.2402 | 0.0371 | 0.0333 |
| **NDCG@20** | 0.1190 | 0.0871 | 0.0616 | 0.0223 | 0.1938 | 0.0205 | 0.0189 |

You can also verify the recommendation performance using the provided pre-trained weights (see **Inference phase**). Note that due to the non-deterministic behavior of PyTorch, the reported performance may vary slightly within a small margin.


### Validated hyperparameters of AlphaFree
We report the validated hyperparameters for each dataset, selected based on validation performance measured by Recall@20, in the table below.

| **Hyperparam** | **Movie**  |**Book**   | **Video**  | **Baby**   | **Steam**  | **Beauty** | **Health** |
|:-------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| $K_c$      | 5 | 5 | 10 | 10 | 3 | 10 | 5 |
| $\lambda_{\texttt{align}}$    | 0.2 | 0.2 | 0.05 | 0.01 | 0.01 | 0.01 | 0.01 |
| $\tau_a$    | 0.2 | 0.1 | 0.01 | 0.2 | 0.2 | 0.1 | 0.2 |
| $\tau_r$     | 0.15 | 0.15 | 0.2 | 0.2 | 0.2 | 0.2 | 0.15 |

#### Description of each hyperparameter
* $K_c$ : The number of similar items (`--K_c`).
* $\lambda_{\texttt{align}}$ : The alignment loss weight (`--lambda_align`).
* $\tau_a$ : The alignment temperature (`--tau_a`).
* $\tau_r$ : The recommendation temperature (`--tau_r`).


## 📚 Citation
If you find this work useful, please cite:
```bibtex
@article{journals/corr/abs-2603-02653,
  author       = {Minseo Jeon and
                  Junwoo Jung and
                  Daewon Gwak and
                  Jinhong Jung},
  title        = {AlphaFree: Recommendation Free from Users, IDs, and GNNs},
  journal      = {CoRR},
  volume       = {abs/2603.02653},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2603.02653},
  doi          = {10.48550/ARXIV.2603.02653},
  eprinttype   = {arXiv},
  eprint       = {2603.02653},
  timestamp    = {Wed, 08 Apr 2026 10:55:11 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2603-02653.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
