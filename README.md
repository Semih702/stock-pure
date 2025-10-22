# STOCK-PURE

A clean and minimal PyTorch-based project template for experimenting with **time series forecasting** and **stock market prediction**.  
The current focus is on building a modular codebase that can easily integrate **state-of-the-art deep learning models** (e.g., Transformers for time series).

---

## 📂 Project Structure

```text
STOCK-PURE/
├─ configs/            # Configuration files (YAML)
│  └─ default.yaml
├─ scripts/            # Helper scripts for running experiments
│  └─ default_run.py
├─ src/                # Source code
│  ├─ data/            # Dataset loading and preprocessing
│  ├─ models/          # Model definitions (e.g. MLP, Transformers)
│  ├─ utils/           # Utilities (seed, logging, checkpointing, metrics)
│  └─ main.py          # Entry point for training / evaluation
├─ requirements.txt    # Python dependencies
└─ README.md           # Project documentation
```

---

## ⚙️ Frameworks & Libraries

This project is primarily built using:

- **Python 3.10+**
- **PyTorch** → core deep learning framework
- **Torchvision** → dataset utilities and transforms
- **NumPy / Pandas** → data processing
- **PyYAML** → config file handling
- **TQDM** → progress bars for training/evaluation

---

## 🚀 Getting Started

### 1. Clone the repository
git clone https://github.com/your-username/stock-pure.git
cd stock-pure

### 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt
pip install -e .

### 4. Run a quick experiment
python scripts/default_run.py

---

## 📊 Fetching Stock Market Data (Yahoo Finance)

To download historical stock data from **Yahoo Finance** and store it locally for experiments, run:

```bash
python src/scripts/fetch_yahoo_stock_data.py
```

This script retrieves daily OHLCV data (Open, High, Low, Close, Volume) for several major tickers such as Apple, Tesla, Microsoft, and Google, starting from **January 1, 2020** up to the current date.

The data will be automatically saved under:

```
src/data/yahoo-finance/
├── apple.csv
├── tesla.csv
├── meta.csv
├── ...
```

Each file contains time-indexed historical prices with all standard columns provided by Yahoo Finance.

**Note:**  
- Ensure you have internet access and `yfinance` installed (`pip install yfinance`).  
- You can modify the script to change the date range or the list of tickers.

## 🧠 Quick TimesNet Training (Single-CSV Experiment)

Once your Yahoo Finance data is ready, you can train a small TimesNet-based model on any single stock CSV using the helper script below.  
This performs an **automatic train/val/test split**, builds sliding windows, and trains a minimal TimesNet variant (`TinyForecastSP`) for a few epochs.

### ▶️ Run a quick test

```bash
python src/scripts/train_timesnet_from_csv.py \
  --in-dir src/data/yahoo-finance \
  --name apple.csv \
  --context 96 \
  --horizon 24 \
  --epochs 3 \
  --batch-size 64 \
  --lr 1e-3 \
  --features Close Open High Low Volume \
  --target Close \
  --device cpu
```

This will:

- Automatically split `apple.csv` into `train`, `val`, and `test` under:
  ```
  src/data/yahoo-finance/splits/apple/
  ```
- Train a small TimesNet model for **3 epochs** (quick smoke test).
- Save a report with metrics such as **MAE**, **RMSE**, and **MAPE** under:
  ```
  results/quickrun/timesnet_from_csv.json
  ```

To skip re-splitting if the data already exists, just add:
```bash
--skip-split
```

### 💡 Example output

```
🔧 Splitting src/data/yahoo-finance/apple.csv into train/val/test...
epoch 1/3 | train L1=1.2345
epoch 2/3 | train L1=0.9831
epoch 3/3 | train L1=0.9512
val L1=0.9974
Test metrics: MAE=0.9521 | RMSE=1.2243 | MAPE=0.0357
Wrote: results/quickrun/timesnet_from_csv.json
```

**Tip:**  
Run with `--device cuda` if a GPU is available for faster experiments.

## Running Benchmarks

You can run both the stockpure implementation and the upstream TimesNet implementation benchmarks using the helper script:

```text
python scripts/run_timesnet_bench.py \
  --upstream-root .cache/upstream_timesnet \
  --device cpu \
  --epochs 3 \
  --write-report
```

- --upstream-root → path to the cloned upstream TimesNet repo.

- --device → choose cpu or cuda.

- --epochs → number of epochs to train for quick smoke tests.

- --write-report → generate a merged Markdown report in results/bench/.

This will create JSON and Markdown reports under results/bench/ for easy comparison.

## 🤝 Contributing

The common practice in ML/DL projects is to **prototype first in Jupyter notebooks**, then refactor stable code into the main project structure.  
This ensures fast iteration while keeping the repository clean and reproducible.

### Typical Workflow

1. **Explore in Jupyter**
   - Use a notebook (e.g. `notebooks/01-data-exploration.ipynb`) to:
     - Inspect and visualize raw data.
     - Prototype small model variants (MLP, LSTM, Transformer).
     - Debug shapes, losses, and metrics.
   - Notebooks are for **experiments and validation**, not long-term code storage.

2. **Move code into `src/`**
   - Once something works in a notebook, refactor it into the project:
     - Data preprocessing → `src/data/`
     - Model definitions → `src/models/`
     - Training loop logic → `src/train_loop.py`
   - Keep notebooks for demos, but treat `src/` as the **source of truth**.

3. **Integrate with scripts/configs**
   - Add configs in `configs/` for hyperparameters.
   - Run training via `src/main.py` or a helper in `scripts/`.
   - Update `requirements.txt` if new dependencies were added.

4. **Validate before commit**
   - Run a short local training (1–2 epochs) to make sure nothing breaks.
   - Optionally add a minimal test (e.g. forward pass with dummy data).

5. **Commit and push**
   ```bash
   git checkout -b feature/my-new-feature
   git add .
   git commit -m "Add PatchTST model and integrate with training loop"
   git push origin feature/my-new-feature

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
