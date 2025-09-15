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

### 4. Run a quick experiment
python scripts/default_run.py

---

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

## 📌 Roadmap

- [ ] Add dataset preprocessing utilities  
- [ ] Implement baseline models (MLP, LSTM)  
- [ ] Integrate state-of-the-art architectures (PatchTST, iTransformer, TimeMixer)  
- [ ] Add backtesting and evaluation metrics  

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
