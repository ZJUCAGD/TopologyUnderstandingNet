# Code of "TUN: Detecting Significant Points in Persistence Diagrams with Deep Learning"

### Project Structure
```
persistence_significance_detection/
├─ configs/
├─ data/
├─ results/
├─ scripts/
├─ src/
├─ test_exist_model.ipynb
└─ train_and_test.ipynb
```


### Requirements
- Python >= 3.8
- PyTorch >= 2.8.0
- CUDA support (optional, for GPU acceleration)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Main dependencies:
```
torch>=2.8.0
torchvision>=0.22.1
numpy>=2.0.2
matplotlib>=3.3.0
scikit-learn>=0.24.0
tqdm>=4.60.0
tensorboard>=2.5.0
seaborn>=0.11.0
pandas>=1.3.0
scipy>=1.7.0
plotly>=5.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
```


### 2. Prepare Data
#### Training data download (Google Drive): [data.zip](https://drive.google.com/file/d/1_I9JjBFIBkmxwPYljdZAPyrHhLe9eFah/view?usp=sharing).  
After downloading, extract the archive and place the `data` folder in the project root directory.

### 3. Train and Test
A trained best TUN model is provided at `results\checkpoints\best_model_1D.pth`.  
To evaluate it on the four test sets, run the Jupyter notebook `test_exist_model.ipynb`.

To train your own model and evaluate it, run the Jupyter notebook `train_and_test.ipynb` and follow the prompts.  
Trained models are saved to `results\checkpoints\best_model_1D.pth`.
