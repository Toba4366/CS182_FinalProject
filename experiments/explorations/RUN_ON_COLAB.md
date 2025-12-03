# Running RNN Explorations on Google Colab

## Why Use Colab?
- ⚡ **5-10x faster** with free GPU (T4)
- 💾 **More memory** for deep models
- 🆓 **Completely free**

## Setup Instructions

### 1. Upload Your Project to Google Drive
Your project is already in Google Drive! Just note the path.

### 2. Create a Colab Notebook

Create a new notebook in Colab and run these cells:

**Cell 1: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2: Navigate to Project**
```python
import os
# Update this path to match your Google Drive structure
project_path = '/content/drive/MyDrive/Trenton Mac/CS 182/Project/CS182_FinalProject'
os.chdir(project_path)
print(f"Working directory: {os.getcwd()}")
```

**Cell 3: Verify GPU**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Cell 4: Install Dependencies (if needed)**
```python
# Usually not needed, but just in case:
# !pip install torch matplotlib seaborn pandas
```

**Cell 5: Set Python Path**
```python
import sys
sys.path.insert(0, os.getcwd())
print("Python path configured")
```

**Cell 6: Run Quick Test (GRU only, ~5 min)**
```python
!python experiments/explorations/run_gru_experiment.py \
    --d-model 256 \
    --num-layers 2 \
    --epochs 20 \
    --batch-size 8 \
    --experiment-name gru_baseline
```

**Cell 7: Run All Experiments (~30-45 min total)**
```python
!python experiments/explorations/run_all_exploration.py --mode all
```

### 3. Download Results

After experiments complete:

```python
# Zip the results
!cd experiments/explorations && zip -r results.zip results/

# Download via Colab files panel or:
from google.colab import files
files.download('experiments/explorations/results.zip')
```

Or just leave them in Google Drive and run the analysis notebook locally!

## Tips

### Enable GPU
1. Go to **Runtime → Change runtime type**
2. Select **T4 GPU** (free tier)
3. Click **Save**

### Monitor Progress
The scripts will print progress after each epoch. You'll see:
- Epoch number
- Training/validation loss
- Validation accuracy
- Time per epoch

### If Colab Disconnects
Results are saved after each experiment completes, so you can resume:
- Re-mount Drive
- Navigate to project
- Run `--mode` for remaining experiments only

### Batch Size Adjustment
With GPU, you can increase batch size for even faster training:
```python
!python experiments/explorations/run_all_exploration.py --mode all
# Edit the script to use --batch-size 16 or 32
```

## Alternative: Run Specific Experiments

If you only want certain tests:

**Capacity tests only (~15 min)**
```python
!python experiments/explorations/run_all_exploration.py --mode capacity
```

**Depth tests only (~15 min)**
```python
!python experiments/explorations/run_all_exploration.py --mode depth
```

**GRU only (~5 min)**
```python
!python experiments/explorations/run_all_exploration.py --mode gru
```

## Expected Timeline on Colab GPU

| Experiment | CPU Time | GPU Time |
|-----------|----------|----------|
| RNN 256d (baseline) | 15-20 min | 3-5 min |
| RNN 512d | 20-25 min | 4-6 min |
| RNN 1024d | 30-40 min | 5-8 min |
| RNN 2L (baseline) | 15-20 min | 3-5 min |
| RNN 5L | 20-25 min | 4-6 min |
| RNN 16L | 30-45 min | 5-8 min |
| GRU baseline | 15-20 min | 3-5 min |
| **TOTAL** | **2-2.5 hours** | **30-45 min** |

## After Completion

Once experiments finish, you can:
1. Run the analysis notebook locally (loads JSON files from Drive)
2. Or run the analysis notebook in Colab too
3. Download visualizations back to your local machine

The results will be saved in `experiments/explorations/results/` in your Google Drive!
