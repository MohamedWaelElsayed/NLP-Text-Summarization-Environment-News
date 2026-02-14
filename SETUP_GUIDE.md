# Setup & Implementation Guide

## Quick Start

### Option 1: Google Colab (Recommended) ⭐

**Advantages**: Free GPU, no installation, cloud-based

```
1. Go to: https://colab.research.google.com/drive/1tYMNYayKk7C9D9PtqMBKrp-zSYTu3rd9?usp=sharing
2. Click "Copy to Drive"
3. Run cells sequentially
4. All dependencies pre-configured ✅
5. GPU automatically enabled
```

**Expected Runtime**: ~3-4 hours (full notebook)

---

### Option 2: Local Machine Setup

#### Prerequisites

```bash
# System Requirements
- Python 3.10+ (tested on 3.12)
- CUDA 12.0+ (for GPU acceleration)
- 8GB RAM minimum (16GB+ recommended)
- 50GB disk space (for dataset + models)

# Verify Python
python --version
python -m pip --version
```

#### Installation Steps

**Step 1: Clone Repository**

```bash
git clone <repository-url>
cd NLP-Text-Summarization-Environment-News
```

**Step 2: Create Virtual Environment**

```bash
# Using venv (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# OR using conda
conda create -n nlp-env python=3.12
conda activate nlp-env
```

**Step 3: Install Dependencies**

**Method A: Using uv (fastest)**

```bash
pip install uv
uv pip install -r requirements.txt
```

**Method B: Using pip**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Method C: Manual Installation**

```bash
# Core ML Libraries
pip install tensorflow[and-cuda] torch torchvision torchaudio
pip install transformers sentencepiece datasets
pip install evaluate accelerate
pip install rouge_score

# Data Processing
pip install kagglehub numpy pandas matplotlib seaborn

# Jupyter
pip install jupyter jupyterlab ipykernel

# Optional (for visualization)
pip install plotly scikit-learn nltk
```

**Step 4: Verify Installation**

```bash
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "from transformers import AutoModel; print('Transformers installed ✓')"
```

**Step 5: Set up Kaggle API (for data)**

```bash
# Create Kaggle account if not exists
# Go to: https://www.kaggle.com/account

# Download kaggle.json from account settings
# Place in ~/.kaggle/kaggle.json

# Set permissions (Linux/Mac)
chmod 600 ~/.kaggle/kaggle.json

# Verify
kaggle datasets list | head
```

---

## Running the Project

### Full Pipeline (All Milestones)

```bash
# Launch Jupyter
jupyter-notebook NLP_Project.ipynb

# OR use JupyterLab (recommended)
jupyter-lab NLP_Project.ipynb
```

### Run Specific Milestone

```python
# In Jupyter, identify milestone by markdown headers
# Each milestone clearly marked with "# **Milestone N**"

# Milestone 1: Run cells 1-24
# Milestone 2: Run cells 25-85
# Milestone 3: Run cells 86-145
```

### Command-Line Execution (Optional)

```bash
# Convert notebook to Python script
jupyter nbconvert --to script NLP_Project.ipynb

# Run specific section (edit main.py accordingly)
python NLP_Project.py
```

---

## Troubleshooting

### Issue 1: CUDA Not Detected

```
Error: CUDA device not available

Solution:
1. Verify GPU: nvidia-smi
2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Install cuDNN: https://developer.nvidia.com/cudnn
4. Reinstall PyTorch with CUDA:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: Memory Error During Training

```
Error: CUDA out of memory / RuntimeError: CUDA out of memory

Solutions:
1. Reduce batch size in hyperparameters:
   batch_size = 30 (instead of 60 for M2)
   per_device_batch_size = 2 (instead of 4 for M3)

2. Clear CUDA cache between runs:
   import torch
   torch.cuda.empty_cache()

3. Use CPU (slower but memory unlimited):
   device = "cpu"

4. Use Google Colab (has more GPU memory)
```

### Issue 3: Kaggle API Error

```
Error: Kaggle API token not found

Solution:
1. Go to https://www.kaggle.com/account
2. Download kaggle.json
3. Place in ~/.kaggle/kaggle.json
4. Run: chmod 600 ~/.kaggle/kaggle.json
5. Test: kaggle datasets list

Or in Colab:
from google.colab import files
files.upload()  # Upload kaggle.json
! mkdir -p ~/.kaggle
! mv kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
```

### Issue 4: ImportError for Modules

```
Error: ModuleNotFoundError: No module named 'X'

Solution:
1. Identify missing module (from error message)
2. Install: pip install <module-name>
3. Restart kernel: Kernel → Restart (in Jupyter)
4. Re-run cell

For common issues:
nltk: pip install nltk (then: nltk.download('punkt'))
sklearn: pip install scikit-learn
torch: pip install torch (see CUDA installation above)
```

### Issue 5: Long Training Time

```
If M3 fine-tuning takes >1 hour:

Optimization options:
1. Reduce epochs: num_train_epochs = 5 (instead of 10)
2. Reduce samples: use fewer training samples
3. Skip validation: eval_strategy = "no"
4. Use GPU: ensure CUDA is available
5. Use Colab: Always faster for this project
```

---

## Model Inference Examples

### Using Pre-trained M3 Model

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model
model_path = "my_fine_tuned_bart"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example text
text = """
Recent studies show that global carbon emissions
are accelerating despite renewable energy investments.
Climate scientists warn that immediate policy
action is required to limit warming to 1.5°C.
"""

# Generate summary
inputs = tokenizer(text, return_tensors="pt",
                   max_length=1024, truncation=True)
inputs = inputs.to(device)

summary_ids = model.generate(inputs["input_ids"],
                            max_length=60, num_beams=4)
summary = tokenizer.decode(summary_ids[0],
                          skip_special_tokens=True)

print("Original:", text)
print("Summary:", summary)
```

### Batch Processing

```python
# Process multiple texts efficiently

texts = [
    "Text 1 about environmental policy...",
    "Text 2 about renewable energy...",
    "Text 3 about climate change...",
]

# Tokenize batch
inputs = tokenizer(texts, return_tensors="pt",
                   padding=True, truncation=True,
                   max_length=1024)
inputs = inputs.to(device)

# Generate summaries
summary_ids = model.generate(inputs["input_ids"],
                            max_length=60, num_beams=4)

# Decode all
summaries = [tokenizer.decode(ids, skip_special_tokens=True)
             for ids in summary_ids]

for text, summary in zip(texts, summaries):
    print(f"Input: {text[:50]}...")
    print(f"Summary: {summary}\n")
```

### Using Transformers Pipeline (Simplified)

```python
from transformers import pipeline

# Create summarization pipeline
summarizer = pipeline("summarization",
                     model="my_fine_tuned_bart")

# Generate summary
result = summarizer("Your environmental news text...",
                   max_length=60,
                   min_length=30,
                   do_sample=False)

print(result[0]['summary_text'])
```

---

## Performance Optimization

### For Training

```python
# Mixed Precision (2x faster on modern GPUs)
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    fp16=True,  # Enable automatic mixed precision
    gradient_accumulation_steps=4,  # Larger effective batch
    learning_rate=3e-5,
    # ... other args
)

# Gradient Checkpointing (saves memory)
model.gradient_checkpointing_enable()

# DataLoader with pinned memory
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, pin_memory=True)
```

### For Inference

```python
# ONNX Export (2x faster)
from optimum.onnxruntime import ORTModelForSeq2SeqLM

model = ORTModelForSeq2SeqLM.from_pretrained(
    "my_fine_tuned_bart", export=True
)

# Quantization (smaller model, faster inference)
from torch.quantization import quantize_dynamic
quantized = quantize_dynamic(model, {torch.nn.Linear},
                            dtype=torch.qint8)

# Caching (repeated queries)
from functools import lru_cache

@lru_cache(maxsize=1000)
def summarize_cached(text):
    return summarize(text)
```

---

## Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY my_fine_tuned_bart/ ./model/
COPY app.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

```bash
# Build image
docker build -t nlp-summarizer .

# Run container
docker run -p 5000:5000 --gpus all nlp-summarizer
```

### FastAPI Server

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = FastAPI()

# Load model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained("my_fine_tuned_bart")
tokenizer = AutoTokenizer.from_pretrained("my_fine_tuned_bart")
model = model.to(device)

class TextInput(BaseModel):
    text: str
    max_length: int = 60
    num_beams: int = 4

@app.post("/summarize")
def summarize(input_data: TextInput):
    inputs = tokenizer(input_data.text, return_tensors="pt",
                      truncation=True)
    inputs = inputs.to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=input_data.max_length,
        num_beams=input_data.num_beams
    )

    summary = tokenizer.decode(summary_ids[0],
                             skip_special_tokens=True)

    return {"summary": summary}

# Run: uvicorn app:app --reload
```

```bash
# Test the API
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your environmental news..."}'
```

---

## File Structure After Setup

```
NLP-Text-Summarization-Environment-News/
├── NLP_Project.ipynb              ← Main notebook
├── README.md                      ← Project overview
├── ARCHITECTURE.md                ← Technical details
├── RESULTS.md                     ← Performance analysis
├── SETUP_GUIDE.md                 ← This file
├── requirements.txt               ← Python dependencies
├── my_fine_tuned_bart/           ← Trained model (M3)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.json
├── Milestone_1/
│   └── README.md
├── Milestone_2/
│   └── README.md
├── Milestone_3/
│   └── README.md
└── colab_link.md

# Generated during execution (in Jupyter):
├── model_architecture_m2.png      ← M2 model diagram
└── [Cache files & datasets]
```

---

## Next Steps

1. **Complete Setup** ✅
   - Run Milestone 1 (data exploration)
   - Takes ~2 hours

2. **Train Custom Model** (Optional)
   - Run Milestone 2 (LSTM)
   - Takes ~2-2.5 hours
   - Learn deep learning fundamentals

3. **Deploy Production Model** ✅
   - Run Milestone 3 (BART fine-tuning)
   - Takes ~1.5 hours
   - Better performance than M2

4. **Evaluate & Compare**
   - Run ROUGE evaluation
   - Compare M2 vs M3
   - Understand trade-offs

5. **Deploy to Production** (Optional)
   - Use FastAPI + Docker
   - Create API endpoint
   - Monitor performance

---

## Quick Reference

### Common Commands

```bash
# Activate environment
source env/bin/activate  # Linux/Mac
env\Scripts\activate      # Windows

# Install packages
pip install transformers torch

# Launch Jupyter
jupyter notebook

# Run specific cell in Colab
# Click play button next to cell (or Ctrl+Enter)

# Download trained model
from google.colab import files
files.download("my_fine_tuned_bart/pytorch_model.bin")

# Upload to Colab
files.upload()
```

### Useful Hyperparameters

| Parameter         | M2 Value | M3 Value | Impact                                |
| ----------------- | -------- | -------- | ------------------------------------- |
| batch_size        | 60       | 4        | ↓ = slower training, ↑ = memory error |
| learning_rate     | N/A      | 3e-5     | ↓ = slower, ↑ = instability           |
| epochs            | 40       | 10       | ↑ = longer training                   |
| max_input_length  | 50       | 512      | ↑ = slower, more memory               |
| max_output_length | 20       | 60       | ↑ = longer summaries                  |
| num_beams         | -        | 4        | ↑ = better quality, slower            |

---

## Support & Resources

### Official Documentation

- TensorFlow: https://www.tensorflow.org/
- PyTorch: https://pytorch.org/
- Transformers: https://huggingface.co/docs/transformers
- BART Model: https://huggingface.co/facebook/bart-large-cnn

### Community

- GitHub Issues: https://github.com/[repo]/issues
- Hugging Face Forums: https://discuss.huggingface.co/
- Stack Overflow: [tag:transformers] [tag:pytorch]

### Related Papers

- BART: https://arxiv.org/abs/1910.13461
- Attention: https://arxiv.org/abs/1706.03762
- LSTM: https://arxiv.org/abs/seqQtop
