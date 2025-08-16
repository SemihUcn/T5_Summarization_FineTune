# Fine-Tuning T5 for Custom Summarization

This project demonstrates how to **fine-tune a T5 model** for the **summarization task** using the **Hugging Face Transformers** library.  
The **Samsum dataset** is used, which consists of dialogues and their corresponding summaries.  
The goal is to train a model that can generate meaningful and concise summaries from dialogues.

---

## 🚀 Setup

First, create a Conda environment:

```bash
conda create -n llm-env python=3.10 -y
conda activate llm-env
```

Install the required packages:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.43.3 accelerate datasets bertviz umap-learn sentencepiece urllib3 py7zr matplotlib
```

---

## 📂 Project Structure

- **Fine Tuning T5 for Custom Summarization.ipynb**  
  Jupyter Notebook with the fine-tuning and evaluation workflow.

- **environment.yml**  
  Conda environment definition file.

- **README.md**  
  Project documentation file.

---

## 🧩 Usage

### 1. Run the model

```python
from transformers import pipeline

pipe = pipeline('summarization', model='t5_samsum_summarization', device=0)

custom_dialogue = """ 
Laxmi Kant: What work are you planning to give Tom?
Juli: I was hoping to send him on a business trip first.
Laxmi Kant: Cool. Is there any suitable work for him?
Juli: He did excellent in the last quarter. I will assign him a new project once he is back.
"""

output = pipe(custom_dialogue)
print(output)
```

### 2. Example Output

```text
[{'summary_text': 'Juli plans to send Tom on a business trip first. After he returns, she will assign him a new project.'}]
```

---

## 📊 Training Details

- **Model:** T5-small fine-tuned on the Samsum dataset.  
- **Tokenization:** `max_length=200`, `truncation=True`, `padding=max_length`.  
- **Data collator:** `DataCollatorForSeq2Seq`.  
- **Optimizer:** AdamW with learning rate scheduling.  
- **Evaluation metric:** ROUGE.

---

## 🔮 Future Work

- Fine-tuning on Turkish summarization datasets.  
- Experiments with larger models (T5-base, T5-large).  
- Adding an English-to-Turkish translation pipeline for summaries.  

---

## 📌 Requirements

- Python 3.10  
- CUDA-enabled GPU (>= 8GB VRAM recommended)  
- PyTorch, Transformers, Datasets, Accelerate, Matplotlib  

---

