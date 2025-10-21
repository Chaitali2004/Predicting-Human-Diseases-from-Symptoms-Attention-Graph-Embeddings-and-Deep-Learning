# Predicting-Human-Diseases-from-Symptoms-Attention-Graph-Embeddings-and-Deep-Learning


**Project Overview**
This project demonstrates a novel approach for **predicting human diseases from patient-reported symptoms** using a combination of **attention-weighted symptom embeddings**, **severity scaling**, **symptom co-occurrence graph embeddings**, and **structured features**. A **deep neural network** is used to integrate these heterogeneous features and predict diseases, while explainability techniques provide clinical insights.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Pipeline](#pipeline)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Visualization & Explainability](#visualization--explainability)
7. [Future Work](#future-work)

---

## Motivation

Accurate early disease prediction is crucial for healthcare. This project aims to:

* Understand relationships between symptoms and diseases
* Leverage **severity information** for symptom importance
* Provide interpretable predictions for clinical insights

---

## Dataset

* **Source:** Pre-generated symptom-based human disease dataset
* **Columns:** `Symptom_1` … `Symptom_17`, `severity`, `Disease`
* **Preprocessing:** Missing symptoms filled with 'none', symptom lists created per patient

---

## Pipeline

1. **Symptom Embedding:** Use **GloVe 50d embeddings** to convert symptom text to vectors.
2. **Attention × Severity:** Compute patient-specific attention scores and scale by severity.
3. **Symptom Co-occurrence Graph:** Build co-occurrence matrix and reduce dimensionality with **TruncatedSVD**.
4. **Structured Features:** Encode symptom presence with **one-hot vectors**.
5. **Feature Combination:** Concatenate **attention embeddings**, **graph embeddings**, and **structured features**.
6. **Deep Neural Network:** Fully connected network predicts diseases using combined features.
7. **Explainability:**

   * Attention heatmaps for patients
   * SHAP analysis
   * Top-k predictive symptoms per disease

---

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <your-repo-folder>

# Create environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

* Python ≥ 3.8
* pandas, numpy, matplotlib, networkx
* scikit-learn
* gensim
* tensorflow / keras
* shap

---

## Results

**Example metrics:**

```
Accuracy: 0.95
Precision, Recall, F1-score per disease: see classification_report output

```
  <img width="818" height="819" alt="image" src="https://github.com/user-attachments/assets/1be9641d-5b22-4bdf-8a35-9a24d583cd30" />

**Example Usage**
<img width="705" height="432" alt="image" src="https://github.com/user-attachments/assets/5dcda35a-2334-49cc-8133-0a7fc39c4cca" />


**Example visualizations:**

* Model training metrics (accuracy, classification report)
  <img width="818" height="819" alt="image" src="https://github.com/user-attachments/assets/1be9641d-5b22-4bdf-8a35-9a24d583cd30" />

* Confusion Matrix

Confusion matrix to evaluate how well a classification model performs by detailing its true positive, true negative, false positive, and false negative predictions, offering a more comprehensive understanding than simple accuracy.
  
<img width="754" height="455" alt="image" src="https://github.com/user-attachments/assets/e0aa901c-7019-4dd0-80ae-a52b6489d5ed" />

  
* Attention × severity heatmaps

Highlights which symptoms are most important for individual patients.
  <img width="957" height="701" alt="image" src="https://github.com/user-attachments/assets/e8468965-c2b0-42ea-b73d-fdc50f268a68" />

* Top-k predictive symptoms

Shows the most predictive symptoms per disease.
  <img width="809" height="393" alt="image" src="https://github.com/user-attachments/assets/b26f4526-6bab-4bef-9997-10f7d001f4f6" />
  <img width="789" height="393" alt="image" src="https://github.com/user-attachments/assets/ea939dd5-4fdf-4fc5-a84f-614211ef0e52" />
  <img width="717" height="393" alt="image" src="https://github.com/user-attachments/assets/11ef672c-6ac4-418b-89b8-b9321549a627" />

* SHAP plots

Quantifies feature contribution globally and locally.
  <img width="826" height="182" alt="image" src="https://github.com/user-attachments/assets/6129747f-3525-482f-b9ab-0b4b138e858b" />

* t-SNE visualization of patient embeddings

Visualizes patient embeddings in 2D space, colored by disease.
  <img width="1203" height="912" alt="image" src="https://github.com/user-attachments/assets/18bb2fdd-d6ac-4061-bd73-b4d986d50e08" />


---

## Future Work

* Integrate **temporal symptom progression** for longitudinal predictions.
* Explore **multi-branch neural networks** for structured vs. embedding inputs.
* Apply pipeline to larger, real-world datasets for clinical validation.

---

🧑‍💻 Author

Chaitali Jain B.Tech (Engineering Student) AI & Data Science Enthusiast

📜 License

MIT License © 2025 Chaitali Jain
