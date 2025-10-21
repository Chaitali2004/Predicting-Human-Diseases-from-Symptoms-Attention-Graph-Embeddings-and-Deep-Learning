# %%
# -------------------- Full Research-Novel Pipeline --------------------
# Patient Disease Prediction using:
# 1. Attention-weighted Symptom Embeddings scaled by Severity
# 2. Symptom Co-occurrence Graph Embeddings (TruncatedSVD)
# 3. Structured Symptom Features (One-hot Encoding)
# 4. Deep Neural Network Classifier for multi-class disease prediction
# 5. Explainability & Visualization:
#    - Attention × Severity heatmaps per patient
#    - SHAP feature importance
#    - Top-k predictive symptoms per disease
#    - t-SNE visualization of patient embeddings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import gensim.downloader as api
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# -------------------- Settings --------------------
DATA_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms.csv"
symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]
embedding_name = "glove-wiki-gigaword-50"   # 50d, lightweight
svd_graph_dims = 16                         # dimension for symptom graph embeddings
rf_n_estimators = 200
random_state = 42

# -------------------- 1. Load data --------------------
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# Ensure disease label column exists (try 'Disease' then 'diseases')
label_col = 'Disease' if 'Disease' in df.columns else ('diseases' if 'diseases' in df.columns else None)
if label_col is None:
    raise KeyError("No column named 'Disease' or 'diseases' found. Rename your target column to 'Disease'.")
print("Label column:", label_col)

# -------------------- 2. Prepare symptom columns --------------------
# Make sure symptom columns exist; if missing create placeholder 'none'
for col in symptom_cols:
    if col not in df.columns:
        df[col] = 'none'
    df[col] = df[col].fillna('none').astype(str)

# Build symptom lists per patient (list of strings)
symptom_lists = df[symptom_cols].apply(lambda row: [str(s).strip() for s in row if str(s).strip().lower() not in ['', 'none', 'nan']], axis=1)

# -------------------- 3. Symptom frequency -> per-symptom severity mapping (if no explicit severity) --------------------
# If df already has a 'severity' column at sample-level, we'll still build per-symptom severity mapping (global)
if 'severity' in df.columns:
    print("Found 'severity' column in data; sample-level severity will be used where relevant.")
else:
    print("No sample-level 'severity' column found; building per-symptom severity mapping from frequency (1..5).")

# Flatten symptom occurrences to compute frequency
all_symptoms_flat = [s.lower() for lst in symptom_lists for s in lst]
symptom_counts = pd.Series(all_symptoms_flat).value_counts()
# Map symptom -> severity 1..5 by quantiles
quantiles = symptom_counts.quantile([0.2, 0.4, 0.6, 0.8]).values
def freq_to_sev(cnt):
    if cnt <= quantiles[0]:
        return 1
    elif cnt <= quantiles[1]:
        return 2
    elif cnt <= quantiles[2]:
        return 3
    elif cnt <= quantiles[3]:
        return 4
    else:
        return 5
symptom_to_severity = {sym: freq_to_sev(int(cnt)) for sym, cnt in symptom_counts.items()}

# create per-patient per-symptom severity arrays (aligned to symptom_list ordering)
def get_symptom_severity_array(symptom_list):
    return np.array([symptom_to_severity.get(s.lower(), 1) for s in symptom_list], dtype=float)

# -------------------- 4. Load pre-trained embeddings (GloVe 50d) --------------------
print("Loading embeddings (this may download the model first time)...")
w2v = api.load(embedding_name)
embed_dim = w2v.vector_size
print("Embedding dim:", embed_dim)

def embed_symptom(symptom_str):
    """Average word embeddings for words in symptom_str; fallback zero vector."""
    words = [w for w in str(symptom_str).lower().split() if w in w2v]
    if not words:
        return np.zeros(embed_dim, dtype=float)
    vecs = np.vstack([w2v[w] for w in words])
    return vecs.mean(axis=0)

# -------------------- 5. Attention-weighted embedding per patient (with severity scaling) --------------------
def attention_severity_embedding(symptom_list):
    """
    For an input symptom_list (list of symptom strings):
    - embed each symptom (via embed_symptom)
    - compute attention scores: dot(emb_i, mean_context)
    - softmax -> attn weights
    - multiply attn weights by (normalized) per-symptom severity
    - return weighted sum of symptom vectors
    """
    if len(symptom_list) == 0:
        return np.zeros(embed_dim, dtype=float)
    # embed each symptom
    vecs = np.vstack([embed_symptom(s) for s in symptom_list])  # shape (k, D)
    # compute context vector (mean)
    context = vecs.mean(axis=0)
    # attention scores: similarity of each symptom to context
    scores = vecs.dot(context)
    # numerical stability for softmax
    exp_scores = np.exp(scores - np.max(scores))
    attn = exp_scores / (np.sum(exp_scores) + 1e-12)  # shape (k,)
    # severity array for symptoms
    sev = get_symptom_severity_array(symptom_list)  # values 1..5
    # normalize severity to [0.5,1.5] to avoid zeroing out (optional)
    sev_norm = 0.5 + (sev - 1) / 4.0  # maps 1->0.5, 5->1.5
    # combine attention with severity
    combined_weights = attn * sev_norm
    # normalize combined_weights to sum to 1
    combined_weights = combined_weights / (combined_weights.sum() + 1e-12)
    # weighted sum
    weighted_vec = (vecs * combined_weights[:, None]).sum(axis=0)
    return weighted_vec

# build attention embeddings matrix
print("Building attention-weighted embeddings for each patient...")
X_att_embeddings = np.vstack([attention_severity_embedding(lst) for lst in symptom_lists])

# -------------------- 6. Symptom co-occurrence graph -> SVD graph embeddings --------------------
# Build co-occurrence matrix over the unique symptom set
unique_symptoms = sorted(symptom_counts.index.tolist())  # lowercase symptoms
symptom_index = {sym: idx for idx, sym in enumerate(unique_symptoms)}
n_sym = len(unique_symptoms)
print("Number of unique symptom tokens:", n_sym)

# Initialize co-occurrence matrix
cooc = np.zeros((n_sym, n_sym), dtype=float)
# For each patient increment co-occurrence counts for pairs of present symptoms
for lst in symptom_lists:
    lower = [s.lower() for s in lst]
    idxs = [symptom_index[s] for s in lower if s in symptom_index]
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            cooc[idxs[i], idxs[j]] += 1
            cooc[idxs[j], idxs[i]] += 1

# Optionally apply small smoothing
cooc += 1e-6

# Reduce co-occurrence to low-dim embeddings (TruncatedSVD)
print("Computing graph embeddings via TruncatedSVD (dim={})...".format(svd_graph_dims))
svd = TruncatedSVD(n_components=min(svd_graph_dims, n_sym-1), random_state=random_state)
symptom_graph_embs = svd.fit_transform(cooc)  # shape (n_sym, svd_graph_dims)

# For a patient, create graph-based features by summing symptom node embeddings of present symptoms
def patient_graph_features(symptom_list):
    idxs = [symptom_index[s.lower()] for s in symptom_list if s.lower() in symptom_index]
    if not idxs:
        return np.zeros(svd_graph_dims, dtype=float)
    return symptom_graph_embs[idxs].sum(axis=0)

X_graph_feats = np.vstack([patient_graph_features(lst) for lst in symptom_lists])

# -------------------- 7. Structured features (MultiLabelBinarizer) --------------------
mlb = MultiLabelBinarizer()
X_structured = mlb.fit_transform(symptom_lists)  # shape (n_samples, n_unique_symptoms_present)

# -------------------- 8. Combine features --------------------
# Normalize embedding and graph features separately before concatenation
scaler_embed = StandardScaler()
X_att_embeddings_scaled = scaler_embed.fit_transform(X_att_embeddings)

scaler_graph = StandardScaler()
X_graph_feats_scaled = scaler_graph.fit_transform(X_graph_feats)

# combine: [attention-embed | graph-emb | structured one-hot]
X_combined = np.hstack([X_att_embeddings_scaled, X_graph_feats_scaled, X_structured])
print("Combined feature shape:", X_combined.shape)

# -------------------- 9. Labels + Encode --------------------
le = LabelEncoder()
y = le.fit_transform(df[label_col].astype(str).values)
print("Number of classes:", len(le.classes_))

# -------------------- 10. Train-test split --------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_combined, y, np.arange(len(y)), test_size=0.2, random_state=random_state, stratify=y
)

# -------------------- 11. Deep Neural Network Classifier --------------------
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Input dimensions for each branch
att_dim = X_att_embeddings_scaled.shape[1]
graph_dim = X_graph_feats_scaled.shape[1]
struct_dim = X_structured.shape[1]

# Define each input branch
att_input = Input(shape=(att_dim,), name='AttentionInput')
x1 = Dense(128, activation='relu')(att_input)
x1 = Dropout(0.3)(x1)
x1 = BatchNormalization()(x1)

graph_input = Input(shape=(graph_dim,), name='GraphInput')
x2 = Dense(64, activation='relu')(graph_input)
x2 = Dropout(0.3)(x2)
x2 = BatchNormalization()(x2)

struct_input = Input(shape=(struct_dim,), name='StructuredInput')
x3 = Dense(128, activation='relu')(struct_input)
x3 = Dropout(0.3)(x3)
x3 = BatchNormalization()(x3)

# Concatenate all branches
merged = Concatenate()([x1, x2, x3])
merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.4)(merged)
merged = BatchNormalization()(merged)

# Output layer
output = Dense(num_classes, activation='softmax')(merged)

# Build model
model = Model(inputs=[att_input, graph_input, struct_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# -------------------- 12. Train --------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    [X_att_embeddings_scaled[idx_train], X_graph_feats_scaled[idx_train], X_structured[idx_train]],
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -------------------- 13. Evaluate --------------------
y_pred_probs = model.predict([X_att_embeddings_scaled[idx_test], X_graph_feats_scaled[idx_test], X_structured[idx_test]])
y_pred = np.argmax(y_pred_probs, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))




# %%
# -------------------- 13. Evaluate Model --------------------
# Slice test features
X_att_test = X_att_embeddings_scaled[idx_test]
X_graph_test = X_graph_feats_scaled[idx_test]
X_struct_test = X_structured[idx_test]

# Evaluate on test set
test_loss, test_acc = model.evaluate(
    [X_att_test, X_graph_test, X_struct_test],
    y_test_cat,
    verbose=1
)
print("\nTest Accuracy: {:.2f}%".format(test_acc * 100))

# Predict class probabilities
y_pred_probs = model.predict([X_att_test, X_graph_test, X_struct_test])
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
from sklearn.metrics import classification_report
print("\nClassification Report:\n")
print(classification_report(np.argmax(y_test_cat, axis=1), y_pred, target_names=le.classes_))


# %%
# -------------------- 14. Manual Sample Predictions --------------------
def predict_disease(symptom_list):
    """Predict disease for a new list of symptoms."""
    # attention embedding
    att_vec = attention_severity_embedding(symptom_list)
    att_vec_scaled = scaler_embed.transform(att_vec.reshape(1, -1))

    # graph embedding
    graph_vec = patient_graph_features(symptom_list)
    graph_vec_scaled = scaler_graph.transform(graph_vec.reshape(1, -1))

    # structured one-hot vector
    struct_vec = mlb.transform([symptom_list])

    # predict
    pred_prob = model.predict([att_vec_scaled, graph_vec_scaled, struct_vec])
    pred_label = le.inverse_transform([np.argmax(pred_prob)])
    print(f"\nSymptoms: {symptom_list}")
    print(f"Predicted Disease: {pred_label[0]}")
    print(f"Confidence: {np.max(pred_prob)*100:.2f}%")

# Examples
predict_disease(["fever", "cough", "fatigue"])
predict_disease(["joint pain", "rash", "nausea"])
predict_disease(["chest pain", "shortness of breath", "dizziness"])


# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# -------------------- Predict classes --------------------
y_pred_probs = model.predict([X_test[:, :X_att_embeddings_scaled.shape[1]],     # attention embeddings
                              X_test[:, X_att_embeddings_scaled.shape[1]:
                                      X_att_embeddings_scaled.shape[1]+X_graph_feats_scaled.shape[1]],  # graph features
                              X_test[:, X_att_embeddings_scaled.shape[1]+X_graph_feats_scaled.shape[1]:]  # structured features
                             ])
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# -------------------- Confusion Matrix --------------------
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

plt.figure(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix for Disease Prediction")
plt.show()


# %%
# -------------------- 13. t-SNE / UMAP Visualization of Patient Embeddings --------------------
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import numpy as np

# Use only the embedding features (scaled) for visualization
X_vis = np.hstack([X_att_embeddings_scaled, X_graph_feats_scaled])

# t-SNE
tsne = TSNE(n_components=2, random_state=random_state)
X_tsne = tsne.fit_transform(X_vis)

plt.figure(figsize=(10,7))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df[label_col], palette="tab10")
plt.title("t-SNE of Patient Embeddings by Disease")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.show()




# %%
import shap
import numpy as np

# Concatenate features for SHAP
X_combined_train = np.hstack([
    X_att_embeddings_scaled[idx_train],
    X_graph_feats_scaled[idx_train],
    X_structured[idx_train]
])

X_combined_test = np.hstack([
    X_att_embeddings_scaled[idx_test],
    X_graph_feats_scaled[idx_test],
    X_structured[idx_test]
])

# Use a small background sample
background = X_combined_train[:100]

# Pick a test sample
sample_idx = 0
test_sample = X_combined_test[sample_idx:sample_idx+1]

# Prediction function (accepts concatenated input)
def model_predict(X):
    # Split input back into branches for your model
    att_dim = X_att_embeddings_scaled.shape[1]
    graph_dim = X_graph_feats_scaled.shape[1]
    struct_dim = X_structured.shape[1]

    X_att = X[:, :att_dim]
    X_graph = X[:, att_dim:att_dim+graph_dim]
    X_struct = X[:, att_dim+graph_dim:]
    return model.predict([X_att, X_graph, X_struct])

# Create SHAP KernelExplainer
explainer = shap.KernelExplainer(model_predict, background)

# Compute SHAP values
shap_values = explainer.shap_values(test_sample)

# Visualize
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=None)


# %%
import seaborn as sns

# Build attention matrix for all patients (n_samples x max_symptoms)
max_symptoms = max(symptom_lists.apply(len))
att_matrix = np.zeros((len(symptom_lists), max_symptoms))

for i, lst in enumerate(symptom_lists):
    if len(lst) == 0:
        continue
    vecs = np.vstack([embed_symptom(s) for s in lst])
    context = vecs.mean(axis=0)
    scores = vecs.dot(context)
    exp_scores = np.exp(scores - np.max(scores))
    attn = exp_scores / exp_scores.sum()
    sev = get_symptom_severity_array(lst)
    sev_norm = 0.5 + (sev - 1)/4
    combined = attn * sev_norm
    combined /= combined.sum()
    att_matrix[i, :len(lst)] = combined

plt.figure(figsize=(12,8))
sns.heatmap(att_matrix, cmap="YlGnBu", cbar_kws={'label':'Attention×Severity'})
plt.xlabel("Symptom position (patient-specific)")
plt.ylabel("Patients")
plt.title("Attention × Severity Heatmap across patients")
plt.show()


# %%
# Select 2-3 diseases by name
selected_diseases = ['AIDS', 'Arthritis', 'Chicken pox']  # replace with actual names in your dataset

top_k = 5
top_symptoms_selected = {}

for disease in selected_diseases:
    # Get label index
    if disease not in disease_names:
        print(f"Disease {disease} not found in dataset.")
        continue
    class_idx = np.where(disease_names == disease)[0][0]
    
    # Find all patients with this disease
    patient_idxs = np.where(y == class_idx)[0]
    
    # Initialize symptom scores
    symptom_scores = np.zeros(len(unique_symptoms))
    
    for idx in patient_idxs:
        patient_symptoms = symptom_lists.iloc[idx]
        if len(patient_symptoms) == 0:
            continue
        
        # Compute attention × severity
        vecs = np.vstack([embed_symptom(s) for s in patient_symptoms])
        context = vecs.mean(axis=0)
        scores = vecs.dot(context)
        exp_scores = np.exp(scores - np.max(scores))
        attn = exp_scores / (exp_scores.sum() + 1e-12)
        sev = get_symptom_severity_array(patient_symptoms)
        sev_norm = 0.5 + (sev - 1)/4.0
        combined = attn * sev_norm
        combined = combined / (combined.sum() + 1e-12)
        
        # Add to global scores
        for s, c in zip(patient_symptoms, combined):
            if s.lower() in symptom_index:
                symptom_scores[symptom_index[s.lower()]] += c
    
    # Get top-k
    top_idx = np.argsort(symptom_scores)[::-1][:top_k]
    top_symptoms_selected[disease] = [(unique_symptoms[i], symptom_scores[i]) for i in top_idx]


# %%
import matplotlib.pyplot as plt

for disease, top_symptoms in top_symptoms_selected.items():
    symptoms, scores = zip(*top_symptoms)
    plt.figure(figsize=(8,4))
    plt.barh(symptoms[::-1], scores[::-1], color='skyblue')
    plt.xlabel("Cumulative Attention × Severity")
    plt.title(f"Top-{top_k} Predictive Symptoms for {disease}")
    plt.show()



