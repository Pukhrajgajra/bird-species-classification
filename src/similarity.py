import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import json

def extract_embeddings(model, data_gen, n_batches=None):
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output
    )
    embeddings = []
    labels = []
    total = n_batches or len(data_gen)

    print(f"Extracting embeddings from {total} batches...")
    for i, (batch_images, batch_labels) in enumerate(data_gen):
        if i >= total:
            break
        batch_embeddings = embedding_model.predict(batch_images, verbose=0)
        embeddings.extend(batch_embeddings)
        labels.extend(np.argmax(batch_labels, axis=1))
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{total} batches")

    return np.array(embeddings), np.array(labels)

def find_similar_species(embeddings, labels, class_names, query_class_idx, top_n=5):
    query_mask = labels == query_class_idx
    query_embeddings = embeddings[query_mask]

    if len(query_embeddings) == 0:
        return []

    query_mean = np.mean(query_embeddings, axis=0, keepdims=True)
    class_means = []
    unique_classes = np.unique(labels)

    for cls_idx in unique_classes:
        cls_mask = labels == cls_idx
        cls_mean = np.mean(embeddings[cls_mask], axis=0)
        class_means.append((cls_idx, cls_mean))

    class_indices = [c[0] for c in class_means]
    class_vectors  = np.array([c[1] for c in class_means])
    similarities   = cosine_similarity(query_mean, class_vectors)[0]

    sorted_idx = np.argsort(similarities)[::-1]
    results = []
    for idx in sorted_idx[1:top_n+1]:
        cls_idx  = class_indices[idx]
        sim_score = similarities[idx]
        cls_name  = class_names[cls_idx].split(".")[-1] if cls_idx < len(class_names) else str(cls_idx)
        results.append({"class_idx": int(cls_idx), "class_name": cls_name, "similarity": float(sim_score)})

    return results

def plot_tsne(embeddings, labels, class_names, n_samples=3000, save_path="results/plots/tsne.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[idx]
        labels_sample     = labels[idx]
    else:
        embeddings_sample = embeddings
        labels_sample     = labels

    print(f"Running t-SNE on {len(embeddings_sample)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_sample)

    plt.figure(figsize=(20, 16))
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels_sample))))

    for i, cls_idx in enumerate(np.unique(labels_sample)):
        mask = labels_sample == cls_idx
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]], s=8, alpha=0.6,
            label=class_names[cls_idx].split(".")[-1] if cls_idx < len(class_names) else str(cls_idx)
        )

    plt.title("t-SNE Visualization of Bird Species Embeddings\n(Each color = one species)", fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

def build_similarity_report(embeddings, labels, class_names, save_path="results/metrics/similarity_report.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    report = {}
    unique_classes = np.unique(labels)

    print(f"Building similarity report for {len(unique_classes)} classes...")
    for cls_idx in unique_classes[:20]:
        cls_name = class_names[cls_idx].split(".")[-1]
        similar  = find_similar_species(embeddings, labels, class_names, cls_idx, top_n=3)
        report[cls_name] = similar

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Similarity report saved to {save_path}")
    print("\nSample findings:")
    for species, similars in list(report.items())[:3]:
        print(f"  {species} is most similar to:")
        for s in similars:
            print(f"    - {s['class_name']} (similarity: {s['similarity']:.3f})")
    return report