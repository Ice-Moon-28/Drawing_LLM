from generation_pipeline.util import print_clusters_cosine_similarity, read_sentence_from_file, cluster_sentence, print_clusters, save_clusters_to_json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

sentences = read_sentence_from_file("descriptions.txt")

# 加载模型并计算句子嵌入

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

clusters = cluster_sentence(
    sentences=sentences,
    model=model,
    distance_threshold=0.02
)

save_clusters_to_json(clusters, "clusters.json")

print_clusters(clusters)

print_clusters_cosine_similarity(clusters)

# 展示聚类结果并计算相似度
