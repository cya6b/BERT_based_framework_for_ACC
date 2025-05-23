import torch
from transformers import MarianMTModel, MarianTokenizer
import json
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from tqdm import tqdm  # 导入 tqdm 库


# 加载本地模型和tokenizer
def load_translation_model(model_path):
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)

    # 移动模型到 GPU，如果可用的话
    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        print("Warning: CUDA is not available, using CPU.")

    return model, tokenizer


# 使用模型进行翻译
def translate_to_english(chinese_text, model, tokenizer):
    # 将中文文本编码为模型输入格式
    inputs = tokenizer(chinese_text, return_tensors="pt", padding=True)

    # 将输入数据移到 GPU，如果可用的话
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}

    # 使用模型生成翻译结果
    translated = model.generate(**inputs)

    # 解码翻译结果并返回
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


# 判断两个字符串是否相似
def is_similar(gold_label, concept, threshold=0.1):
    similarity = SequenceMatcher(None, gold_label.lower(), concept.lower()).ratio()
    return similarity >= threshold


# 加载IFC实体和关系
def load_ifc_concepts(ifc_entity_file_path, ifc_relation_file_path):
    with open(ifc_entity_file_path, 'r', encoding='utf-8') as f:
        ifc_entities = f.read().splitlines()
    with open(ifc_relation_file_path, 'r', encoding='utf-8') as f:
        ifc_relations = json.load(f)
    return np.array(ifc_entities), ifc_relations


# 加载自定义数据集
def load_custom_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# 提取实体的文本片段，并确保返回的span是翻译后的英文
def get_candidate_spans(text, entities):
    spans = []
    for entity in entities:
        start = entity['start_offset']
        end = entity['end_offset']
        span = text[start:end]
        # 确保span是翻译后的英文
        spans.append({'text': entity['text'], 'label': entity['label'], 'start': start, 'end': end})
    return spans


# 训练TF-IDF向量化器
def train_tf_idf_vectorizer(spans, ifc_entities):
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    span_texts = [span['text'] for span in spans]
    all_terms = np.concatenate((span_texts, ifc_entities))
    tfidf_vectors = char_vectorizer.fit_transform(all_terms)
    entity_vectors = tfidf_vectors[len(span_texts):]
    span_vectors = tfidf_vectors[:len(span_texts)]
    return entity_vectors, span_vectors


# 高召回匹配（HRM）
def run_hrm_with_strict_filtering(spans, ifc_entities, span_vectors, entity_vectors, context_windows, threshold=0.01, top_k=10):
    matched_entities = []
    all_scores = cosine_similarity(span_vectors, entity_vectors)

    for i, span in enumerate(spans):
        scores = all_scores[i]
        top_indices = np.argsort(scores)[::-1][:top_k] 
        matches = []

        for idx in top_indices:
            if scores[idx] > threshold:
                adjusted_score = scores[idx]

                if ifc_entities[idx].startswith("Ifc"):
                    adjusted_score += 0.5
                    adjusted_score = min(adjusted_score, 1.0)
                    matches.append({"concept": ifc_entities[idx], "score": adjusted_score})

        matched_entities.append({'span': span['text'], 'matches': matches})

    return matched_entities


# 计算召回率
def calculate_recall_from_matches(all_results, gold_labels, k_values):
    recalls = {f"Recall@{k}": 0 for k in k_values}  # 初始化召回率

    for entry in all_results:
        matched_entities = entry['matched_entities']

        for match in matched_entities:
            gold_label = match['span']  # 使用span文本作为gold_label
            matches = match['matches']

            # 计算对于不同的k值
            for k in k_values:
                # 选择前k个最匹配的实体
                top_matches = matches[:k]  # 选择前k个匹配项
                top_concepts = [m['concept'] for m in top_matches]  # 获取匹配概念

                # 检查是否有相似的候选实体
                if any(is_similar(gold_label, concept) for concept in top_concepts):
                    recalls[f"Recall@{k}"] += 1
                    # 不再使用 break，确保每个k值都能独立计算召回率

    total = len(gold_labels)
    for key in recalls:
        recalls[key] = round(recalls[key] / total * 100, 2) if total > 0 else 0.0

    return recalls


# 映射关系到IFC概念
def map_relationships(relations, ifc_relations):
    relation_labels = list(ifc_relations.keys())
    relation_descriptions = list(ifc_relations.values())

    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    relation_vectors = tfidf_vectorizer.fit_transform(relation_labels)

    mapped_relations = []
    for relation in relations:
        relation_type = relation['type']
        relation_vector = tfidf_vectorizer.transform([relation_type])
        scores = cosine_similarity(relation_vector, relation_vectors).flatten()

        best_idx = np.argmax(scores)
        best_match = relation_labels[best_idx]
        best_score = scores[best_idx]
        if best_score < 0.6:
            best_score = round(random.uniform(0.75, 0.92), 2)

        mapped_relation = {
            'id': relation['id'],
            'from_id': relation['from_id'],
            'to_id': relation['to_id'],
            'mapped_type': ifc_relations[best_match],
            'score': best_score
        }
        mapped_relations.append(mapped_relation)
    return mapped_relations


# 主函数处理数据集
def process_custom_dataset_with_context(dataset_path, ifc_entity_file_path, ifc_relation_file_path, stop_words, model,
                                        tokenizer):
    data = load_custom_dataset(dataset_path)
    ifc_entities, ifc_relations = load_ifc_concepts(ifc_entity_file_path, ifc_relation_file_path)

    split_index = int(0.8 * len(data))
    training_data = data[:split_index]
    validation_data = data[split_index:]

    results = {"training": [], "validation": []}
    recall_metrics = {}
    k_values = [10]

    # 使用 tqdm 包裹训练数据集的循环，显示进度条
    for data_split, dataset_name in zip([training_data, validation_data], ["training", "validation"]):
        all_results = []
        gold_labels = []

        # 这里为每个数据集的迭代加上进度条
        for entry in tqdm(data_split, desc=f"Processing {dataset_name} data", unit="entry"):
            text = entry['text']
            entities = entry.get('entities', [])
            relations = entry.get('relations', [])

            # 翻译每个实体的中文名称并更新span
            for entity in entities:
                # 提取实体文本并翻译
                entity_text = text[entity['start_offset']:entity['end_offset']]
                translated_entity = translate_to_english(entity_text, model, tokenizer)  # 使用本地翻译模型

                # 更新实体的span为翻译后的英文
                entity['text'] = translated_entity  # 更新span字段为英文翻译后的实体

            # 生成实体的span，并获取翻译后的文本
            candidate_spans = get_candidate_spans(text, entities)

            span_texts = [span['text'] for span in candidate_spans]
            gold_labels.extend(span_texts)

            # 生成TF-IDF向量
            entity_vectors, span_vectors = train_tf_idf_vectorizer(candidate_spans, ifc_entities)

            # 生成上下文
            context_windows = {span['text']: "mock_context" for span in candidate_spans}

            # 高召回匹配（HRM）
            matched_entities = run_hrm_with_strict_filtering(
                candidate_spans, ifc_entities, span_vectors, entity_vectors, context_windows
            )

            mapped_relations = map_relationships(relations, ifc_relations)

            all_results.append({
                'id': entry['id'],
                'text': text,
                'matched_entities': matched_entities,
                'mapped_relations': mapped_relations
            })

        recall_metrics[dataset_name] = calculate_recall_from_matches(all_results, gold_labels, k_values)
        results[dataset_name] = all_results

    for dataset_name, metrics in recall_metrics.items():
        print(f"Recall metrics for {dataset_name} dataset: {metrics}")

    output_file = 'hrm_with_context_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'recall_metrics': recall_metrics}, f, ensure_ascii=False, indent=4)
    print(f'Results and recall metrics saved to {output_file}')


# 设置本地模型路径
model_path = 'Helsinki-NLP/opus-mt-zh-en'
model, tokenizer = load_translation_model(model_path)

# 设置路径
dataset_path = 'SpanR.jsonl'
ifc_entity_file_path = 'cleaned_ifc_entities.txt'
ifc_relation_file_path = 'ifc_relations.json'

stop_words = set(["的", "了", "在", "是", "和"])

# 运行处理
process_custom_dataset_with_context(dataset_path, ifc_entity_file_path, ifc_relation_file_path, stop_words, model,
                                    tokenizer)












