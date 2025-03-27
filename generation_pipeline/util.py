
from collections import defaultdict
import csv
import json
import logging
import os
import re
import pandas as pd
import torch
from lxml import etree
import re2

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from generation_pipeline.setting import Deepseek_Api_Key, OpenAI_Api_Key
from openai import OpenAI, OpenAIError
import kagglehub

svg_constraints = kagglehub.package_import('metric/svg-constraints')

default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""

def prompt_with_deepseek(description):
    client = OpenAI(api_key=Deepseek_Api_Key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=description,
            stream=False,
            timeout=10  # 设置超时，避免卡住
        )

        print("🤖 Response: ", response)

        return response.choices[0].message.content

    except OpenAIError as e:
        print("❌ 请求出错：", e)
        return ""


def prompt_with_openai1(description):
    client = OpenAI(api_key=OpenAI_Api_Key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": description},
        # ],
        messages=description,
        stream=False
    )

    return response.choices[0].message.content

def extract_answers(text):
    pattern = r'<answer>(.*?)</answer>'
    answers = re.findall(pattern, text, re.DOTALL)
    return [answer.strip() for answer in answers]

def extract_svg(output_decoded):
    matches = re.findall(r"<svg.*?</svg>", output_decoded, re.DOTALL | re.IGNORECASE)
    if matches and (len(matches) == 1):
        svg = matches[-1]

        return svg
    else:
        return None

if __name__ == "__main__":
    response_text = """
    <answer>a turquoise river winding through an amber canyon</answer>
    <answer>navy trousers with geometric gold embroidery</answer>
    <answer>a coral reef beneath shimmering sunlight</answer>
    """

    descriptions = extract_answers(response_text)
    print(descriptions)

def read_sentence_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines_array = [line.strip() for line in file if line.strip() if line.strip != '']

        return lines_array
    
def cluster_sentence(
    sentences,
    model,
    distance_threshold
):
    embeddings = model.encode(sentences)

    # 层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )

    labels = clustering.fit_predict(embeddings)
    sentence_to_embedding = {}

    # 2️⃣ 先用 defaultdict(set) 组织 sentence 到 cluster
    clusters = defaultdict(set)

    for sentence, label, embedding in zip(sentences, labels, embeddings):
        clusters[label.item()].add(sentence)  # 只存储 sentence
        sentence_to_embedding[sentence] = embedding  # 存储 sentence -> embedding 映射

    clusters = {label: [(sentence, sentence_to_embedding[sentence]) for sentence in list(sentences)] for label, sentences in clusters.items()}

    return clusters
    
def print_clusters(clusters):

    print("Cluster Numbers: ", len(clusters))
    print("Cluster Sentence Numbers: ", sum([len(cluster) for cluster in clusters.values()]))

    for label, cluster_sentences in clusters.items():
        print(f"\nCluster {label}:")
        for sentence in cluster_sentences:
            print(f" - {sentence[0]}")


def print_clusters_cosine_similarity(clusters):

    print("Cluster Numbers: ", len(clusters))
    print("Cluster Sentence Numbers: ", sum([len(cluster) for cluster in clusters.values()]))

    for label, cluster_items in clusters.items():
        print(f"\nCluster {label} (共{len(cluster_items)}句):")
        cluster_sentences = [item[0] for item in cluster_items]
        cluster_embeddings = np.array([item[1] for item in cluster_items])

        # 计算句子之间的cosine相似度矩阵
        sim_matrix = cosine_similarity(cluster_embeddings)

        for i in range(len(cluster_sentences)):
            for j in range(i + 1, len(cluster_sentences)):
                sim = sim_matrix[i][j]
                print(f" - 「{cluster_sentences[i]}」 ↔ 「{cluster_sentences[j]}」: similarity = {sim:.4f}")

def transform_cluster_into_object(clusters):
    abject = []
    for label, cluster_items in clusters.items():
        abject.append({
            "label": label,
            "sentences": [item[0] for item in cluster_items]
        })
    return abject

def save_clusters_to_json(clusters, filename="clusters.json"):
    obj = transform_cluster_into_object(clusters)
    
    # 写入 JSON 文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

    print(f"✅ Clusters saved to {filename}")

def save_to_json(obj, filename="clusters.json"):
    # **确保目录存在**
    if os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # **写入 JSON 文件**
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

    print(f"✅ Clusters saved to {filename}")

def save_to_svg(svg_string, filename="clusters.svg"):
    if os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg_string)

    print(f"✅ Clusters saved to {filename}")

def read_svg_as_string(filename):
    """
    读取 SVG 文件内容为字符串
    :param filename: SVG 文件路径
    :return: SVG 文件的字符串内容
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ 文件不存在: {filename}")

    with open(filename, 'r', encoding='utf-8') as f:
        svg_string = f.read()

    return svg_string

def read_from_json(filename="clusters.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f) 
    
def transform_json_into_array(jsons, transform_func):
    items = []
    for json in jsons:
        array = transform_func(json)

        for item in array:
            items.append(item)
        
    return items
        

    
def enforce_constraints(svg_string: str) -> str:
    """Enforces constraints on an SVG string, removing disallowed elements
    and attributes.

    Parameters
    ----------
    svg_string : str
        The SVG string to process.

    Returns
    -------
    str
        The processed SVG string, or the default SVG if constraints
        cannot be satisfied.
    """
    logging.info('Sanitizing SVG...')

    constraints = svg_constraints.SVGConstraints()

    try:
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        root = etree.fromstring(svg_string, parser=parser)
    except etree.ParseError as e:
        logging.error('SVG Parse Error: %s. Returning default SVG.', e)
        return None

    elements_to_remove = []
    for element in root.iter():
        tag_name = etree.QName(element.tag).localname

        # Remove disallowed elements
        if tag_name not in constraints.allowed_elements:
            elements_to_remove.append(element)
            continue  # Skip attribute checks for removed elements

        # Remove disallowed attributes
        attrs_to_remove = []
        for attr in element.attrib:
            attr_name = etree.QName(attr).localname
            if (
                attr_name
                not in constraints.allowed_elements[tag_name]
                and attr_name
                not in constraints.allowed_elements['common']
            ):
                attrs_to_remove.append(attr)

        for attr in attrs_to_remove:
            logging.debug(
                'Attribute "%s" for element "%s" not allowed. Removing.',
                attr,
                tag_name,
            )
            del element.attrib[attr]

        # Check and remove invalid href attributes
        for attr, value in element.attrib.items():
            if etree.QName(attr).localname == 'href' and not value.startswith('#'):
                logging.debug(
                    'Removing invalid href attribute in element "%s".', tag_name
                )
                del element.attrib[attr]

        # Validate path elements to help ensure SVG conversion
        if tag_name == 'path':
            d_attribute = element.get('d')
            if not d_attribute:
                logging.warning('Path element is missing "d" attribute. Removing path.')
                elements_to_remove.append(element)
                continue # Skip further checks for this removed element
            # Use regex to validate 'd' attribute format
            path_regex = re2.compile(
                r'^'  # Start of string
                r'(?:'  # Non-capturing group for each command + numbers block
                r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                r'\s*'  # Optional whitespace after command
                r'(?:'  # Non-capturing group for optional numbers
                r'-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?'  # First number
                r'(?:[\s,]+-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)*'  # Subsequent numbers with mandatory separator(s)
                r')?'  # Numbers are optional (e.g. for Z command)
                r'\s*'  # Optional whitespace after numbers/command block
                r')+'  # One or more command blocks
                r'\s*'  # Optional trailing whitespace
                r'$'  # End of string
            )
            if not path_regex.match(d_attribute):
                logging.warning(
                    'Path element has malformed "d" attribute format. Removing path.'
                )
                elements_to_remove.append(element)
                continue
            logging.debug('Path element "d" attribute validated (regex check).')
    
    # Remove elements marked for removal
    for element in elements_to_remove:
        if element.getparent() is not None:
            element.getparent().remove(element)
            logging.debug('Removed element: %s', element.tag)

    try:
        cleaned_svg_string = etree.tostring(root, encoding='unicode')
        return cleaned_svg_string
    except ValueError as e:
        logging.error(
            'SVG could not be sanitized to meet constraints: %s', e
        )
        return None
    


def write_to_csv(data, filename="clusters.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'sentences'])
        for item in data:
            writer.writerow([item["label"], item["sentences"]])


def transform_svg_to_png(input_svg, output_png):
    import cairosvg

    try:
        cairosvg.svg2png(url=input_svg, write_to=output_png)
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")


def read_from_csv(filename="clusters.csv"):
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def preprocess_svg_csv(svg_csv):

    pass

def preprocess_description_csv(description_csv):

    pass

def sort_csv_by_id(filename, key):
    df = pd.read_csv(filename)
    df_sorted = df.sort_values(by=key)  # 按照 id 升序排列
    df_sorted.to_csv(filename, index=False, encoding='utf-8')


def merge_and_sort_excels(file_list, key, output_file):
    # 读取所有 Excel 文件并合并为一个 DataFrame
    dfs = [pd.read_excel(file) for file in file_list]
    combined_df = pd.concat(dfs, ignore_index=True)

    # 按照 'score' 列排序，降序
    combined_df = combined_df.sort_values(by=key, ascending=False)

    # 重设 id（从 1 开始）
    combined_df = combined_df.reset_index(drop=True)
    combined_df['id'] = range(1, len(combined_df) + 1)

    combined_df.to_excel(output_file, index=False)

    return combined_df