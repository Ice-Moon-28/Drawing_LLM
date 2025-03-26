import csv
import io
from math import prod
from statistics import mean
import statistics
from statistics import mean
from tqdm import tqdm

import cairosvg
import clip
import kagglehub
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)
from generation_pipeline.setting import device

svg_constraints = kagglehub.package_import('metric/svg-constraints')


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, write_to_csv: bool, csv_filename: str, sorted_csv_filename: str
) -> float:
    """Calculates a fidelity score by comparing generated SVG images to target text descriptions.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing target text descriptions. Must have a column named 'description'.
    submission : pd.DataFrame
        A DataFrame containing generated SVG strings. Must have a column named 'svg'.
    row_id_column_name : str
        The name of the column containing row identifiers. This column is removed before scoring.

    Returns
    -------
    float
        The mean fidelity score (a value between 0 and 1) representing the average similarity between the generated SVGs and their descriptions.
        A higher score indicates better fidelity.

    Raises
    ------
    ParticipantVisibleError
        If the 'svg' column in the submission DataFrame is not of string type or if validation of the SVG fails.

    Examples
    --------
    >>> import pandas as pd
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'description': ['red ball', 'swimming pool']
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'svg': ['<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red"/></svg>',
    ...         '<svg viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="blue"/></svg>']
    ... })
    >>> score(solution, submission, 'id')
    0...
    """
    # Validate
    id = 0
    del solution[row_id_column_name], submission[row_id_column_name]
    if not pd.api.types.is_string_dtype(submission.loc[:, 'svg']):
        raise ParticipantVisibleError('svg must be a string.')
    # check that SVG code meets defined constraints
    constraints = svg_constraints.SVGConstraints()
    try:
        for svg in submission.loc[:, 'svg']:
            
            constraints.validate_svg(svg)
    except:
        raise ParticipantVisibleError('SVG code violates constraints.')

    # Score
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    results = []
    svg_res = []

    if write_to_csv:
        f = open(csv_filename, 'w', newline='', encoding='utf-8')
    
        writer = csv.writer(f)
        writer.writerow(['id', 'svg', 'description', 'score'])

        f2 = open(sorted_csv_filename, 'w', newline='', encoding='utf-8')
        writer2 = csv.writer(f2)
        writer2.writerow(['id', 'svg', 'description', 'score'])

        res = []


    try:
        for svg, description in tqdm(
            zip(submission['svg'], solution['description']),
            total=len(submission),
            desc="Scoring",
            unit="item"
        ):
            image = svg_to_png(svg)
            vqa_score = vqa_evaluator.score(image, 'SVG illustration of ' + description)
            aesthetic_score = aesthetic_evaluator.score(image)
            instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=2.0)

            # 输出本次评分结果
            print(f"Description: {description}, Score: {instance_score:.4f}, SVG: {svg}")

            results.append(instance_score)
            svg_res.append({
                "svg": svg,
                "description": description,
                "score": instance_score,
            })

            if write_to_csv:
                writer.writerow([id, svg, description, instance_score])
                id += 1

                res.append({
                    "svg": svg,
                    "description": description,
                    "score": instance_score,
                })

    except:
        raise ParticipantVisibleError('SVG failed to score.')


    # 假设 results 是你已有的得分列表
    # 例如：
    # results = [0.85, 0.90, 0.75, 0.80, 0.95]

    max_value = max(results)
    min_value = min(results)
    avg_value = mean(results)
    std_value = statistics.stdev(results)  # 计算样本标准差

    res = sorted(res, key=lambda x: x['score'], reverse=True)

    if write_to_csv:
        for item in res:
            writer2.writerow([item['svg'], item['description'], item['score']])

    print(f"Max Fidelity: {max_value:.4f}")
    print(f"Min Fidelity: {min_value:.4f}")
    print(f"Mean Fidelity: {avg_value:.4f}")
    print(f"Std Fidelity: {std_value:.4f}")

    return svg_res, results, float(avg_value)


class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description."""

    def __init__(self):
        # self.quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        # )
        self.model_path = kagglehub.model_download(
            'google/paligemma-2/transformers/paligemma2-10b-mix-448'
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map=device,
            # quantization_config=self.quantization_config,
        )
        self.questions = {
            'fidelity': 'Does <image> portray "{}" without any lettering? Answer yes or no.',
            'text': '<image> Text present: yes or no?',
        }

    def score(self, image: Image.Image, description: str) -> float:
        """Evaluates the fidelity of an image to a target description using VQA yes/no probabilities.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to evaluate.
        description : str
            The text description that the image should represent.

        Returns
        -------
        float
            The score (a value between 0 and 1) representing the match between the image and its description.
        """
        p_fidelity = self.get_yes_probability(image, self.questions['fidelity'].format(description))
        p_text = self.get_yes_probability(image, self.questions['text'])
        return p_fidelity * (1 - p_text)

    def mask_yes_no(self, logits):
        """Masks logits for 'yes' or 'no'."""
        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        mask = torch.full_like(logits, float('-inf'))
        mask[:, yes_token_id] = logits[:, yes_token_id]
        mask[:, no_token_id] = logits[:, no_token_id]
        mask[:, yes_with_space_token_id] = logits[:, yes_with_space_token_id]
        mask[:, no_with_space_token_id] = logits[:, no_with_space_token_id]
        return mask

    def get_yes_probability(self, image, prompt) -> float:
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(
            device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Logits for the last (predicted) token
            masked_logits = self.mask_yes_no(logits)
            probabilities = torch.softmax(masked_logits, dim=-1)

        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        prob_yes = probabilities[0, yes_token_id].item()
        prob_no = probabilities[0, no_token_id].item()
        prob_yes_space = probabilities[0, yes_with_space_token_id].item()
        prob_no_space = probabilities[0, no_with_space_token_id].item()

        total_yes_prob = prob_yes + prob_yes_space
        total_no_prob = prob_no + prob_no_space

        total_prob = total_yes_prob + total_no_prob
        renormalized_yes_prob = total_yes_prob / total_prob

        return renormalized_yes_prob


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        # 使用 kagglehub 下载模型文件，返回文件的本地路径
        self.model_path = 'sac+logos+ava1-l14-linearMSE.pth'
        self.clip_model_path = 'ViT-L/14'
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """加载美学预测器模型和 CLIP 模型。"""
        # 加载美学预测器权重
        state_dict = torch.load(self.model_path, weights_only=True, map_location=device)
        predictor = AestheticPredictor(768)  # CLIP ViT L 14 的 embedding dim 为 768
        predictor.load_state_dict(state_dict)
        predictor.to(device)
        predictor.eval()

        # 加载 CLIP 模型和预处理器
        clip_model, preprocessor = clip.load(self.clip_model_path, device=device)
        return predictor, clip_model, preprocessor


    def score(self, image: Image.Image) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        image = self.preprocessor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

        score = self.predictor(torch.from_numpy(image_features).to(device).float())

        return score.item() / 10.0  # scale to [0, 1]


def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate the harmonic mean of two values, weighted using a beta parameter.

    Args:
        a: First value (e.g., precision)
        b: Second value (e.g., recall)
        beta: Weighting parameter

    Returns:
        Weighted harmonic mean
    """
    # Handle zero values to prevent division by zero
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


if __name__ == "__main__":
    AestheticEvaluator()