import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

def train():
    model_path = kagglehub.model_download('google/gemma-2/Transformers/gemma-2-9b-it/2')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
        # quantization_config=quantization_config,
    )

    dataset = load_dataset("icemoon28/drawing_llm")
    train_data = dataset["train"].train_test_split(test_size=0.1)
    train_dataset, eval_dataset = train_data["train"], train_data["test"]

    def formatting_func(example):
        return """Generate SVG code to visually represent the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A red circle with a blue square inside"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{description}"</description>
```svg
{svg_code}
""".format(description=example["description"], svg_code=example["svg"])

    # LoRA 配置（适配大模型）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    def compute_metrics(eval_preds):
        return None


    # 训练参数
    training_args = TrainingArguments(
        output_dir="./gemma-sft-output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False  # 多卡时推荐加上
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=2048,
        packing=False,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()