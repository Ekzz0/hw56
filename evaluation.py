import os
import argparse
import numpy as np
from datasets import load_dataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import ttest_rel, ttest_1samp
from peft import get_peft_model, LoraConfig
from tqdm import tqdm  # Импортируем tqdm
import torch

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_metrics():
    """Define and return evaluation metrics."""
    style_metric = GEval(
        name="Style Matching",
        criteria="Оцени, насколько последняя фраза (actual output) делает анекдот смешным, логичным и остроумным:",
        evaluation_params=[
            LLMTestCaseParams.INPUT, 
            LLMTestCaseParams.ACTUAL_OUTPUT, 
            LLMTestCaseParams.EXPECTED_OUTPUT
        ]
    )

    memorability_metric = GEval(
        name="Memorability",
        criteria="Оцени, насколько последняя фраза (actual output) делает анекдот запоминающимся. Учитывай уникальность, завершённость фразы и отсутствие клише:",
        evaluation_params=[
            LLMTestCaseParams.INPUT, 
            LLMTestCaseParams.ACTUAL_OUTPUT, 
            LLMTestCaseParams.EXPECTED_OUTPUT
        ]
    )

    return style_metric, memorability_metric

def prepare_openai_jokes_dataset(args):
    """Load and prepare dataset for evaluation."""
    dataset = load_dataset("inkoziev/jokes_dialogues")
    data_frame = dataset['train'].to_pandas()

    unique_ids = data_frame['src_hash'].unique()
    sampled_ids = np.random.choice(unique_ids, args.num_samples, replace=False)

    openai_chat_examples = []
    for chat_id in tqdm(sampled_ids, desc="Preparing dataset"):  # Добавляем прогресс-бар для выборки
        chat_data = data_frame[data_frame['src_hash'] == chat_id]
        messages = [
            {"role": "system", "content": "Ты чат бот, который генерирует анекдоты. Продолжи диалог одной репликой так, чтобы получился анекдот."}
        ]

        for _, row in chat_data.iterrows():
            role = "assistant" if row['reply_num'] % 2 == 0 else "user"
            messages.append({"role": role, "content": row['utterance']})

        openai_chat_examples.append({"messages": messages})

    return openai_chat_examples

def get_model_predictions(gt_examples, tokenizer, peft_model, system_message):
    """Generate predictions using the specified model."""
    predictions = []

    for example in tqdm(gt_examples, desc="Generating predictions"):  # Добавляем прогресс-бар для предсказаний
        input_text = system_message['content'] + "\n" + "\n".join(
            msg['content'] for msg in example['messages'] if msg['role'] != 'system'
        )

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=200)
        outputs = peft_model.generate(inputs['input_ids'].to(peft_model.device), max_new_tokens=77, do_sample=True, top_p=0.9, temperature=1.0,  pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predicted_example = example.copy()
        predicted_example['messages'].append({"role": "assistant", "content": generated_text})
        predictions.append(predicted_example)

    return predictions

def evaluate_metrics(gt_examples, predicted_examples1, predicted_examples2, style_metric, memorability_metric):
    """Evaluate predictions using defined metrics."""
    scores1, scores2 = [], []
    scores1_memorability, scores2_memorability = [], []

    for gt, pred1, pred2 in tqdm(zip(gt_examples, predicted_examples1, predicted_examples2), desc="Evaluating metrics"):  # Добавляем прогресс-бар для вычислений метрик
        test_case1 = LLMTestCase(
            input="\n".join(msg['content'] for msg in pred1['messages'][:-1]),
            actual_output=pred1['messages'][-1]['content'],
            expected_output=gt['messages'][-1]['content']
        )
        test_case2 = LLMTestCase(
            input="\n".join(msg['content'] for msg in pred2['messages'][:-1]),
            actual_output=pred2['messages'][-1]['content'],
            expected_output=gt['messages'][-1]['content']
        )

        # Style metric evaluation
        style_metric.measure(test_case1)
        scores1.append(style_metric.score)
        style_metric.measure(test_case2)
        scores2.append(style_metric.score)

        # Memorability metric evaluation
        memorability_metric.measure(test_case1)
        scores1_memorability.append(memorability_metric.score)
        memorability_metric.measure(test_case2)
        scores2_memorability.append(memorability_metric.score)

    return scores1, scores2, scores1_memorability, scores2_memorability

def main(args):
    # Load evaluation metrics
    style_metric, memorability_metric = load_metrics()

    # Prepare dataset
    gt_examples = prepare_openai_jokes_dataset(args)

    # Load LoRA config and apply it to the model
    lora_config = LoraConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    peft_model = get_peft_model(model, lora_config).to("cuda")

    peft_model.eval()

    # Generate predictions
    system_message1 = {"role": "system", "content": "Ты чат бот, который генерирует анекдоты. Продолжи диалог одной репликой так, чтобы получился анекдот."}
    system_message2 = {"role": "system", "content": "Ты чат бот, который генерирует диалоги. Продолжи диалог."}

    predicted_examples1 = get_model_predictions(gt_examples, tokenizer, peft_model, system_message1)
    predicted_examples2 = get_model_predictions(gt_examples, tokenizer, peft_model, system_message2)

    # Evaluate predictions
    scores1, scores2, scores1_memorability, scores2_memorability = evaluate_metrics(
        gt_examples, predicted_examples1, predicted_examples2, style_metric, memorability_metric
    )

    print("=============== Scores ======================")
    ttest_res1 = ttest_1samp(scores1, 0.5)
    print("SCORE1:", np.mean(scores1), scores1)
    print("TTEST1:", ttest_res1)
    ttest_res2 = ttest_1samp(scores2, 0.5)
    print("SCORE2:", np.mean(scores2), scores2)
    print("TTEST2:", ttest_res2)


    # Новая метрика запоминаемости
    print("SCORE1 Memorability:", np.mean(scores1_memorability), scores1_memorability)
    print("TTEST1:", ttest_1samp(scores1_memorability, 0.5))
    print("SCORE2 Memorability:", np.mean(scores2_memorability), scores2_memorability)
    print("TTEST2:", ttest_1samp(scores2_memorability, 0.5))

    rel_ttest_res = ttest_rel(scores1, scores2)

    print("=============== Rel Scores ======================")
    print("TTEST:", rel_ttest_res)
    print("TTEST Memorability:", ttest_rel(scores1_memorability, scores2_memorability))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=30, help="Number of samples to evaluate")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model")
    args = parser.parse_args()

    main(args)
