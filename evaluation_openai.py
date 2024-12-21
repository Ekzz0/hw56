import os
import argparse

import numpy as np
from datasets import load_dataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import ttest_rel, ttest_1samp

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

style_matching_metric_geval = GEval(
    name="Style Matching",
    criteria="Определи, насколько последняя фраза (actual output) делает анекдот смешным, логичным и остроумным:",

    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)

# Новая метрика запоминаемости
memorability_metric_geval = GEval(
    name="Memorability",
    criteria="Оцени, насколько последняя фраза (actual output) делает анекдот запоминающимся. "
             "Учитывай уникальность, завершённость фразы и отсутствие клише:",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)

system_message1 = {"role": "system",
                   "content": "Ты чат бот, который генерирует анекдоты. "
                              "Продолжи диалог одной репликой так, чтоб получился анекдот. "
                              "Example input: - Я не могу выйти на работу, потому что жена сломала два ребра. - А при чём тут ты? "
                              "Expected output: - Мои!"}
system_message2 = {"role": "system",
                   "content": "Ты чат бот, который генерирует диалоги. Продолжи диалог"}


def prepare_openai_jokes_dataset(args):
    ds = load_dataset("inkoziev/jokes_dialogues")
    ds_df = ds['train'].to_pandas()

    # print(ds_df)
    unique_chat_ids = ds_df['src_hash'].unique()
    # unique_chat_ids = set(ds['train']['src_hash'])
    sampled_chat_ids = np.random.choice(list(unique_chat_ids), args.num_samples, replace=False)

    openai_chat_examples = []
    for chat_id in sampled_chat_ids:
        reply_nums = ds_df[ds_df['src_hash'] == chat_id]['reply_num'].values
        reply_nums = np.sort(reply_nums)
        messages = [system_message1]
        if reply_nums[0] != 1:
            continue
        message1 = ds_df[(ds_df['src_hash'] == chat_id) & (ds_df['reply_num'] == 1)]['context'].values[0]
        messages.append({"role": "assistant" if 0 % 2 == reply_nums[-1] % 2 else "user",
                         "content": message1})

        for reply_num in reply_nums:
            message = ds_df[(ds_df['src_hash'] == chat_id) & (ds_df['reply_num'] == reply_num)]['utterance'].values[0]
            message_metadata = {"role": "assistant" if reply_num % 2 == reply_nums[-1] % 2 else "user",
                                "content": message}
            if message_metadata['role'] == 'assistant':
                if reply_num == reply_nums[-1]:
                    message_metadata['weight'] = 1
                else:
                    message_metadata['weight'] = 0
            messages.append(message_metadata)

        openai_chat_example = {"messages": messages}
        openai_chat_examples.append(openai_chat_example)
    return openai_chat_examples


def get_model_predictions(gt_examples, system_message, model_id="gpt-4o-mini"):
    predicted_examples = []
    for i_example in range(len(gt_examples)):
        completion = client.chat.completions.create(
            model=model_id,
            messages=[system_message] + gt_examples[i_example]['messages'][1:-1],
        )
        # predicted_examples.append(completion.choices[0].message.content)
        predicted_example = [system_message] + gt_examples[i_example]['messages'][1:-1]
        predicted_example.append({"role": "assistant", "content": completion.choices[0].message.content})
        predicted_example = {"messages": predicted_example}
        predicted_examples.append(predicted_example)

    return predicted_examples


def evaluate_metrics(gt_examples, predicted_examples1, predicted_examples2):
    scores1 = []
    scores2 = []
    scores1_memorability = []
    scores2_memorability = []
    for i_example in range(len(gt_examples)):
        test_case1 = LLMTestCase(
            input="\n".join([msg['content'] for msg in predicted_examples1[i_example]['messages'][1:-1]]),
            actual_output=predicted_examples1[i_example]['messages'][-1]['content'],
            expected_output=gt_examples[i_example]['messages'][-1]['content'],
        )
        test_case2 = LLMTestCase(
            input="\n".join([msg['content'] for msg in predicted_examples2[i_example]['messages'][1:-1]]),
            actual_output=predicted_examples2[i_example]['messages'][-1]['content'],
            expected_output=gt_examples[i_example]['messages'][-1]['content'],
        )

        style_matching_metric_geval.measure(test_case1)
        predicted_examples1[i_example]['fun_score'] = style_matching_metric_geval.score
        scores1.append(style_matching_metric_geval.score)

        style_matching_metric_geval.measure(test_case2)
        predicted_examples2[i_example]['fun_score'] = style_matching_metric_geval.score
        scores2.append(style_matching_metric_geval.score)

        # Новая метрика запоминаемости
        memorability_metric_geval.measure(test_case1)
        predicted_examples1[i_example]['memorability_score'] = memorability_metric_geval.score
        scores1_memorability.append(memorability_metric_geval.score)

        memorability_metric_geval.measure(test_case2)
        predicted_examples2[i_example]['memorability_score'] = memorability_metric_geval.score
        scores2_memorability.append(memorability_metric_geval.score)

    return gt_examples, predicted_examples1, predicted_examples2, scores1, scores2, scores1_memorability, scores2_memorability


def main(args):
    gt_examples = prepare_openai_jokes_dataset(args)
    print("============== GT EXAMPLES ===============")
    for i_example in range(len(gt_examples)):
        print(gt_examples[i_example])
        print("-----------")
    predicted_examples1 = get_model_predictions(gt_examples, system_message1, model_id=args.model_id)
    predicted_examples2 = get_model_predictions(gt_examples, system_message2, model_id=args.model_id)
    print("=============== GT and Predicted EXAMPLES ======================")
    for i_example in range(len(gt_examples)):
        print(gt_examples[i_example])
        print(predicted_examples1[i_example])
        print(predicted_examples2[i_example])
        print("-----------")
    gt_examples, predicted_examples1, predicted_examples2, scores1, scores2, scores1_memorability, scores2_memorability = evaluate_metrics(gt_examples,
                                                                                               predicted_examples1,
                                                                                               predicted_examples2)

    print("=============== GT and Predicted EXAMPLES with METRICS ======================")
    for i_example in range(len(gt_examples)):
        print(gt_examples[i_example])
        print(predicted_examples1[i_example])
        print(predicted_examples2[i_example])
        print("-----------")

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
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--model_id', type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    main(args)