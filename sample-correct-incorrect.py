""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter, defaultdict
from random import sample
import random
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    exact_match_ids = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                is_exact_match = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                if is_exact_match:
                    exact_match_ids.append(qa['id'])
                exact_match += is_exact_match
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1, 'exact_match_ids': exact_match_ids}

def sample_correct_incorrect(original_correct_list, new_correct_list, num_sample):
    original_correct = set(original_correct_list)
    new_correct = set(new_correct_list)
    both_correct = original_correct & new_correct
    new_incorrect = original_correct - new_correct
    return {'both_correct': sample(both_correct, num_sample), 'new_incorrect': sample(new_incorrect, num_sample)}

def get_contexts(qa_ids, original_dataset, new_dataset, original_predictions, new_predictions):
    contexts = defaultdict(dict)
    for article in original_dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] in qa_ids:
                    c = contexts[qa['id']]
                    c['context'] = paragraph['context']
                    c['answers'] = qa['answers']
                    c['original_question'] = qa['question']

    for article in new_dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] in qa_ids:
                    c = contexts[qa['id']]
                    c['new_question'] = qa['question']
                    c['original_prediction'] = original_predictions[qa['id']]
                    c['new_prediction'] = new_predictions[qa['id']]
    return contexts

if __name__ == '__main__':
    random.seed(0)
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('original_dataset_file', help='Dataset File with the Original Questions')
    parser.add_argument('new_dataset_file', help='Dataset File with N-Word Questions')
    parser.add_argument('original_prediction_file', help='Prediction File on the Original Questions')
    parser.add_argument('new_prediction_file', help='Prediction File on the N-Word Questions')
    parser.add_argument('num_sample', type=int, help='Prediction File on the N-Word Questions')
    parser.add_argument('output_file', help='File to Store the Output JSON')
    args = parser.parse_args()
    with open(args.original_dataset_file) as original_dataset_file:
        original_dataset_json = json.load(original_dataset_file)
        if (original_dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + original_dataset_json['version'],
                  file=sys.stderr)
        original_dataset = original_dataset_json['data']
    with open(args.new_dataset_file) as new_dataset_file:
        new_dataset_json = json.load(new_dataset_file)
        if (new_dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + new_dataset_json['version'],
                  file=sys.stderr)
        new_dataset = new_dataset_json['data']
    with open(args.original_prediction_file) as original_prediction_file:
        original_predictions = json.load(original_prediction_file)
    with open(args.new_prediction_file) as new_prediction_file:
        new_predictions = json.load(new_prediction_file)
    original_prediction_stats = evaluate(new_dataset, original_predictions)
    new_prediction_stats = evaluate(new_dataset, new_predictions)
    correct_incorrect = sample_correct_incorrect(
        original_prediction_stats['exact_match_ids'], new_prediction_stats['exact_match_ids'], args.num_sample)
    both_correct_contexts = get_contexts(
        correct_incorrect['both_correct'],
        original_dataset, new_dataset, original_predictions, new_predictions)
    new_incorrect_contexts = get_contexts(correct_incorrect['new_incorrect'],
        original_dataset, new_dataset, original_predictions, new_predictions)
    final_stats = {
        'cmd': 'python3 ' + ' '.join(sys.argv),
        'original_exact_match': original_prediction_stats['exact_match'],
        'original_f1': original_prediction_stats['f1'],
        'new_exact_match': new_prediction_stats['exact_match'],
        'new_f1': new_prediction_stats['f1'],
        'new_incorrect_contexts': new_incorrect_contexts,
        'both_correct_contexts': both_correct_contexts
    }
    with open(args.output_file, 'w', encoding="utf-8") as output_file:
        json.dump(final_stats, output_file, indent=2, ensure_ascii=False)
