import argparse
from collections import defaultdict
import json
import re

NEW_QUESTION_LENGTH = 3

def tw(w):
  regex = re.compile('[^a-zA-Z0-9]')
  return regex.sub('', w)

def split_string(s):
  return [tw(w) for w in s.split()]

def string_wc(s):
  word_count = defaultdict(int)
  for w in split_string(s):
    word_count[w] += 1
  return word_count

def generate_new_queation(all_wc, context_wc, original_question):
  original_question_words = split_string(original_question)
  score_index = []
  for i in range(len(original_question_words)):
    w = original_question_words[i]
    # Skip 'Which', etc.
    # if context_wc[w] == 0:
      # continue
    score_index.append((context_wc[w] - all_wc[w], i))
  score_index.sort(reverse=True)
  top_n = [i for s, i in score_index[:NEW_QUESTION_LENGTH]]
  top_n.sort()
  return ' '.join([original_question_words[i] for i in top_n])

def generate_new_dataset(dataset_json):
  all_wc = defaultdict(int)
  dataset = dataset_json['data']
  for data in dataset:
    for p in data['paragraphs']:
      for w in p['context'].split():
        all_wc[w] += 1

  for data in dataset:
    for p in data['paragraphs']:
      context_wc = string_wc(p['context'])
      for qa in p['qas']:
        # if qa['id'] == '56be4db0acb8001400a502f0':
          # import pdb; pdb.set_trace()
        qa['question'] = generate_new_queation(all_wc, context_wc, qa['question'])
  return dataset_json

if __name__ == '__main__':
  expected_version = '1.1'
  parser = argparse.ArgumentParser(
    description='Three-word question generator for SQuAD ' + expected_version)
  parser.add_argument('dataset_file', help='Dataset file')
  parser.add_argument('output_dataset_file', help='Output Dataset File')
  args = parser.parse_args()
  dataset_json = None
  with open(args.dataset_file) as dataset_file:
    dataset_json = json.load(dataset_file)
    if (dataset_json['version'] != expected_version):
      print('Evaluation expects v-' + expected_version +
          ', but got dataset with v-' + dataset_json['version'],
          file=sys.stderr)
  new_dataset = json.dumps(generate_new_dataset(dataset_json), indent=2)
  with open(args.output_dataset_file, 'w') as output_dataset_file:
    output_dataset_file.write(new_dataset)
