import nltk
import sys
import glob
import os
import shutil
from tqdm import tqdm
from parse_data import DATA_DIR, OUTPUT_DIR, DATAPATH
from typing import List, Tuple, Dict
from ast import literal_eval as make_tuple

method = sys.argv[1]

lemmatize = True
rm_stopwords = True
num_sentences = 5
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


DATA_TO_TEST = {
    'Overall': 'test.txt.oracle',
    'Future': 'future/test.txt.oracle',
    'Contribution': 'contribution/test.txt.oracle',
    'Baseline': 'baseline/test.txt.oracle',
    'Dataset': 'dataset/test.txt.oracle',
    'Metric': 'metric/test.txt.oracle',
    'Motivation': 'motivation/test.txt.oracle'
}


def clean_sentence(tokens):
    # print('Cleaning sentences...')
    tokens = [t.lower() for t in tokens]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if rm_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def get_probabilities(cluster, lemmatize, rm_stopwords):
    # print('Getting probabilities...')
    # Store word probabilities for this cluster
    word_ps = {}
    # Keep track of the number of tokens to calculate probabilities later
    token_count = 0.0
    # Gather counts for all words in all documents
    for path in cluster:
        with open(path) as f:
            tokens = clean_sentence(nltk.word_tokenize(f.read()))
            token_count += len(tokens)
            for token in tokens:
                if token not in word_ps:
                    word_ps[token] = 1.0
                else:
                    word_ps[token] += 1.0
    # Divide word counts by the number of tokens across all files
    for word_p in word_ps:
        word_ps[word_p] = word_ps[word_p]/float(token_count)
    return word_ps


def get_sentences(cluster):
    # print('Getting sentences...')
    sentences = []
    for path in cluster:
        with open(path) as f:
            sentences += [line.strip() for line in f.readlines()]
    return sentences


def clean_sentence(tokens):
    tokens = [t.lower() for t in tokens]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if rm_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def score_sentence(sentence, word_ps):
    score = 0.0
    num_tokens = 0.0
    sentence = nltk.word_tokenize(sentence)
    tokens = clean_sentence(sentence)
    for token in tokens:
        if token in word_ps:
            score += word_ps[token]
            num_tokens += 1.0
    return float(score)/float(num_tokens)


def max_sentence(sentences, word_ps, simplified):
    max_sentence = None
    max_score = None
    for sentence in sentences:
        score = score_sentence(sentence, word_ps)
        if max_score == None or score > max_score:
            max_sentence = sentence
            max_score = score
    if not simplified:
        update_ps(max_sentence, word_ps)
    return max_sentence, score


def update_ps(max_sentence, word_ps):
    sentence = nltk.word_tokenize(max_sentence)
    sentence = clean_sentence(sentence)
    for word in sentence:
        try:
            word_ps[word] = word_ps[word]**2
        except:
            print(f'warning: word "{word}" not found in word_ps')
    return True


def orig(cluster):
    cluster = glob.glob(cluster)
    word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
    sentences = get_sentences(cluster)
    summary = []
    for i in range(num_sentences):
        summary.append(max_sentence(sentences, word_ps, False))
    return summary, sentences


def simplified(cluster):
    cluster = glob.glob(cluster)
    word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
    sentences = get_sentences(cluster)
    summary = []
    for i in range(num_sentences):
        summary.append(max_sentence(sentences, word_ps, True))
    return summary, sentences


def leading(cluster):
    cluster = glob.glob(cluster)
    sentences = get_sentences(cluster)
    summary = []
    for i in range(num_sentences):
        summary.append(sentences[i])
    return summary, sentences

# COPY from LexRank


def summaries_to_sent_ids(papers: List[List[str]], summaries: List[List[str]]):
    print('Converting summaries to ids...')
    sent_ids = []
    for paper, summary in tqdm(zip(papers, summaries), total=len(papers)):
        temp_sent = []
        for sentence in summary:
            try:
                temp_sent.append(paper.index(sentence))
            except:
                print(f'warning: can\'t find sentence "{sentence}"')
        sent_ids.append(temp_sent)
    return sent_ids


def calculate_performance(predicts: List[Tuple[int]], golds: List[Tuple[int]]):
    print('Calculating performance...')
    total_gold_positive, total_predicted_positive = 0, 0
    total_hit1, total_correct = 0, 0
    for i, (os, ts) in tqdm(enumerate(zip(predicts, golds)), total=len(golds)):
        os = set(os)
        ts = set(ts)

        correct = os & ts
        total_correct += len(correct)
        if len(correct) > 0:
            total_hit1 += 1
        only_in_predict = os - ts
        only_in_annotation = ts - os

        total_gold_positive += len(ts)
        total_predicted_positive += len(os)
        precision = total_correct / total_predicted_positive
        recall = total_correct / total_gold_positive
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

    return {
        'acc_hit1': total_hit1 / len(golds),
        'p': precision,
        'r (acc_sentence_level)': recall,
        'f1': f1
    }


def eval_write_output(summaries: List[List[str]], sent_ids: List[List[int]], scores: List[List[int]]):
    """ evaluate and write sentence prediction and score """
    print('Writting output...')
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Predict

    predicts_file = os.path.join(
        OUTPUT_DIR, f'prediction_{method}_{num_sentences}.txt')

    predict_fp = open(predicts_file, 'w')

    for summary, scores_cont in tqdm(zip(summaries, scores), total=len(summaries)):
        predict_fp.write(f'{summary} {scores_cont}\n')

    predict_fp.close()

    # Performance

    performance_file = os.path.join(
        OUTPUT_DIR, f'performance_{method}_{num_sentences}.txt')
    
    performance_fp = open(performance_file, 'w')

    for topic, test_file in DATA_TO_TEST.items():
        with open(os.path.join(DATAPATH, test_file), 'r') as stream:
            raw_labels = stream.readlines()
        labels = [make_tuple(raw_label) for raw_label in raw_labels]
        performance = calculate_performance(sent_ids, labels)
        print(topic, performance)
        performance_fp.write(f'{topic} {performance}\n')

    performance_fp.close()

#################


def main():
    # method = 'orig'
    cluster = os.path.join(DATA_DIR, '*.txt')
    papers = []
    scores = []
    summaries = []
    for filename in tqdm(glob.glob(cluster)):
        result, sentences = eval(method + "('" + filename + "')")
        summary, score = list(zip(*result))
        papers.append(sentences)
        scores.append(score)
        summaries.append(summary)

    sent_ids = summaries_to_sent_ids(papers, summaries)

    eval_write_output(summaries, sent_ids, scores)


if __name__ == '__main__':
    main()
