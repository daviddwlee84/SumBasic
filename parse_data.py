import os
import shutil
from tqdm import tqdm
import shutil

DATAPATH = '../../../data/test'
DATA_PATH = '../data'
OUTPUT_PATH = '../output'
TARGET = 'SumBasic'

DATA_DIR = os.path.join(DATA_PATH, TARGET)
OUTPUT_DIR = os.path.join(OUTPUT_PATH, TARGET)

DATA_TO_TEST = {
    'Overall': 'test.txt.oracle',
    'Future': 'future/test.txt.oracle',
    'Contribution': 'contribution/test.txt.oracle',
    'Baseline': 'baseline/test.txt.oracle',
    'Dataset': 'dataset/test.txt.oracle',
    'Metric': 'metric/test.txt.oracle',
    'Motivation': 'motivation/test.txt.oracle'
}

TEST_PAPER = {key: value.replace('oracle', 'src') for key, value in DATA_TO_TEST.items()}
TEST_PAPER_SECTION = {key: value.replace('oracle', 'section') for key, value in DATA_TO_TEST.items()}
PAPER_REF = {key: value.replace('oracle', 'ref') for key, value in DATA_TO_TEST.items()}

# Copied from Paper2PPT/neusum_pt/loglinear/Config.py
PossibleSection = {
    ".": [],
    "baseline": ["experiment", "experiments", "introduction", "baseline", "results", "result", "model", "related work"],
    "metric": ["introduction", "abstract", "evaluation", "experiment", "result"],
    "dataset": ["dataset", "experiment", "experiments"],
    "motivation": ["introduction", "abstract"],
    "future": ["conclusion", "future work"],
    "contribution": ["abstract", "introduction"]
}

def parse_data(topic: str):
    """ parse our annotation and write text files into DATA_DIR """

    parse = True
    data_dir = os.path.join(DATA_DIR, topic)
    if os.path.exists(data_dir):
        override = input('Data exist, override (delete and re-parse)? (Y/n): ')
        if override.lower() == 'y':
            shutil.rmtree(data_dir)
        else:
            parse = False
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(DATAPATH, TEST_PAPER[topic]), 'r') as stream:
        raw_papers = stream.readlines()
    papers = [paper.strip().split('##SENT##') for paper in raw_papers]

    if parse:
        print('Converting src to raw text...')
        for i, paper in tqdm(enumerate(papers), total=len(papers)):

            did = f'{i+1}.txt'

            text_file = os.path.join(data_dir, did)
            with open(text_file, 'w') as stream:
                # make sure the sent split are the same as our annotation
                stream.write('\n'.join(paper))


if __name__ == "__main__":
    for topic in DATA_TO_TEST.keys():
        parse_data(topic)
