'''
Get PMI matrices for sentences in SyntaxGym tasks.
'''

from datetime import datetime
from argparse import ArgumentParser
import os
import shutil
import torch
from tqdm import tqdm
import numpy as np

import languagemodel
import embedding


def load_sentences_txt(path):
    with open(path) as f:
        sentences = [line.strip().split(' ')
                     for line in f]
    return sentences


def get_pmi(sentences, n_sentences='all', verbose=False):
    '''get estimates get scores for n (default all) sentences'''

    savez_dict = dict()

    if n_sentences == 'all':
        n_sentences = len(sentences)

    for i, sentence in enumerate(tqdm(sentences[:n_sentences], leave=False)):
        # get a pmi matrix and a pseudo-logprob for the sentence
        pmi_matrix, pseudo_loglik = MODEL.ptb_tokenlist_to_pmi_matrix(
            sentence, add_special_tokens=True, verbose=verbose,
            pad_left=None, pad_right=None)
        if verbose:
            tqdm.write(f"pmi for sentence {i}\n{sentence}")
            tqdm.write(f"pseudo_loglik: {pseudo_loglik}")
            tqdm.write(f"{pmi_matrix}")

        savez_dict[str(i)] = pmi_matrix
    return savez_dict


def save_pmi(
        sentences, n_sentences,
        resultsdir, outfilename='pmi_matrices.npz',
        verbose=False):
    savez_dict = get_pmi(sentences, n_sentences, verbose=verbose)
    save_filepath = os.path.join(resultsdir, outfilename)
    np.savez(save_filepath, **savez_dict)


if __name__ == '__main__':
    ARGP = ArgumentParser()
    ARGP.add_argument('--n_sentences', default='all',
                      help='number of sentences to look at')
    ARGP.add_argument('--txt', default='cpllab-syntactic-generalization/test_suites/txt',
                      help='''specify path/to/results/sentences.txt
                           or directory containing multiple txt files''')
    ARGP.add_argument('--model_spec', default='xlnet-base-cased',
                      help='''specify model
                      (e.g. "xlnet-base-cased", "bert-large-cased")''')
    ARGP.add_argument('--results_dir', default='results/',
                      help='specify path/to/results/directory/')
    ARGP.add_argument('--model_path',
                      help='optional: load model state or embeddings from file')
    ARGP.add_argument('--batch_size', default=32, type=int)
    ARGP.add_argument('--archive', action='store_true',
                      help='to zip archive the pmi matrices folder')
    CLI_ARGS = ARGP.parse_args()

    SPEC_STRING = str(CLI_ARGS.model_spec)
    if CLI_ARGS.model_path and CLI_ARGS.model_spec == 'bert-base-uncased':
        # custom naming just for readability
        import re
        STEPS = re.findall("(\d+)", CLI_ARGS.model_path)
        STEPS = str(int(int(STEPS[-1]) / 1000.0))
        SPEC_STRING = SPEC_STRING + ".ckpt-" + STEPS + "k"

    N_SENTENCES = CLI_ARGS.n_sentences
    if N_SENTENCES != 'all':
        N_SENTENCES = int(N_SENTENCES)

    NOW = datetime.now()
    DATE_SUFFIX = f'{NOW.year}-{NOW.month:02}-{NOW.day:02}-{NOW.hour:02}-{NOW.minute:02}'
    SPEC_SUFFIX = SPEC_STRING+'('+str(CLI_ARGS.n_sentences)+')' if CLI_ARGS.n_sentences != 'all' else SPEC_STRING
    SUFFIX = 'SyntaxGym_' + SPEC_SUFFIX + '_' + DATE_SUFFIX
    RESULTS_DIR = os.path.join(CLI_ARGS.results_dir, SUFFIX + '/')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f'RESULTS_DIR: {RESULTS_DIR}\n')

    print('Running with CLI_ARGS:')
    with open(RESULTS_DIR+'info.txt', mode='w') as infofile:
        for arg, value in sorted(vars(CLI_ARGS).items()):
            argvalue = f"{arg}:\t{value}"
            infofile.write(argvalue+'\n')
            print(argvalue)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

    # Instantiate the language model to use for getting estimates

    if CLI_ARGS.model_spec.startswith('xlnet'):
        MODEL = languagemodel.XLNet(
            DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    elif (CLI_ARGS.model_spec.startswith('bert') or
          CLI_ARGS.model_spec.startswith('distilbert')):
        # DistilBERT will work just like BERT
        if CLI_ARGS.model_path:
            # load checkpointed weights from disk, if specified
            STATE = torch.load(CLI_ARGS.model_path)
            MODEL = languagemodel.BERT(
                DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size,
                state_dict=STATE)
        else:
            MODEL = languagemodel.BERT(
                DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    elif CLI_ARGS.model_spec.startswith('xlm'):
        MODEL = languagemodel.XLM(
            DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    elif CLI_ARGS.model_spec.startswith('bart'):
        MODEL = languagemodel.Bart(
            DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    elif CLI_ARGS.model_spec.startswith('gpt2'):
        MODEL = languagemodel.GPT2(
            DEVICE, CLI_ARGS.model_spec, CLI_ARGS.batch_size)
    elif CLI_ARGS.model_spec == 'w2v':
        W2V_PATH = CLI_ARGS.model_path
        MODEL = embedding.Word2Vec(
            DEVICE, CLI_ARGS.model_spec, W2V_PATH)
    else:
        raise ValueError(
            f'Model spec string {CLI_ARGS.model_spec} not recognized.')

    PMI_DIR = os.path.join(RESULTS_DIR, "pmi_matrices/")
    os.makedirs(PMI_DIR, exist_ok=True)

    if os.path.isfile(CLI_ARGS.txt):
        SENTENCES = load_sentences_txt(CLI_ARGS.txt)
        TASKNAME = os.path.splitext(os.path.basename(CLI_ARGS.txt))[0]
        print(f"getting pmi for: {TASKNAME}")
        save_pmi(
            SENTENCES, N_SENTENCES, PMI_DIR,
            outfilename=TASKNAME + '.npz', verbose=False)
    else:
        counter = 0
        for dir_entry in tqdm(os.scandir(CLI_ARGS.txt)):
            if counter < 1:
                SENTENCES = load_sentences_txt(dir_entry)
                TASKNAME = os.path.splitext(dir_entry.name)[0]
                tqdm.write(f"getting pmi for: {TASKNAME}")
                save_pmi(
                    SENTENCES, N_SENTENCES, PMI_DIR,
                    outfilename=TASKNAME + '.npz', verbose=False)
            counter += 1

    if CLI_ARGS.archive:
        shutil.make_archive(
            base_name=os.path.join(RESULTS_DIR, "pmi_matrices/"),
            format='zip',
            root_dir=PMI_DIR)
        shutil.rmtree(PMI_DIR)










