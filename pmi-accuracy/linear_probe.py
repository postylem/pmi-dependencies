'''
Training a linear probe to extract POS embeddings.
-
July 2020
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from transformers import AutoModel, AutoTokenizer


class CONLLReader():
    def __init__(self, conll_cols, additional_field_name=None):
        if additional_field_name:
            conll_cols += [additional_field_name]
        self.conll_cols = conll_cols
        self.observation_class = namedtuple("Observation", conll_cols)
        self.additional_field_name = additional_field_name

    # Data input
    @staticmethod
    def generate_lines_for_sent(lines):
        '''Yields batches of lines describing a sentence in conllx.

        Args:
            lines: Each line of a conllx file.
        Yields:
            a list of lines describing a single sentence in conllx.
        '''
        buf = []
        for line in lines:
            if line.startswith('#'):
                continue
            if not line.strip():
                if buf:
                    yield buf
                    buf = []
                else:
                    continue
            else:
                buf.append(line.strip())
        if buf:
            yield buf

    def load_conll_dataset(self, filepath):
        '''Reads in a conllx file; generates Observation objects

        For each sentence in a conllx file, generates a single Observation
        object.

        Args:
            filepath: the filesystem path to the conll dataset
            observation_class: namedtuple for observations

        Returns:
        A list of Observations
        '''
        observations = []
        lines = (x for x in open(filepath))
        for buf in self.generate_lines_for_sent(lines):
            conllx_lines = []
            for line in buf:
                conllx_lines.append(line.strip().split('\t'))
            if self.additional_field_name:
                newfield = [None for x in range(len(conllx_lines))]
                observation = self.observation_class(
                    *zip(*conllx_lines), newfield)
            else:
                observation = self.observation_class(
                    *zip(*conllx_lines))
            observations.append(observation)
        return observations


class POSProbe(nn.Module):
    """just applies a linear transform W and log_softmax the result"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args['hidden_dim']
        self.pos_vocabsize = len(args['POS_set'])
        self.linear = nn.Linear(self.hidden_dim, self.pos_vocabsize)
        self.to(args['device'])

    def forward(self, H):
        """
        Performs the linear transform W,
        and takes the log_softmax to get a
        log probability distribution over POS tags
        Args:
            H: a batch of sequences, i.e. a tensor of shape
                (batch_size, max_slen, hidden_dim)
        Returns:
            distances: a list of distance matrices, i.e. a tensor of shape
                (batch_size, max_slen, max_slen)
        """
        # apply W to batch H to get shape (batch_size, max_slen, pos_vocabsize)
        WH = self.linear(H)
        prediction = F.log_softmax(WH, dim=-1)
        return prediction


class POSProbeLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, prediction_batch, label_batch, length_batch):
        """
        Gets the Xent loss between the predicted POS distribution and the label
        Args:
            prediction_batch: pytorch batch of softmaxed POS predictions
            label_batch: pytorch batch of true POS label ids (torch.long)
            length_batch: pytorch batch of sentence lengths

        Returns:
            A tuple of:
                batch_loss: average loss in the batch
                number_of_sentences: number of sentences in the batch
        """
        number_of_sentences = torch.sum(length_batch != 0).float()
        if number_of_sentences > 0:
            batch_loss = nn.CrossEntropyLoss()(
                prediction_batch.view(-1, len(self.args['POS_set'])),
                label_batch.view(-1))
        else:
            batch_loss = torch.tensor(0.0, device=self.args['device'])
        return batch_loss, number_of_sentences


class POSDataset(Dataset):
    """ PyTorch dataloader for POS from Observations.
    """
    def __init__(self, observations, tokenizer, observation_class, POS_set):
        '''
        Args:
            observations: A list of Observations describing a dataset
            tokenizer: an instance of a transformers Tokenizer class
            observation_class: a namedtuple class specifying the fields
            POS_set: the set of POS tags
        '''
        self.observations = observations
        self.POS_set = POS_set
        self.tokenizer = tokenizer
        self.POS_to_id = {POS: i for i, POS in enumerate(self.POS_set)}
        self.observation_class = observation_class
        self.input_ids, self.pos_ids = self.get_input_ids_and_pos_ids()

    def sentences_to_idlists(self):
        '''Replaces strings in an Observation with lists of integer ids.
        Returns:
            A list of observations with nested integer-lists as sentence fields
        '''
        idlist_observations = []
        for obs in tqdm(self.observations, desc="[getting subtoken ids]"):
            idlist = tuple([self.subword_ids(item) for item in obs.sentence])
            idlist_observations.append(self.observation_class(
                # replace 'sentence' field with nested list of token ids
                obs[0], idlist, *obs[2:]))
        return idlist_observations

    def subword_ids(self, item):
        '''Gets a list of subword ids for an item (word).'''
        return self.tokenizer.encode(item, add_special_tokens=False)


    def get_input_ids_and_pos_ids(self):
        '''Gets flat list of input ids and POS ids for each observation
        Returns:
            input_ids: a list containgin a list of input ids for each
                observation
            pos_ids: a list containing a list of POS 'ids' for each
                observation, which will repeat when there is more than
                one subtoken per POS tagged word.
        '''
        idlist_observations = self.sentences_to_idlists()
        subtoken_id_lists = [obs.sentence for obs in idlist_observations]
        pos_label_lists = [obs.xpos_sentence for obs in idlist_observations]
        input_ids, pos_ids = self.repeat_POS_to_match(
            subtoken_id_lists, pos_label_lists)
        return input_ids, pos_ids

    def repeat_POS_to_match(self, list_id, list_POS):
        assert len(list_POS) == len(list_id), "list lengths don't match"
        new_id = []
        new_POS = []
        for i, el_id in enumerate(list_id):
            newlist_id = []
            newlist_POS = []
            for j, elel_id in enumerate(el_id):
                for token_id in elel_id:
                    newlist_id.append(token_id)
                    newlist_POS.append(self.POS_to_id[list_POS[i][j]])
            new_id.append(newlist_id)
            new_POS.append(newlist_POS)
        return new_id, new_POS

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        return self.input_ids[index], self.pos_ids[index], len(self.input_ids[index])

    @staticmethod
    def collate_fn(batch):
        print(type(batch))
        print(len(batch))
        print(batch)
        input_ids = torch.LongTensor([b[0] for b in batch])
        pos_ids = torch.LongTensor([b[1] for b in batch])
        lengths = torch.LongTensor([b[2] for b in batch])
        return input_ids, pos_ids, lengths


def run_train_probe(args, probe, loss, train_loader, dev_loader):
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=0)
    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1

    for epoch_i in tqdm(range(args['epochs']), desc='training'):
        epoch_train_loss = 0
        epoch_dev_loss = 0
        epoch_train_epoch_count = 0
        epoch_dev_epoch_count = 0
        epoch_train_loss_count = 0
        epoch_dev_loss_count = 0

        for batch in tqdm(train_loader, desc='training batch'):
            print(batch)
            probe.train()
            optimizer.zero_grad()
            observation_batch, label_batch, length_batch = batch
            embedding_batch = MODEL(observation_batch)
            prediction_batch = probe(embedding_batch)
            batch_loss, count = loss(
                prediction_batch, label_batch, length_batch)
            batch_loss.backward()
            epoch_train_loss += (batch_loss.detach() *
                                 count.detach()).cpu().numpy()
            epoch_train_epoch_count += 1
            epoch_train_loss_count += count.detach().cpu().numpy()
            optimizer.step()

        for batch in tqdm(dev_loader, desc='dev batch'):
            optimizer.zero_grad()
            probe.eval()
            observation_batch, label_batch, length_batch = batch
            embedding_batch = MODEL(observation_batch)
            prediction_batch = probe(embedding_batch)
            batch_loss, count = loss(
                prediction_batch, label_batch, length_batch)
            epoch_dev_loss += (batch_loss.detach() *
                               count.detach()).cpu().numpy()
            epoch_dev_loss_count += count.detach().cpu().numpy()
            epoch_dev_epoch_count += 1
        scheduler.step(epoch_dev_loss)
        tqdm.write(
            f'[epoch {epoch_i}]'
            f'train loss: {epoch_train_loss/epoch_train_loss_count},'
            f'dev loss: {epoch_dev_loss/epoch_dev_loss_count}'
            )
        if epoch_dev_loss/epoch_dev_loss_count < min_dev_loss - 0.0001:
            save_path = os.path.join(args['results_path'], "state_dict.pt")
            torch.save(probe.state_dict(), save_path)
            min_dev_loss = epoch_dev_loss/epoch_dev_loss_count
            min_dev_loss_epoch = epoch_i
            tqdm.write('Saving probe state_dict')
        elif min_dev_loss_epoch < epoch_i - 4:
            tqdm.write('Early stopping')
            break


def train_probe(args, probe, loss, tokenizer):
    train_dataset, dev_dataset, _ = load_datasets(args, tokenizer)

    params = {'batch_size': 10, 'shuffle': False,
              'collate_fn': POSDataset.collate_fn}
    train_loader = DataLoader(train_dataset, **params)
    dev_loader = DataLoader(dev_dataset, **params)

    run_train_probe(args, probe, loss, train_loader, dev_loader)


def load_datasets(args, tokenizer):
    '''
    Get pytorch Datasets for train, dev, test observations
    '''
    train_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['train_path'])
    dev_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['dev_path'])
    test_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['test_path'])
    reader = CONLLReader(args['conll_fieldnames'])
    train_obs = reader.load_conll_dataset(train_corpus_path)
    dev_obs = reader.load_conll_dataset(dev_corpus_path)
    test_obs = reader.load_conll_dataset(test_corpus_path)

    obs_class = reader.observation_class
    POS_set = args['POS_set']
    train_dataset = POSDataset(train_obs, tokenizer, obs_class, POS_set)
    dev_dataset = POSDataset(dev_obs, tokenizer, obs_class, POS_set)
    test_dataset = POSDataset(test_obs, tokenizer, obs_class, POS_set)

    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    SPEC = 'bert-base-uncased'

    MODEL = AutoModel.from_pretrained(SPEC).to(DEVICE)
    TOKENIZER = AutoTokenizer.from_pretrained(SPEC)

    # UPOS_TAGSET = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ',
    #                'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
    #                'SCONJ', 'SYM', 'VERB', 'X']

    XPOS_TAGSET = ['#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':',
                   'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
                   'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
                   'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                   'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                   'WDT', 'WP', 'WP$', 'WRB', '``']
    ARGS = dict(
        device=DEVICE,
        hidden_dim=MODEL.config.hidden_size,
        epochs=20,
        results_path="probe-results/",
        corpus=dict(root='ptb3-wsj-data/',
                    train_path='CUSTOM.conllx',  # ptb3-wsj-train.conllx
                    dev_path='CUSTOM4.conllx',    # ptb3-wsj-dev.conllx
                    test_path='CUSTOM4.conllx'),  # ptb3-wsj-test.conllx
        conll_fieldnames=[  # Columns of CONLL file
            'index', 'sentence', 'lemma_sentence', 'upos_sentence',
            'xpos_sentence', 'morph', 'head_indices',
            'governance_relations', 'secondary_relations', 'extra_info'],
        POS_set=XPOS_TAGSET,
        )

    PROBE = POSProbe(ARGS)
    LOSS = POSProbeLoss(ARGS)

    train_probe(ARGS, PROBE, LOSS, TOKENIZER)
