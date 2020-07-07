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
from collections import namedtuple
from transformers import AutoModel

import main


class POSProbe(nn.Module):
    """just applies a linear transform W and log_softmax the result"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args['hidden_dim']
        self.pos_vocabsize = args['pos_vocabsize']
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
        Gets the KL-divergence loss between the predicted POS distribution
        and the label POS distribution (1-hot)
        Args:
            prediction_batch: pytorch batch of softmaxed POS predictions
            label_batch: pytorch batch of true POS labels (as 1-hot vectors)
            length_batch: pytorch batch of sentence lengths

        Returns:
            A tuple of:
                batch_loss: average loss in the batch
                number_of_sentences: number of sentences in the batch
        """
        number_of_sentences = torch.sum(length_batch != 0).float()
        if number_of_sentences > 0:
            batch_loss = nn.KLDivLoss(reduction='batchmean')(
                prediction_batch, label_batch.float())
        else:
            batch_loss = torch.tensor(0.0, device=self.args['device'])
        return batch_loss, number_of_sentences


class ObservationDataset(torch.utils.data.Dataset):
    """ Observations PyTorch dataloader."""
    def __init__(self, observations):
        self.observations = observations
        self.set_labels(observations)

    def set_labels(self, observations):
        """ Constructs aand stores label for each observation.

        Args:
          observations: A list of observations describing a dataset
          task: a Task object which takes Observations and constructs labels.
        """
        self.labels = []
        for observation in tqdm(observations, desc='[computing labels]'):
            self.labels.append(self.POS_vector(observation))

    def POS_vector(self, obs):
        # POSlist = obs.xpos_sentence
        # TODO: get the 1-hot POS vector for each word.
        return None

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]


def run_train_probe(args, probe, loss, train_dataset, dev_dataset):
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=0.001)
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

        for batch in tqdm(train_dataset, desc='training batch'):
            probe.train()
            optimizer.zero_grad()
            observation_batch, label_batch, length_batch, _ = batch
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

        for batch in tqdm(dev_dataset, desc='dev batch'):
            optimizer.zero_grad()
            probe.eval()
            observation_batch, label_batch, length_batch, _ = batch
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


def train_probe(args, probe, loss):
    train_dataset, dev_dataset, _ = load_datasets(args)
    run_train_probe(args, probe, loss, train_dataset, dev_dataset)


def load_datasets(args):
    train_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['train_path'])
    dev_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['dev_path'])
    test_corpus_path = os.path.join(
        args['corpus']['root'],
        args['corpus']['test_path'])
    obs_class = namedtuple('Observation', args['conll_fieldnames'])
    train_obs = main.load_conll_dataset(train_corpus_path, obs_class)
    dev_obs = main.load_conll_dataset(dev_corpus_path, obs_class)
    test_obs = main.load_conll_dataset(test_corpus_path, obs_class)

    train_dataset = ObservationDataset(train_obs)
    dev_dataset = ObservationDataset(dev_obs)
    test_dataset = ObservationDataset(test_obs)

    return train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    MODEL = AutoModel.from_pretrained('bert-base-uncased').to(DEVICE)
    POS_VOCABSIZE = 30  # or whatever

    ARGS = dict(
        device=DEVICE,
        hidden_dim=MODEL.config.hidden_size,
        pos_vocabsize=POS_VOCABSIZE,
        epochs=20,
        results_path="probe-results/",
        corpus=dict(root='ptb3-wsj-data/',
                    train_path='ptb3-wsj-train.conllx',
                    dev_path='ptb3-wsj-dev.conllx',
                    test_path='ptb3-wsj-test.conllx'),
        conll_fieldnames=[  # Columns of CONLL file
            'index', 'sentence', 'lemma_sentence', 'upos_sentence',
            'xpos_sentence', 'morph', 'head_indices',
            'governance_relations', 'secondary_relations', 'extra_info']
        )

    PROBE = POSProbe(ARGS)
    LOSS = POSProbeLoss(ARGS)

    train_probe(ARGS, PROBE, LOSS)
