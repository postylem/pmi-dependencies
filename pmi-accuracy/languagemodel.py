"""
Methods for getting probability estimates from a language model
Classes should exist for XLNet, BERT, ...ELMo, baselines...
-
March 2020
"""
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class LanguageModel:
  """
  Base class for getting probability estimates from a pretrained contextual embedding model.
  Contains methods to be used by XLNet or BERT
  """
  def ptb_tokenlist_to_pmi_matrix(
    self, ptb_tokenlist, add_special_tokens=True,
    pad_left=None, pad_right=None, verbose=True):
    """Maps tokenlist to PMI matrix (override in implementing class)."""
    raise NotImplementedError

class XLNetSentenceDataset(torch.utils.data.Dataset):
  """Dataset class for XLNet"""
  def __init__(self, input_ids, ptbtok_to_span, span_to_ptbtok,
               mask_id=6, n_pad_left=0, n_pad_right=0):
    self.input_ids = input_ids
    self.n_pad_left = n_pad_left
    self.n_pad_right = n_pad_right
    self.mask_id = mask_id
    self.ptbtok_to_span = ptbtok_to_span
    self.span_to_ptbtok = span_to_ptbtok
    self._make_tasks()
  @staticmethod
  def collate_fn(batch):
    """concatenate and prepare batch"""
    tbatch = {}
    tbatch["input_ids"] = torch.LongTensor([b['input_ids'] for b in batch])
    tbatch["perm_mask"] = torch.FloatTensor([b['perm_mask'] for b in batch])
    tbatch["target_mask"] = torch.FloatTensor([b['target_mask'] for b in batch])
    tbatch["target_id"] = [b['target_id'] for b in batch]
    tbatch["source_span"] = [b['source_span'] for b in batch]
    tbatch["target_span"] = [b['target_span'] for b in batch]
    return tbatch

  def _make_tasks(self):
    tasks = []
    len_s = len(self.input_ids) # length in subword tokens
    len_t = len(self.ptbtok_to_span) # length in ptb tokens
    for source_span in self.ptbtok_to_span:
      for target_span in self.ptbtok_to_span:
        for idx_target, target_pos in enumerate(target_span):
          # these are the positions of the source span
          abs_source = [self.n_pad_left + s for s in source_span]
          # this is the token we want to predict in the target span
          abs_target_curr = self.n_pad_left + target_pos
          # these are all the tokens we need to mask in the target span
          abs_target_next = [self.n_pad_left + t
                             for t in target_span[idx_target:]]
          # we replace all hidden target tokens with <mask>
          input_ids = np.array(self.input_ids)
          input_ids[abs_target_next] = self.mask_id
          # create permutation mask
          perm_mask = np.zeros((len_s, len_s))
          perm_mask[:, abs_target_next] = 1.
          # if the source span is different from target span,
          # then we need to mask all of its tokens
          if source_span != target_span:
            input_ids[abs_source] = self.mask_id
            perm_mask[:, abs_source] = 1.
          # build prediction mask
          target_mask = np.zeros((1, len_s))
          target_mask[0, abs_target_curr] = 1.
          # build all
          task_dict = {}
          task_dict["input_ids"] = input_ids
          task_dict["source_span"] = source_span
          task_dict["target_span"] = target_span
          task_dict["target_mask"] = target_mask
          task_dict["perm_mask"] = perm_mask
          task_dict["target_id"] = self.input_ids[abs_target_curr]
          tasks.append(task_dict)
    self._tasks = tasks

  def __len__(self):
    return len(self._tasks)

  def __getitem__(self, idx):
    return self._tasks[idx]

class XLNet(LanguageModel):
  """Class for using XLNet as estimator"""
  def __init__(self, device, model_spec, batchsize):
    from transformers import XLNetLMHeadModel, XLNetTokenizer
    self.device = device
    self.model = XLNetLMHeadModel.from_pretrained(model_spec).to(device)
    self.tokenizer = XLNetTokenizer.from_pretrained(model_spec)
    self.batchsize = batchsize
    print(f"XLNet(batchsize={batchsize}) initialized.")

  def _create_pmi_dataset(self, ptb_tokenlist, 
    pad_left=None, pad_right=None,
    add_special_tokens=True, verbose=True):

    # map each ptb token to a list of spans
    # [0, 1, 2] -> [(0,), (1, 2,), (3,)]
    tokens, ptbtok_to_span = self.make_subword_lists(
      ptb_tokenlist, add_special_tokens=False)

    # map each span to the ptb token position
    # {(0,): 0, (1, 2,): 1, (3,): 2}
    span_to_ptbtok = {}
    for i, span in enumerate(ptbtok_to_span):
      assert span not in span_to_ptbtok
      span_to_ptbtok[span] = i

    # just convert here, tokenization is taken care of by make_subword_lists
    ids = self.tokenizer.convert_tokens_to_ids(tokens)

    # add special characters add optional padding
    if pad_left:
      pad_left_tokens, _ = self.make_subword_lists(pad_left)
      pad_left = self.tokenizer.convert_tokens_to_ids(pad_left_tokens)
      if add_special_tokens:
        pad_left += [self.tokenizer.sep_token_id]
    else:
      pad_left = []
    if pad_right:
      pad_right_tokens, _ = self.make_subword_lists(pad_right)
      pad_right = self.tokenizer.convert_tokens_to_ids(pad_right_tokens)
    else:
      pad_right = []
    if add_special_tokens:
      pad_right += [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]
    ids = pad_left + ids + pad_right
    n_pad_left = len(pad_left)
    n_pad_right = len(pad_right)

    if verbose:
      print(f"PTB token list:\n{ptb_tokenlist}")
      print(f"resulting subword tokens:\n{tokens}")
      print(f"ptbtok->pos:\n{ptbtok_to_span}")
      print(f"pos->ptbtok:\n{span_to_ptbtok}")
      print(f'padleft:{pad_left}\npadright:{pad_right}')
      print(f'input_ids:{ids}')

    # setup data loader
    dataset = XLNetSentenceDataset(
      ids, ptbtok_to_span, span_to_ptbtok,
      mask_id=self.tokenizer.mask_token_id,
      n_pad_left=n_pad_left, n_pad_right=n_pad_right)
    loader = torch.utils.data.DataLoader(
      dataset, shuffle=False, batch_size=self.batchsize,
      collate_fn=XLNetSentenceDataset.collate_fn)
    return dataset, loader

  def ptb_tokenlist_to_pmi_matrix(
    self, ptb_tokenlist, add_special_tokens=True,
    pad_left=None, pad_right=None, verbose=True):

    # create dataset for observed ptb sentence
    dataset, loader = self._create_pmi_dataset(
      ptb_tokenlist, verbose=verbose,
      pad_left=pad_left, pad_right=pad_right,
      add_special_tokens=add_special_tokens)

    # use model to compute PMIs
    results = []
    for batch in tqdm(loader, leave=False):
      outputs = self.model(
        batch['input_ids'].to(self.device),
        perm_mask=batch['perm_mask'].to(self.device),
        target_mapping=batch['target_mask'].to(self.device))
      outputs = F.log_softmax(outputs[0], 2)
      for i, output in enumerate(outputs):
        # the token id we need to predict, this belongs to target span
        target_id = batch['target_id'][i]
        assert output.size(0) == 1
        log_target = output[0, target_id].item()
        result_dict = {}
        result_dict['source_span'] = batch['source_span'][i]
        result_dict['target_span'] = batch['target_span'][i]
        result_dict['log_target'] = log_target
        result_dict['target_id'] = target_id
        results.append(result_dict)

    num_ptbtokens = len(ptb_tokenlist)
    log_p = np.zeros((num_ptbtokens, num_ptbtokens))
    num = np.zeros((num_ptbtokens, num_ptbtokens))
    for result in results:
      log_target = result['log_target']
      source_span = result['source_span']
      target_span = result['target_span']
      ptbtok_source = dataset.span_to_ptbtok[source_span]
      ptbtok_target = dataset.span_to_ptbtok[target_span]
      if len(target_span) == 1:
        # sanity check: if target_span is 1 token, then we don't need
        # to accumulate subwords probabilities
        assert log_p[ptbtok_target, ptbtok_source] == 0.
      # we accumulate all log probs for subwords in a given span
      log_p[ptbtok_target, ptbtok_source] += log_target
      num[ptbtok_target, ptbtok_source] += 1

    # PMI(w_i, w_j | c ) = log p(w_i | c) - log p(w_i | c \ w_j)
    # log_p[i, i] is log p(w_i | c)
    # log_p[i, j] is log p(w_i | c \ w_j)
    log_p_wi_I_c = np.diag(log_p)
    pmi_matrix = log_p_wi_I_c[:, None] - log_p
    return pmi_matrix

  def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
    '''
    Takes list of items from Penn Treebank tokenized text,
    runs the tokenizer to decompose into the subword tokens expected by XLNet,
    including appending special characters '<sep>' and '<cls>', if specified.
    Implements some simple custom adjustments to make the results more like what might be expected.
    [TODO: this could be improved, if it is important.
    For instance, currently it puts an extra space before opening quotes]
    Returns:
      tokens: a flat list of subword tokens
      ptbtok_to_span: a list of tuples, of length = len(ptb_tokenlist <+ special tokens>)
        where the nth tuple is token indices for the nth ptb word.
    '''
    subword_lists = []
    for word in ptb_tokenlist:
      if word == '-LCB-': word = '{'
      elif word == '-RCB-': word = '}'
      elif word == '-LSB-': word = '['
      elif word == '-RSB-': word = ']'
      elif word == '-LRB-': word = '('
      elif word == '-RRB-': word = ')'
      word_tokens = self.tokenizer.tokenize(word)
      subword_lists.append(word_tokens)
    if add_special_tokens:
      subword_lists.append(['<sep>'])
      subword_lists.append(['<cls>'])
    # Custom adjustments below
    for i, subword_list_i in enumerate(subword_lists):
      if subword_list_i[0][0] == '▁' and subword_lists[i-1][-1] in ('(','[','{'):
        # print(f'{i}: removing extra space after character. {subword_list_i[0]} => {subword_list_i[0][1:]}')
        subword_list_i[0] = subword_list_i[0][1:]
        if subword_list_i[0] == '':
          subword_list_i.pop(0)
      if subword_list_i[0] == '▁' and subword_list_i[1] in (')',']','}',',','.','"',"'","!","?") and i != 0:
        # print(f'{i}: removing extra space before character. {subword_list_i} => {subword_list_i[1:]}')
        subword_list_i.pop(0)
      if subword_list_i == ['▁', 'n', "'", 't'] and i != 0:
        # print(f"{i}: fixing X▁n't => Xn 't ")
        del subword_list_i[0]
        del subword_list_i[0]
        subword_lists[i-1][-1] += 'n'

    tokens = list(itertools.chain(*subword_lists))
    ptbtok_to_span = []
    pos = 0
    for token in subword_lists:
      ptbtok_to_span.append(())
      for _ in token:
        ptbtok_to_span[-1] = ptbtok_to_span[-1] + (pos,)
        pos += 1
    return tokens, ptbtok_to_span


class BERT(LanguageModel):
  """Class for using BERT as estimator"""
  def __init__(self, device, model_spec, batchsize):
    print("initializing...")
    from transformers import BertForMaskedLM, BertTokenizer
    self.device = device
    self.model = BertForMaskedLM.from_pretrained(model_spec).to(device)
    self.tokenizer = BertTokenizer.from_pretrained(model_spec)
    self.batchsize = batchsize
    print(f"BERT model initialized (batchsize={batchsize}).")

  def ptb_tokenlist_to_pmi_matrix(
    self, ptb_tokenlist, add_special_tokens=True,
    pad_left=None, pad_right=None, verbose=True):
    '''
    input: ptb_tokenlist: PTB-tokenized sentence as list
    return: pmi matrix
    '''
    raise NotImplementedError
