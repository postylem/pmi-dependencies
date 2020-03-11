"""
Methods for getting probability estimates from a language model
Classes should exist for XLNet, BERT, ...ELMo, baselines...

-
Jacob Louis Hoover
February 2020
"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

class LanguageModel:
  """Abstract class for getting probability estimates.  (Do I need abstract classes?)"""

  @staticmethod
  def ptb_tokenlist_to_pmi_matrix(ptb_tokenlist):
    """Maps tokenlist to PMI matrix. To be overriden in implementing classes."""
    raise NotImplementedError

class XLNet(LanguageModel):
  """Class for using XLNet as estimator"""
  def __init__(self, device, xlnet_spec, batchsize, offline=False):
    if offline:
      # import pytorch transformers instead of transformers, for use on Compute Canada
      from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
    else:
      from transformers import XLNetLMHeadModel, XLNetTokenizer
    self.device = device
    self.model = XLNetLMHeadModel.from_pretrained(xlnet_spec).to(device)
    self.tokenizer = XLNetTokenizer.from_pretrained(xlnet_spec)
    self.batchsize = batchsize
    print(f"XLNet(batchsize={batchsize}) initialized.")

  def ptb_tokenlist_to_pmi_matrix(self, ptb_tokenlist, paddings=([], []), verbose=False):
    '''
    input: ptb_tokenlist: PTB-tokenized sentence as list
      paddings: tuple (prepad_tokenlist,postpad_tokenlist) each a list of PTB-tokens
    return: pmi matrix
    '''
    subwords_nested = self.make_subword_lists(ptb_tokenlist, add_special_tokens=False)
    # indices[i] = list of subtoken indices in subtoken list corresponding to word i in ptb_tokenlist
    indices = [self.word_index_to_subword_indices(i, subwords_nested) for i, _ in enumerate(subwords_nested)]

    flattened_sentence = [tok for sublist in subwords_nested for tok in sublist]
    if verbose:
      print(f"PTB tokenlist, on which tokenizer will be run:\n{ptb_tokenlist}")
      print(f"flattened list of resulting subtokens:\n{flattened_sentence}")
      print(f"correspondence indices:\n{indices}")

    # Now add padding before and/or after
    prepadding = [i for x in self.make_subword_lists(paddings[0], add_special_tokens=False) for i in x]
    postpadding = [i for x in self.make_subword_lists(paddings[1], add_special_tokens=True) for i in x]

    # # To use static prepadding text, and no postpadding, which is what Michaela used,
    # # uncomment this (but, it doesn't seem to change anything drastically):
    # padtextlist = """In 1991 , the remains of Russian Tsar Nicholas II and his family
    #             (except for Alexei and Maria) are discovered .
    #             The voice of Nicholas 's young son , Tsarevich Alexei Nikolaevich, narrates the
    #             remainder of the story . 1883 Western Siberia ,
    #             a young Grigori Rasputin is asked by his father and a group of men to perform magic .
    #             Rasputin has a vision and denounces one of the men as a horse thief . Although his
    #             father initially slaps him for making such an accusation , Rasputin watches as the
    #             man is chased outside and beaten . Twenty years later , Rasputin sees a vision of
    #             the Virgin Mary , prompting him to become a priest . Rasputin quickly becomes famous,
    #              with people , even a bishop , begging for his blessing . <eod>""".split()
    # paddings = (padtextlist, [])
    # prepadding = [i for x in self.make_subword_lists(paddings[0], add_special_tokens=False) for i in x]
    # postpadding = [i for x in self.make_subword_lists(paddings[1], add_special_tokens=True) for i in x]

    print(f"\n/¯¯¯¯¯¯ WHAT's GOING ON WITH THE PADDING? :\nprepadding\t{prepadding}\npostpadding\t{postpadding}\n")
    padded_input = [*prepadding, *flattened_sentence, *postpadding]
    print(f"padded_input:\n{padded_input}")
    perm_mask, target_mapping = self.make_mask_and_mapping(indices, padlens=(len(prepadding), len(postpadding)))
    if not target_mapping.size(0) == perm_mask.size(0) == 2*len(indices)*(len([i for x in indices for i in x])):
      raise ValueError("Uh oh! Check batch dimension on perm mask and target mapping tensors.")

    # start and finish indices of batchsized chunks (0,batchsize),(batchsize+1,2*batchsize), ... 
    index_tuples = self.make_chunks(perm_mask.size(0), self.batchsize)

    padded_sentence_as_ids = self.tokenizer.convert_tokens_to_ids(padded_input)
    list_of_output_tensors = []
    for index_tuple in tqdm(index_tuples, 
                            desc=f'{perm_mask.size(0)} in batches of {self.batchsize}', leave=False):
      # input_id_batch is just padded_sentence_as_ids repeated as many times as needed for the batch
      input_id_batch = torch.tensor(padded_sentence_as_ids).repeat((index_tuple[1]-index_tuple[0]), 1)
      with torch.no_grad():
        logits_outputs = self.model(
          input_id_batch.to(self.device),
          perm_mask=perm_mask[index_tuple[0]:index_tuple[1]].to(self.device),
          target_mapping=target_mapping[index_tuple[0]:index_tuple[1]].to(self.device))
        # note, logits_output is a degenerate tuple: ([self.batchsize, 1, self.tokenizer.vocabsize()],)
        # log softmax across the vocabulary (dimension 2 of the logits tensor)
        outputs = F.log_softmax(logits_outputs[0], 2)
        list_of_output_tensors.append(outputs)

    outputs_all_batches = torch.cat(list_of_output_tensors).cpu().numpy() 
    # outputs_all_batches is of shape (2 * numwords * numsubwords, 1 , vocabsize)
    pmis = self.get_pmi_matrix_from_outputs(outputs_all_batches, indices, padded_sentence_as_ids)
    return torch.tensor(pmis)

  def get_pmi_matrix_from_outputs(self, outputs, indices, sentence_as_ids):
    '''
    Gets pmi matrix from the outputs of xlnet
    Input: 
      outputs, tensor shape (2 * numwords * numsubwords, 1, vocabsize), 
        (for each PTB word, first the tensor for each subword numerator, 
        then for each subword denominator)
      indices, list of lists of subwords corresponding to each word
      sentence_as_ids, list of subword token ids in flattened sentence
    '''
    # this is a bit complicated looking but I'm just getting it so that
    # pad[i] = the number of items in the batch before the ith word's predictions
    lengths = [len(l) for l in indices]
    cumsum = np.empty(len(lengths)+1, dtype=int)
    np.cumsum(lengths, out=cumsum[1:])
    cumsum[:1] = 0
    pad = list(len(indices)*2*(cumsum))[:-1]

    pmis = np.ndarray(shape=(len(indices), len(indices)))
    for i, indices_i in enumerate(indices):
      for j, _ in enumerate(indices):
        start = pad[i] + j*2*len(indices_i)
        end = start + 2*len(indices_i)
        output_span = outputs[start:end]
        pmis[i][j] = self.get_pmi_from_outputs(output_span, indices_i, sentence_as_ids)
    return pmis

  @staticmethod
  def get_pmi_from_outputs(outputs, w1_indices, sentence_as_ids):
    '''
    Gets estimated PMI(w1;w2) from XLNet output by collecting and summing over the
    subword predictions, and subtracting the prediction without w2 from the prediction with w2
    '''
    len1 = len(w1_indices)
    outputs = outputs.reshape(2, len1, -1) # reshape to be tensor.Size[2,len1,vocabsize]
    log_numerator = sum(outputs[0][i][sentence_as_ids[w1_index]] for i, w1_index in enumerate(w1_indices))
    log_denominator = sum(outputs[1][i][sentence_as_ids[w1_index]] for i, w1_index in enumerate(w1_indices))
    pmi = (log_numerator - log_denominator)
    return pmi

  def make_mask_and_mapping(self, indices, padlens=(0, 2)):
    '''
    input:
      indices (list): a list with len(ptb_tokenlist) elements, each element being a list of subword indices. 
        (e.g. [[0, 1], [2], [3, 4]])
        so, the nth item is a list of indices in flattened subtoken list that correspond to nth ptb word
        ( e.g. the example above is for the ptb tokenlist ('Odds', 'and', 'Ends') 
          with corresponding subtokenlist ['▁Odd', 's', '▁and', '▁End', 's'] )
      padlens: tuple of ints, length of padding before, and after, resp.
        by default, no prepadding and postpadding only 2 special characters.
    output: perm_mask, target_mapping
      specification tensors for the whole sentence, for XLNet input;
        - predict each position in each ptb-word once per ptb-word in the sentence
          (except the special characters, need not be predicted)
        - to do this twice (once for numerator once for denominator)
      thus, batchsize of perm_mask and target_mapping will each be
        2 * len(indices) * len(flattened(indices))
    '''
    perm_masks = []
    target_mappings = []
    prepadlen, postpadlen = padlens
    # seqlen = number of items in indices flattened, plus padding lengths
    seqlen = prepadlen + len([i for x in indices for i in x]) + postpadlen
    # increment each index in nested list by prepadding amount prepad
    print(f'subword indices raw:\n{indices}')
    indices_incremented = [[i+prepadlen for i in l] for l in indices]
    print(f'incremented by prepadlen={prepadlen}:\n{indices_incremented}\n\\______\n')
    for word_i in indices_incremented:
      for word_j in indices_incremented:
        pm_ij, tm_ij = self.make_mask_and_mapping_single_pair(word_i, word_j, seqlen)
        perm_masks.append(pm_ij)
        target_mappings.append(tm_ij)
    perm_mask = torch.cat(perm_masks, 0)
    target_mapping = torch.cat(target_mappings, 0)

    return perm_mask, target_mapping

  @staticmethod
  def make_mask_and_mapping_single_pair(w1_indices, w2_indices, seqlen):
    '''
    Takes two lists of integers (representing the indices of the subtokens of two PTB tokens, resp.)
    and returns a permutation mask tensor and an target mapping tensor for use in XLNet.
    The first dimension (batch_dim) of each of these tensors will be twice the length of w1_indices.
    input:
      w1_indices, w2_indices: lists of indices (ints)
      seqlen = the length of the sentence as subtokens (with padding, special tokens)
    returns:
      perm_mask: a tensor of shape (2*len1, seqlen, seqlen)
      target_mapping: a tensor of shape (2*len1, 1, seqlen)
    '''
    len1 = len(w1_indices)
    batch_dim = len1*2 #  times 2 for numerator and denominator

    perm_mask = torch.zeros((batch_dim, seqlen, seqlen), dtype=torch.float)
    target_mapping = torch.zeros((batch_dim, 1, seqlen), dtype=torch.float)
    for i, index in enumerate(w1_indices):
      # the 1st and the len1+1th are the same, just for the numerator and denominator resp. 
      perm_mask[(i, len1+i), :, index:w1_indices[-1]+1] = 1.0 # mask the other w1 tokens to the right
      perm_mask[len1+i, :, w2_indices] = 1.0 # mask w2's indices for the denominator
      target_mapping[(i, len1+i), :, index] = 1.0 # predict just subtoken i of w1
    return perm_mask, target_mapping

  # Tokenization
  def make_subword_lists(self, ptb_tokenlist, add_special_tokens=False):
    '''
    Takes list of items from Penn Treebank tokenized text,
    runs the tokenizer to decompose into the subword tokens expected by XLNet,
    including appending special characters '<sep>' and '<cls>', if specified.
    Implements some simple custom adjustments to make the results more like what might be expected.
    [TODO: this could be improved, if it is important.
    For instance, currently it puts an extra space before opening quotes]
    Returns: a list with a sublist for each Treebank item
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
        if subword_list_i[0] == '': subword_list_i.pop(0)
      if subword_list_i[0] == '▁' and subword_list_i[1] in (')',']','}',',','.','"',"'","!","?") and i != 0:
        # print(f'{i}: removing extra space before character. {subword_list_i} => {subword_list_i[1:]}')
        subword_list_i.pop(0)
      if subword_list_i == ['▁', 'n', "'", 't'] and i != 0:
        # print(f"{i}: fixing X▁n't => Xn 't ")
        del subword_list_i[0]
        del subword_list_i[0]
        subword_lists[i-1][-1]+='n'
    return subword_lists

  @staticmethod
  def word_index_to_subword_indices(word_index, nested_list):
    '''
    Convert from word index (for nested list of subword tokens),
    to list of subword token indices at that word index
    ( e.g. for inputs word_index = 2, nested_list=[['▁Odd', 's'], ['▁and'], ['▁End', 's']]
      the output is [3, 4])
    '''
    if word_index > len(nested_list):
      raise ValueError('word_index exceeds length of nested_list')
    count = 0
    for subword_list in nested_list[:word_index]:
      count += len(subword_list)
    # maybe can do this with functools.reduce
    # count = reduce(lambda x, y: len(x) + len(y),nested_list[:word_index])
    return list(range(count, count+len(nested_list[word_index])))

  @staticmethod
  def make_chunks(m, n):
    '''
    makes (start,finish) tuples of indices 
    for use in indexing into a list of length m in chunks of size n
    
    input:
      m = number of items
      n = size of chunks
    returns: list of tuples of indices of form (start,finish)
    '''
    r = m%n
    d = m//n
    chunked = []
    for i in range(d):
      chunked.append((n*i, n*(i+1)))
    if r != 0:
      chunked.append((d*n, d*n+r))
    return chunked
