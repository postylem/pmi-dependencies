# comp551-miniproj4
Exploring Hewitt and Manning's 2019 NAACL paper.
[original repo here](https://github.com/john-hewitt/structural-probes)

# Reproducing our results

## Running a probe

### 1. Setup dataset
To reproduce our experiments playing with and H&M's probe, and combining layers, 
- get access to the dataset **Treebank-3** (proprietary, [from the LDC](https://catalog.ldc.upenn.edu/LDC99T42)). 
- install Stanford [CORENLP](https://stanfordnlp.github.io/CoreNLP/) locally

Use script [scripts/convert_splits_to_depparse.sh](scripts/convert_splits_to_depparse.sh) (set the correct filepath to your Treebank WSJ parsed files (extension .MRG) and location of CORENLP) to generate dependency parses from the constituency trees for the full-PTB dataset. Use [scripts/convert_minisplits_to_depparse.sh](scripts/convert_minisplits_to_depparse.sh) to generate our mini-PTB dataset.  After this is done, you will not need the original Treebank files anymore.

Use [scripts/convert_raw_to_BERT_vectors.sh](scripts/convert_raw_to_BERT_vectors.sh) to run BERT on the .txt files in order to get embeddings to probe (these will be big files, especially the train split, of course: for the full-PTB, on the order of 200GB total for dev,test,train.  For small-PTB, 15GB).

### 2. Set config file
Write a config file for the test you want to run. The config files for all our tests on BERT-large are saved in [cluster_configs.tar.gz](cluster_configs.tar.gz).  The options for config file are mostly as described in [Hewitt's readme](hewitt-repo/README.md).  We have added some options though:

#### Extra config options:
- under `model`, there are the additional parameters `concat_with_layer` and `concat_with_another_layer`: adding a second and third layer to probe at the same time (just concatenating). Also the probe rank multiplied by 2 (3) for convenience (so just leave it as 1024 for BERT-large, and the probe will be full rank). Be sure to have `all_layers: False`.
- to probe *all* layers concatenated, set `all_layers: True` (this will also multiply the probe rank by the model number of layers (12 for BERT-base, 24 for BERT-large)).  In this case the previous options will be ignored, and should be absent for clarity.

Example: probing layers 16+13+18:

```
model:
  # ...
  all_layers: False 			# set to true to concatenate all layers
  model_layer: 16				# layer to probe
  concat_with_layer: 13 		# optional additional layer to concatenate
  concat_with_another_layer: 18 # optional third layer to concatenate
```

Note: layers are numbered with 0-indexing in all of the code, but referenced using 1-indexing in the paper (so choose `model_layer: 15` to probe the 16th layer, as described in the paper).

To run a deep probe:
- under `probe:` set `deep_probe: True`

```
probe:
  task_signature: word_pair
  task_name: parse-distance
  deep_probe: True # set as True to use deep probe (rather than linear)
  # ...
```

### 3. Run experiment
Having chosen config file `<config>.yaml`, run a probe by running Hewitt's `run_experiment.py`.  From a terminal in the root of this repo (with python v3), run
```
python hewitt-repo/structural-probes/run_experiment.py configs/<config>.yaml
```


## Other parts
Our scripts for comparing the layers are in the scripts directory. 
- Use [scripts/get_uuas.py](scripts/get_uuas.py) to get the UUAS measure of similarity of layers (which uses a modified version of Hewitt's reporter.py).
- Use [scripts/tree_dist.py](scripts/tree_dist.py) to get the L2 distance between layers' predicted matrices.

#### Visualization
Use [scripts/visualize_layer_dists.py](scripts/visualize_layer_dists.py) to generate heatmap distance matrices.

# Some results summarized.

### Below are some of our results.

UUAS for BERT-base all 12-layers concatenated, and BERT-large multiple consecutive layers

layers | uuas | batch size (if not 20)
--- | --- | ---
base: all 12 | 0.8645431336460251 | 15
large: 13:19 | 0.8721724158404085 | 4
large: 12:20 | 0.876447188743098  | 1
large: 10:20 | 0.8764175028201627 | 1

UUAS for a few specific three-layer linear probes on BERT-large

layers | uuas
--- | ---
16+15+13 | 0.8612776821231372
16+17+13 | 0.8589621801341804
16+14+13 | 0.8612479962002019
16+14+15 | 0.8589918660571157
16+14+17 | 0.8561123315323873
16+15+17 | 0.8503532624829306


UUAS for all possible layers concatenated with layer 16

layer | linear probe uuas |
--- | 	:--- 	|
00+16	|	0.8359555898592888	|
01+16	|	0.8337291456391379	|
02+16	|	0.8329573116428189	|
03+16	|	0.8381820340794395	|
04+16	|	0.8397553879950128	|
05+16	|	0.8450988541233747	|
06+16	|	0.84685032357656	|
07+16	|	0.8489580241049695	|
08+16	|	0.8483049338003918	|
09+16	|	0.8464644065784005	|
10+16	|	0.8498486017930298	|
11+16	|	0.8513328979397969	|
12+16	|	0.8516594430920857	|
13+16	|	0.8523125333966634	|
14+16	|	0.8488392804132281	|
15+16	|	0.8458706881196937	|
16+16	|	0.8211126283916167	|
17+16	|	0.8324823368758535	|
18+16	|	0.8318589324942113	|
19+16	|	0.8314136436501811	|
20+16	|	0.8320073621088879	|
21+16	|	0.8317401888024699	|
22+16	|	0.830285578578638	|
23+16	|	0.8310277266520216	|


UUAS for our runs on our mini dataset (roughly 1/10 of the PTB), with BERT-base

layer | linear probe uuas | 2-layer probe uuas
--- | --- | ---
5	|					|0.7971
6   | 0.7503	        |0.8089
7   | 0.7500			|0.8097
8   | 0.7440, 0.7461	|0.8000
9   | 0.7171			|0.7759
10  | 0.6995            |0.7609, 0.7673
11	|					|0.7362

UUAS for our runs on the entire dataset (with BERT-large)

layer | linear probe uuas | 2-layer probe uuas
--- | 	:--- 	| :---								
00	|	0.5542	|	0.6704			
01	|	0.5658	|	0.7010			  
02	|	0.5628	|	0.6976			
03	|	0.6228	|	0.7743			
04	|	0.6436	|	0.7847			
05	|	0.6941	|	0.8126			  
06	|	0.7056	|	0.8165
07	|	0.7012	|	0.8216
08	|	0.6976	|	0.8159			  
09	|	0.7076	|	0.8214			  
10	|	0.7384	|	0.8287			  
11	|	0.7691	|	0.8413			  
12	|	0.7908	|	0.8623			
13	|	0.8117	|	0.8705			  
14	|	0.8229	|	0.8719			
15	|	0.8280	|	0.8786				
16	|	0.8216	|	0.8704			
17	|	0.8064	|	0.8580			
18	|	0.7917	|	0.8428			
19	|	0.7615	|	0.8259			  
20	|	0.7095	|	0.7949			
21	|	0.6871	|	0.7701			
22	|	0.6639	|	0.7530			
23	|	0.6549	|	0.7349


