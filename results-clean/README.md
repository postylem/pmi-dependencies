# Some results

CPMI accuracy results and saved matrices for the models reported are included in this directory.

### baselines
linear : **0.50**

random non-proj:   0.13

rndom projective: 0.27


## Testing

### xlnet-base-cased
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.43  |  0.41  |  0.38  |  0.43  |
|  proj    |  0.46  |  0.44  |  0.41  |  0.43  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.44  |  0.42  |  0.39  |  0.44  |
|  proj    |  0.47  |  0.45  |  0.43  |  0.44  |

| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.42  |  0.40  |  0.39  |  0.43  |
|  proj    |  0.46  |  0.44  |  0.44  |  0.44  |
### xlnet-large-cased
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.38  |  0.35  |  0.33  |  0.37  |
|  proj    |  0.42  |  0.40  |  0.38  |  0.39  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.39  |  0.36  |  0.36  |  0.39  |
|  proj    |  0.43  |  0.41  |  0.40  |  0.41  |

| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.38  |  0.36  |  0.37  |  0.39  |
|  proj    |  0.43  |  0.41  |  0.41  |  0.41  |
### bert-base-uncased
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.44  |  0.43  |  0.42  |  0.45  |
|  proj    |  0.46  |  0.45  |  0.44  |  0.44  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.45  |  0.44  |  0.42  |  0.46  |
|  proj    |  0.46  |  0.46  |  0.44  |  0.44  |
### bert-base-cased
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.45  |  0.44  |  0.43  |  0.46  |
|  proj    |  0.47  |  0.46  |  0.45  |  0.45  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.46  |  0.44  |  0.44  |  0.47  |
|  proj    |  0.47  |  0.46  |  0.46  |  0.46  |

| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.46  |  0.44  |  0.44  |  0.47  |
|  proj    |  0.47  |  0.46  |  0.46  |  0.46  |
### bert-large-uncased
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.43  |  0.41  |  0.40  |  0.44  |
|  proj    |  0.45  |  0.44  |  0.43  |  0.43  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.44  |  0.43  |  0.42  |  0.45  |
|  proj    |  0.46  |  0.45  |  0.44  |  0.44  |
### bert-large-cased
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.45  |  0.45  |  0.42  |  0.46  |
|  proj    |  0.46  |  0.46  |  0.44  |  0.45  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.47  |  0.47  |  0.43  |**0.48**|
|  proj    |**0.48**|**0.48**|  0.45  |  0.45  |

| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |**0.48**|  0.47  |  0.44  |**0.48**|
|  proj    |**0.48**|**0.48**|  0.45  |  0.45  |
### xlm-mlm-en-2048
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.39  |  0.36  |  0.37  |  0.39  |
|  proj    |  0.43  |  0.41  |  0.41  |  0.42  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.40  |  0.38  |  0.38  |  0.40  |
|  proj    |  0.43  |  0.42  |  0.42  |  0.42  |

| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.41  |  0.39  |  0.39  |  0.41  |
|  proj    |  0.44  |  0.43  |  0.42  |  0.42  |
### bart-large
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.37  |  0.24  |  0.39  |  0.37  |
|  proj    |  0.38  |  0.28  |  0.40  |  0.38  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.38  |  0.23  |  0.40  |  0.38  |
|  proj    |  0.39  |  0.27  |  0.41  |  0.40  |

| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.38  |  0.23  |  0.41  |  0.38  |
|  proj    |  0.40  |  0.27  |  0.42  |  0.40  |
### distilbert-base-cased
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.46  |  0.44  |  0.43  |  0.46  |
|  proj    |  0.48  |  0.46  |  0.46  |  0.46  |

| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.47  |  0.45  |  0.45  |  0.47  |
|  proj    |  0.48  |  0.47  |  0.46  |  0.47  |

| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.48  |  0.46  |  0.45  |  0.48  |
|  proj    |**0.49**|  0.48  |  0.47  |  0.47  |

### gpt2  (just for exploration)
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.43  |  0.09  |  0.43  |  0.37  |
|  proj    |  0.43  |**0.51**|  0.43  |  0.44  |

### Word2Vec trained on wikipedia and bookcorpus
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.305 |  0.323 |  0.254 |  0.287 |
|  proj    |  0.411 |  0.399 |  0.323 |  0.419 |
### Word2Vec `gensim` implementation
| *pad 0*  |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.308 |  0.291 |  0.314 |  0.307 |
|  proj    |  0.408 |  0.376 |  0.401 |  0.441 |
--------------------------------------------------


## Checkpoints along training (BERT), 500 sentences

- linear         0.499
- random nonproj 0.135
- random proj    0.267

### bert-base-uncased pad60 off-shelf
| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.452 |  0.434 |  0.432 |  0.462 |
|  proj    |  0.463 |  0.452 |  0.448 |  0.442 |
### bert-base-uncased pad60 1500000
| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.415 |  0.404 |  0.388 |  0.424 |
|  proj    |  0.434 |  0.423 |  0.41  |  0.413 |
### bert-base-uncased pad60 1000000
| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.419 |  0.405 |  0.399 |  0.437 |
|  proj    |  0.439 |  0.426 |  0.425 |  0.431 |
### bert-base-uncased pad60 500000
| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.376 |  0.339 |  0.351 |  0.375 |
|  proj    |  0.409 |  0.372 |  0.389 |  0.395 |
### bert-base-uncased pad60 100000
| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.431 |  0.409 |  0.42  |  0.447 |
|  proj    |  0.444 |  0.425 |  0.437 |  0.446 |
### bert-base-uncased pad60 50000
| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.151 |  0.148 |  0.152 |  0.149 |
|  proj    |  0.27  |  0.261 |  0.256 |  0.265 |
### bert-base-uncased pad60 10000
| *pad 60* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.115 |  0.129 |  0.095 |  0.104 |
|  proj    |  0.244 |  0.21  |  0.165 |  0.178 |

And, with absolute value

### bert-base-uncased pad30 off-shelf
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.450 |  0.436 |  0.424 |  0.461 |
|  proj    |  0.464 |  0.459 |  0.443 |  0.442 |
### bert-base-uncased pad30 1500000
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.396 |  0.357 |  0.374 |  0.377 |
|  proj    |  0.423 |  0.389 |  0.406 |  0.385 |
### bert-base-uncased pad30 1000000
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.432 |  0.388 |  0.403 |  0.418 |
|  proj    |  0.455 |  0.417 |  0.430 |  0.420 |
### bert-base-uncased pad30 500000
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.335 |  0.289 |  0.313 |  0.310 |
|  proj    |  0.377 |  0.341 |  0.362 |  0.342 |
### bert-base-uncased pad30 100000
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.460 |  0.432 |  0.430 |  0.457 |
|  proj    |  0.475 |  0.453 |  0.450 |  0.462 |
### bert-base-uncased pad30 50000
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.131 |  0.127 |  0.135 |  0.130 |
|  proj    |  0.224 |  0.228 |  0.233 |  0.246 |
### bert-base-uncased pad30 10000
| *pad 30* |  sum   |  triu  |  tril  |  none  |
|  ------  |  ----  |  ----  |  ----  |  ----  |
|  nonproj |  0.098 |  0.105 |  0.129 |  0.101 |
|  proj    |  0.143 |  0.166 |  0.198 |  0.188 |

------------------------------------------------

## How to treat negative CPMI values?

PMI may be positive or negative.  A positive value means that the two outcomes are more likely together than they are individually (the joint probability of the outcomes is higher than the product of their marginal probabilities).  A negative value means the opposite: the outcomes are less likely to appear jointly than they are individually.  PMI=0 means the outcomes are independent of each other.

Under the predictability-dependency hypothesis, a high PMI value should be correlated with dependency, and pairs of words with low absolute value PMI would be less likely to be syntactically dependent. It isn't clear what the prediction would be for word-pairs with large negative PMI values.

So far, the working assumption was that these should be treated as being anticorrelated with dependency.  However, it's worth testing what happens if we take the absolute value of the CPMI, before we extract dependency structures.

### bert-large-cased_pad30

*no absolute value:*
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.469  |  0.465  |  0.432  |**0.479**|
|  proj     |**0.477**|**0.477**|**0.447**|  0.448  |

*absolute value after symmetrizing:*
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.482  |  0.480  |  0.447  |  0.485  |
|  proj     |**0.492**|  0.493  |  0.461  |  0.458  |

*absolute value before symmetrizing:*
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.493  |  0.480  |  0.447  |  0.485  |
|  proj     |**0.499**|  0.493  |  0.461  |  0.458  |


Okay, so it's actually a bit better when we use the absolute value of the CPMI.  What about if we *prioritize* the negative CPMI values?  Accuracy suffers catastrophically.  
(TODO: Confirm this)

|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.122  |  0.119  |  0.118  |  0.106  |
|  proj     |  0.133  |  0.120  |  0.121  |  0.109  |

### testing with more models:

Okay, so taking the absolute value of CPMI (before symmetrizing), the results across models look like:


#### bart-large 60             
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.404  |  0.203  |  0.426  |  0.398  |
|  proj     |  0.419  |  0.248  |  0.440  |  0.426  |

#### bert-base-cased 30       
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.493  |  0.462  |  0.457  |  0.486  |
|  proj     |  0.504  |  0.482  |  0.475  |  0.474  |

#### bert-base-uncased 30     
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.48,  |  0.458  |  0.447  |  0.474  |
|  proj     |  0.493  |  0.479  |  0.466  |  0.462  |

#### bert-large-cased 60      
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.496  |  0.487  |  0.448  |  0.489  |
|  proj     |  0.501  |  0.497  |  0.461  |  0.454  |

#### bert-large-uncased 30    
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.464  |  0.437  |  0.424  |  0.454  |
|  proj     |  0.477  |  0.456  |  0.444  |  0.439  |

#### distilbert-base-cased 60 
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.498  |  0.476  |  0.464  |  0.492  |
|  proj     |  0.511  |  0.493  |  0.481  |  0.483  |

#### w2v 0                   
|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.094  |  0.121  |  0.091  |  0.097  |
|  proj     |  0.145  |  0.176  |  0.146  |  0.163  |

#### xlm-mlm-en-2048 60       

|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.434  |  0.405  |  0.400  |  0.422  |
|  proj     |  0.465  |  0.437  |  0.434  |  0.435  |

#### xlnet-base-cased 30      

|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.457  |  0.421  |  0.402  |  0.446  |
|  proj     |  0.485  |  0.457  |  0.440  |  0.446  |

#### xlnet-large-cased 30     

|           |  sum    |  triu   |  tril   |  none   |
|  ------   |  ----   |  ----   |  ----   |  ----   |
|  nonproj  |  0.399  |  0.359  |  0.367  |  0.384  |
|  proj     |  0.437  |  0.405  |  0.409  |  0.408  |



-----------------------

#### bart-large 60             
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.404  |  0.384  |
|  proj     |  0.419  |  0.390  |

#### bert-base-cased 30       
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.493  |  0.461  |
|  proj     |  0.504  |  0.472  |

#### bert-base-uncased 30     
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.48,  |  0.450  |
|  proj     |  0.493  |  0.464  |

#### bert-large-cased 60      
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.496  |  0.475  |
|  proj     |  0.501  |  0.482  |

#### bert-large-uncased 30    
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.464  |  0.442  |
|  proj     |  0.477  |  0.458  |

#### distilbert-base-cased 60 
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.498  |  0.475  |
|  proj     |  0.511  |  0.489  |

#### w2v 0                   
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.094  |  0.308  |
|  proj     |  0.145  |  0.408  |

#### xlm-mlm-en-2048 60       
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.434  |  0.407  |
|  proj     |  0.465  |  0.441  |

#### xlnet-base-cased 30      
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.457  |  0.439  |
|  proj     |  0.485  |  0.468  |

#### xlnet-large-cased 30     
|           | sum(abs)|  sum    |
|  ------   |  ----   |  ----   |
|  nonproj  |  0.399  |  0.386  |
|  proj     |  0.437  |  0.430  |

