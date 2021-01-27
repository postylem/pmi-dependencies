library(tidyverse)
theme_set(theme_minimal())

#### LOAD MODELS ####
bert <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=bert-large-cased_pad60*.csv"))
bert_base <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=bert-base-cased*.csv"))
bert_large <- bert
xlnet<- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=xlnet-base-cased_pad30*.csv"))
xlnet_base <- xlnet
xlnet_large <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=xlnet-large-cased*.csv"))
xlm  <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=xlm-mlm-en-2048_pad60*.csv"))
bart <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=bart-large_pad60*.csv"))
gpt2 <- read_csv(Sys.glob("by_wordpair/wordpair_gpt2_pad30*.csv"))
dbert<- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=distilbert-base-cased_pad60*.csv"))
w2v  <- read_csv(Sys.glob("by_wordpair/wordpair_w2v_pad0*.csv"))

lstm  <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=lstm*.csv"))
onlstm <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=onlstm_pad*.csv"))
onlstm_syd <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=onlstm_syd*.csv"))

dbert$model <- "DistilBERT"
xlnet$model <- "XLNet"
xlnet_large$model <- "XLNet-large"
xlnet_base$model <- "XLNet-base"
bert$model  <- "BERT"
bert_base$model  <- "BERT-base"
bert_large$model  <- "BERT-large"
xlm$model   <- "XLM"
bart$model  <- "Bart"
gpt2$model  <- "GPT2"
w2v$model   <- "Word2Vec"

lstm$model <- "LSTM"
onlstm$model <- "ONLSTM"
onlstm_syd$model <- "ONLSTM-SYD"

all_models = list(dbert,bert_base,bert_large,xlnet_base,xlnet_large,xlm,bart,gpt2,w2v,lstm,onlstm,onlstm_syd)

# the avg_recall of the linear model is just the avg
# number of len1 / number of len1 edges in sentence
# =1

# avg_precision for linear model is average of number of len1 / number of edges in sentence
# that is, the
# average proportion adjacent for gold edges, which is .49
avg_proportion_adjacent = bert %>% mutate(shortdep=lin_dist==1,longdep=!shortdep) %>% filter(gold_edge==T) %>%
  group_by(sentence_index,longdep) %>% summarise(n=n()) %>%
  pivot_wider(names_from = longdep, values_from = c(n), names_prefix = "n_long", values_fill = list(n = 0)) %>%
  ungroup() %>% mutate(n_edges=n_longFALSE+n_longTRUE, proportion_adjacent=n_longFALSE/n_edges) %>%
  summarise(avg_prop_adj=mean(proportion_adjacent)) %>%
  add_column(model = dataframe$model[[1]]) %>% column_to_rownames("model")

# Overall accuracy scores
#### A version averaged by sentence ####
avg_uuas <- function(dataframe){
  n_edges = dataframe %>% filter(gold_edge==T) %>% group_by(sentence_index) %>% summarise(n=n()) # total number of edges
  acc_df = dataframe %>% filter(pmi_edge_sum==T,gold_edge==T)  %>% group_by(sentence_index) %>% summarise(n_acc=n()) %>%
    left_join(n_edges,by="sentence_index") %>% mutate(uuas=n_acc/n) %>% summarise(avg_uuas=mean(uuas))
  uuas = acc_df["avg_uuas"][[1]]
  out_df = uuas %>% as.data.frame(row.names=dataframe$model[[1]])
  colnames(out_df) = "avg_uuas"
  return(out_df)
}
all_avg_uuas_overall = do.call(rbind,c(lapply(all_models,avg_uuas)))
binary_dist_avg_precis <- function(dataframe){
  dataframe = mutate(dataframe, longdep=lin_dist>1)
  #' Prepare csv as df data grouped by 'longdep'
  n_pmi_edges = dataframe %>% filter(pmi_edge_sum==T) %>% group_by(longdep,sentence_index) %>% summarise(n=n())
  precis_df = dataframe %>% filter(pmi_edge_sum==T) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(longdep,acc,sentence_index) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "n_pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(n_pmi_edges, by=c("longdep","sentence_index")) %>%
    mutate(precis = n_pmiTRUE/n)
  out_df = precis_df %>% summarise(avg_precis=mean(precis), n_pmiFALSE=sum(n_pmiFALSE),n_pmiTRUE=sum(n_pmiTRUE), n=sum(n))
  return(out_df)
}
binary_dist_avg_recall <- function(dataframe){
  dataframe = mutate(dataframe, longdep=lin_dist>1)
  #' Prepare csv as df data grouped by 'lin_dist'
  n_gold_edges = dataframe %>% filter(relation!="NONE") %>% group_by(longdep,sentence_index) %>% summarise(n=n())
  recall_df = dataframe %>% filter(relation!="NONE") %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(longdep,acc,sentence_index) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "n_pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(n_gold_edges, by=c("longdep","sentence_index")) %>%
    mutate(recall = n_pmiTRUE/n)
  out_df = recall_df %>% summarise(avg_recall=mean(recall), n_pmiFALSE=sum(n_pmiFALSE),n_pmiTRUE=sum(n_pmiTRUE), n=sum(n))
  return(out_df)
}
binary_dist_avg_precis_recall <- function(dataframe){
  df = left_join(binary_dist_avg_precis(dataframe), binary_dist_avg_recall(dataframe), by=c("longdep"))
  df %>% select(c("longdep","avg_precis","avg_recall")) %>%
    pivot_wider(names_from = longdep, values_from = c(avg_precis,avg_recall), names_prefix = "longdep") %>%
    add_column(model = dataframe$model[[1]]) %>% column_to_rownames("model")
}
avg_precis_recalls = do.call(rbind,c(lapply(all_models, binary_dist_avg_precis_recall)))
# Combined table
avg_accuracy_table = cbind(all_avg_uuas_overall,avg_precis_recalls)

#### A version averaged over all edges ####
uuas_overall <- function(dataframe){
  n_edges = nrow(dataframe %>% filter(gold_edge==T)) # total number of edges
  acc_df = dataframe %>% filter(pmi_edge_sum==T,gold_edge==T) %>% summarise(n_acc=n())
  n_acc = acc_df["n_acc"][[1]]
  uuas = n_acc/n_edges
  out_df = uuas %>% as.data.frame(row.names=dataframe$model[[1]])
  colnames(out_df) = "uuas"
  return(out_df)
}
all_uuas_overall = do.call(rbind,c(lapply(all_models,uuas_overall)))
# Getting a precision and recall score grouped by whether lin_dist =1 or >1
binary_dist_precis <- function(dataframe){
  dataframe = mutate(dataframe, longdep=lin_dist>1)
  #' Prepare csv as df data grouped by 'longdep'
  precis_len = dataframe %>% filter(pmi_edge_sum==T) %>% group_by(longdep) %>% summarise(n=n())
  precis_df = dataframe %>% filter(pmi_edge_sum==T) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(longdep,acc) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "n_pmi", values_from = c(n), values_fill = list(n = 0))
  out_df = left_join(precis_df, precis_len, by="longdep") %>%
    mutate(precis = n_pmiTRUE/n)
  return(out_df)
}
binary_dist_recall <- function(dataframe){
  dataframe = mutate(dataframe, longdep=lin_dist>1)
  #' Prepare csv as df data grouped by 'lin_dist'
  recall_len = dataframe %>% filter(relation!="NONE") %>% group_by(longdep) %>% summarise(n=n())
  recall_df = dataframe %>% filter(relation!="NONE") %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(longdep,acc) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "n_pmi", values_from = c(n), values_fill = list(n = 0))
  out_df = left_join(recall_df, recall_len, by="longdep") %>%
    mutate(recall = n_pmiTRUE/n)
  return(out_df)
}
binary_dist_precis_recall <- function(dataframe){
  df = left_join(binary_dist_precis(dataframe), binary_dist_recall(dataframe), by=c("longdep"))
  df %>% select(c("longdep","precis","recall")) %>%
    pivot_wider(names_from = longdep, values_from = c(precis,recall), names_prefix = "longdep") %>%
    add_column(model = dataframe$model[[1]]) %>% column_to_rownames("model")
}
precis_recalls = do.call(rbind,c(lapply(all_models, binary_dist_precis_recall)))
# Combined table
accuracy_table = cbind(all_uuas_overall,precis_recalls)

# We get the following:
# model           avg_uuas avg_precis_len=1 avg_recall_len=1 avg_precis_len>1 avg_recall_len>1
# linear_baseline .50     .49               1.               --               0
# DistilBERT      .51     .59               .77              .36              .25
# BERT-base       .50     .60               .79              .32              .22
# BERT-large      .50     .57               .86              .27              .14
# XLNet-base      .48     .62               .70              .32              .27
# XLNet-large     .43     .64               .64              .25              .24
# XLM             .46     .65               .68              .27              .25
# Bart            .42     .58               .65              .23              .19
# GPT2            .38     .57               .40              .29              .37
# Word2Vec        .40     .63               .60              .21              .22
# LSTM            .46     .57               .74              .27              .21
# ONLSTM          .47     .58               .74              .28              .21
# ONLSTM-SYD      .47     .58               .74              .28              .21



#### POS preliminary ####

pos_bert_base <- read_csv(Sys.glob("../results-clean/pos-cpmi/simple_probe/xpos_bert-base-cased*/loaded*/wordpair_*.csv"))
pos_bert_large <- read_csv(Sys.glob("../results-clean/pos-cpmi/simple_probe/xpos_bert-large-cased*/loaded*/wordpair_*.csv"))
pos_xlnet_base <- read_csv(Sys.glob("../results-clean/pos-cpmi/simple_probe/xpos_xlnet-base-cased*/loaded*/wordpair_*.csv"))
pos_xlnet_large <- read_csv(Sys.glob("../results-clean/pos-cpmi/simple_probe/xpos_xlnet-large-cased*/loaded*/wordpair_*.csv"))

pos_bert_base$model <- "XPOS_BERTbase"
pos_bert_large$model <- "XPOS_BERTlarge"
pos_xlnet_base$model <- "XPOS_XLNetbase"
pos_xlnet_large$model <- "XPOS_XLNetlarge"

pos_models = list(pos_bert_base, pos_bert_large, pos_xlnet_base, pos_xlnet_large)
pos_all_avg_uuas_overall = do.call(rbind,c(lapply(pos_models,avg_uuas)))
pos_avg_precis_recalls = do.call(rbind,c(lapply(pos_models, binary_dist_avg_precis_recall)))
pos_avg_accuracy_table = cbind(pos_all_avg_uuas_overall,pos_avg_precis_recalls)
print(pos_avg_accuracy_table, digits = 2)
