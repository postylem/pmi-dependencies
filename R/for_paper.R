library(tidyverse)
library(gridExtra)
theme_set(theme_minimal())
library(ggthemes)
library(scales)
recolor_plot <- function(plt,colors){
  plt + scale_color_manual(values=colors)
}

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

baseline_linear <- read_csv(Sys.glob("by_wordpair/wordpair_linear_baseline*.csv"))
baseline_random <- read_csv(Sys.glob("by_wordpair/wordpair_random_baseline_pad0_2021-01-26*.csv"))
baseline_random2 <- read_csv(Sys.glob("by_wordpair/wordpair_random_baseline_pad0_2021-01-27*.csv"))

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

baseline_linear$model <- "baseline_linear"
baseline_random$model <- "baseline_random"
baseline_random2$model <- "baseline_random2"

all_models_raw = list(dbert,bert_base,bert_large,xlnet_base,xlnet_large,xlm,bart,gpt2,w2v,lstm,onlstm,onlstm_syd,baseline_linear,baseline_random,baseline_random2)

make_dep_len_nopunct <- function(dataframe,verbose=TRUE) {
  # makes a dep_len feature, which records the distance ignoring tokens not in gold tree
  # that is, skipping over punctuation or other ignored symbols in calculating distance.
  if (verbose) {message(dataframe$model[[1]])}
  newdf = tibble()
  # dataframe = dataframe %>% filter(sentence_index %in% c(1,2,3,4))
  pb <- txtProgressBar(style = 3)
  for (s_index in unique(dataframe$sentence_index)) {
    # TODO: for loop is very slow. find another way.
    dfi = dataframe %>% filter(sentence_index==s_index)
    i1s = filter(dfi,gold_edge==T)$i1
    i2s = filter(dfi,gold_edge==T)$i2
    is = sort(union(i1s,i2s))
    dfi <- mutate(dfi, word_i1 = match(i1,is), word_i2 = match(i2,is), dep_len = word_i2-word_i1) %>%
      select(sentence_index, i1,i2,dep_len,word_i1,word_i2,dep_len,everything())
    newdf <- rbind(newdf,dfi)
    setTxtProgressBar(pb, s_index/max(unique(dataframe$sentence_index)))
  }
  close(pb)
  return(newdf)
}
# THIS TAKES AN HOUR:
all_models = lapply(all_models_raw, FUN=make_dep_len_nopunct)

dbert <- all_models[[1]]
bert <- all_models[[3]]
xlnet <- all_models[[4]]
xlm <- all_models[[6]]
bart <- all_models[[7]]
w2v <- all_models[[9]]
lstm <- all_models[[10]]
onlstm <- all_models[[11]]
onlstm_syd <- all_models[[12]]
baseline_linear <- all_models[[13]]
baseline_random <- all_models[[14]]


#### TABLE Overall accuracy scores avg by sentnce ####
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
  dataframe = mutate(dataframe, longdep=dep_len>1)
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
  dataframe = mutate(dataframe, longdep=dep_len>1)
  #' Prepare csv as df data grouped by 'longdep'
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
    pivot_wider(names_from = longdep, values_from = c(avg_precis,avg_recall), names_prefix = "longdep", values_fill = list(n = 0)) %>%
    add_column(model = dataframe$model[[1]])
}
avg_precis_recalls = do.call(bind_rows,c(lapply(all_models, binary_dist_avg_precis_recall))) %>% column_to_rownames("model")

# Combined table
avg_accuracy_table = cbind(all_avg_uuas_overall,avg_precis_recalls)

# write the table, with the columns in the right order
write.csv(avg_accuracy_table[,c(1,2,4,3,5)],"avg_accuracy_table.csv")

#### PLOTTING ####
#### Dep len histograms ####
prepare_df <- function(df){
  df <- df %>% mutate(acc=gold_edge==pmi_edge_sum)
  df$relation[is.na(df$relation)]<-"NONE"
  df <- df %>% prepare_POS() %>% add_class_predictor()
  return(df)
}


# quick histograms
gold.len <- bert %>% filter(gold_edge==T) %>% group_by(dep_len) %>% count
bert.len <- bert %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
xlnet.len <- xlnet %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
xlm.len <- xlm %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
# gpt2.len <- gpt2 %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
bart.len <- bart %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
dbert.len <- dbert %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
w2v.len <- w2v %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count
baseline_rand.len <- baseline_random %>% filter(pmi_edge_sum==T) %>% group_by(dep_len) %>% count

modelcols<-hue_pal()(6)

plothist<-function(df.len,fill,xlab,ylabel=T){
  yval = c(5, 10, 15, 20, 25)
  p<-df.len %>%  ggplot(aes(x=dep_len,y=n)) + geom_col(fill=fill,colour=fill) +
    scale_x_continuous(
      breaks = c(1,2,3,4,5,6,7,8,9),
      limits=c(0,10)
    ) +
    scale_y_continuous(labels = paste0(yval, "k"),
                       trans = "identity",
                       breaks = 10^3 * yval,
                       limits=c(0,26000)
    ) +
    theme(axis.title.y = element_blank()) +
    xlab(xlab)
  return(p)
}

hgold <- gold.len %>%  ggplot(aes(x=dep_len,y=n)) + geom_col()  + scale_x_continuous(breaks = c(1,2,3,4,5,6,7,8,9), limits=c(0,10)) + xlab("gold")
hgold <- plothist(gold.len,"black", "gold")
hbart <- plothist(bart.len,modelcols[[1]], "Bart")
hbert <- plothist(bert.len,modelcols[[2]], "BERT")
hdbert <-plothist(dbert.len,modelcols[[3]],"DistilBERT")
hw2v <- plothist(w2v.len,modelcols[[4]],"Word2Vec")
hxlm <-  plothist(xlm.len,modelcols[[5]],  "XLM")
hxlnet <-plothist(xlnet.len,modelcols[[6]],"XLNet")
hrand <- plothist(baseline_rand.len,"darkgrey","random")
# hgpt2 <- plothist(gpt2.len,modelcols[[7]], "GPT2")

p.lindisthist <- grid.arrange(arrangeGrob(hgold,heights=unit(0.5,"npc")),
             arrangeGrob(hbart, hbert,  hdbert,
                         hw2v,  hxlnet, hxlm,
                         nrow=2),
             nrow=1,widths=c(1,3),
             top="Dependency arc length histograms")
ggsave("plots/lindisthist-norand.pdf",plot=p.lindisthist, width = 5, height = 3, units = "in")

p.lindisthist.rnd <- grid.arrange(arrangeGrob(hgold, hrand,nrow=2),
             arrangeGrob(hbart, hbert,  hdbert,
                         hw2v,  hxlnet, hxlm,
                         nrow=2),
             nrow=1,widths=c(1,3),
             top="Dependency arc length histograms")

ggsave("plots/lindisthist.pdf",plot=p.lindisthist.rnd, width = 5, height = 3, units = "in")

## Proportions length 1

# sum(abs)
gold.len[1,]$n/(bert %>% filter(gold_edge==T) %>% count())[[1]]
# .4892236
bart.len[1,]$n/(xlm %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .5779846
bert.len[1,]$n/(bert %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .7519789
dbert.len[1,]$n/(dbert %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .6536717
xlm.len[1,]$n/(xlm %>% filter(pmi_edge_sum==T) %>% count())[[1]]
# .5220124
xlnet.len[1,]$n/(xlnet %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
# .5706027
# gpt2.len[1,]$n/(gpt2 %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
w2v.len[1,]$n/(w2v %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
# .4718508
baselin_rand.len[1,]$n/(baseline_random %>% filter(pmi_edge_sum==T) %>%  count())[[1]]
# .350123
#

#### Accuracy by dep len ####

prepare_by_len_gold <- function(dataframe){
  #' Prepare csv as df data grouped by 'dep_len'
  len = dataframe %>% filter(relation!="NONE") %>%
    group_by(dep_len) %>% summarise(meanpmi=mean(pmi_sum), varpmi=var(pmi_sum), n=n())
  dataframe = dataframe %>% filter(relation!="NONE") %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(dep_len,acc) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(len, by="dep_len") %>%
    mutate(pct_acc = pmiTRUE/n)
  return(dataframe)
}

xlnet.len.gold <- prepare_by_len_gold(xlnet)
bert.len.gold <-  prepare_by_len_gold(bert)
xlm.len.gold <-   prepare_by_len_gold(xlm)
bart.len.gold <-  prepare_by_len_gold(bart)
# gpt2.len.gold <-  prepare_by_len_gold(gpt2)
dbert.len.gold <- prepare_by_len_gold(dbert)
w2v.len.gold <-   prepare_by_len_gold(w2v)

lstm.len.gold <-   prepare_by_len_gold(lstm)
onlstm.len.gold <-   prepare_by_len_gold(onlstm)
onlstm_syd.len.gold <-   prepare_by_len_gold(onlstm_syd)

baseline_rand.len.gold <- prepare_by_len_gold(baseline_random)

# To get five models in one df
join_five <- function(df1, df2, df3, df4, df5,
                      #' to full_join five data frames
                      by=c("n","meanlen"),
                      suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".XLM")){
  return(
    full_join(df1,df2,by=by,suffix=suffixes[1:2]) %>%
      full_join(rename_at(df3, vars(-by), function(x){paste0(x,suffixes[3])}), by=by) %>%
      full_join(rename_at(df4, vars(-by), function(x){paste0(x,suffixes[4])}), by=by) %>%
      full_join(rename_at(df5, vars(-by), function(x){paste0(x,suffixes[5])}), by=by) %>%
      pivot_longer(cols = -by, names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)"))
}

# The five models in one df
five.len.gold <- join_five(dbert.len.gold,bart.len.gold,bert.len.gold,xlnet.len.gold,w2v.len.gold,
                           by = c("n","dep_len"),
                           suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))

five.len.gold.rnd <- join_five(dbert.len.gold,baseline_rand.len.gold,bert.len.gold,xlnet.len.gold,w2v.len.gold,
                               by = c("n","dep_len"),
                               suffixes=c(".DistilBERT",".random",".BERT",".XLNet",".Word2Vec"))

plotby_dep_len <- function(df, title="Accuracy (recall) by arc length", recoloring=NULL){
  p <- df %>% filter(n>25) %>%
    ggplot(aes(y=pct_acc, x=dep_len)) +
    geom_text(aes(label=n, y=Inf), hjust=0, size=2.5, colour="grey") +
    annotate("text",x=Inf,y=Inf, label="n", size=2.5, hjust=0, vjust=0, colour="grey") +
    geom_line(aes(group=dep_len), colour="grey") +
    geom_point(aes(size=n, colour=model), alpha=0.7) +
    coord_flip(clip = "off") +
    theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
          plot.margin = ggplot2::margin(0, 50, 0, 2, "pt"),
          legend.spacing = unit(0, 'cm')
    ) + scale_x_continuous(trans="identity",breaks = seq(1, 23, by = 2), minor_breaks = seq(1, 23, by = 1)) +
    ylab("recall (# CPMI arc = gold arc)/(# gold arcs)") +
    xlab("arc length") +
    ggtitle(title)
  if(is.null(recoloring)){
    return(p)
  } else {
    p <- p %>% recolor_plot(recoloring)
  }
}
p.dep_len <- plotby_dep_len(
  five.len.gold, title="Accuracy (recall) by arc length (n>25)")
ggsave("plots/acc_deplen.pdf",plot=p.dep_len, width = 5, height = 3.2, units = "in")

len.recolors <- hue_pal()(5)
len.recolors[[1]] <-modelcols[[2]]
len.recolors[[2]] <-modelcols[[3]]
len.recolors[[3]] <- "darkgrey"
len.recolors[[4]] <-modelcols[[4]]
len.recolors[[5]] <-modelcols[[6]]

p.dep_len.rnd <- plotby_dep_len(
  five.len.gold.rnd, title="Accuracy (recall) by arc length (n>25)",recoloring = len.recolors)
ggsave("plots/acc_deplen-rand.pdf",plot=p.dep_len.rnd, width = 5, height = 3.2, units = "in")

#### Accuracy by gold dep label ####


prepare_by_relation <- function(dataframe,length_greater_than=0){
  #' Prepare csv as df data grouped by 'relation'
  relation_len = dataframe %>% filter(gold_edge==T,
                                      dep_len>length_greater_than) %>%
    group_by(relation) %>% summarise(medlen=median(dep_len), meanlen=mean(dep_len), n=n(),
                                     meanpmi=mean(pmi_sum), varpmi=var(pmi_sum))
  dataframe = dataframe %>% filter(gold_edge==T,
                                   dep_len>length_greater_than) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(relation,acc) %>% summarise(n=n(), medlen=median(dep_len), meanlen=mean(dep_len)) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n,medlen,meanlen), values_fill = list(n = 0)) %>%
    left_join(relation_len, by="relation") %>% mutate(pct_acc = n_pmiTRUE/n)
  return(dataframe)
}

xlnet.relation <-prepare_by_relation(xlnet)
bert.relation <- prepare_by_relation(bert)
xlm.relation <-  prepare_by_relation(xlm)
bart.relation <- prepare_by_relation(bart)
dbert.relation <-prepare_by_relation(dbert)
# gpt2.relation <- prepare_by_relation(gpt2)
w2v.relation  <- prepare_by_relation(w2v)

lstm.relation  <- prepare_by_relation(lstm)
onlstm.relation  <- prepare_by_relation(onlstm)
onlstm_syd.relation  <- prepare_by_relation(onlstm_syd)

baseline_rand.relation <- prepare_by_relation(baseline_random)

five.relation <- join_five(dbert.relation,bart.relation,bert.relation,xlnet.relation,w2v.relation,
                           by=c("n","relation","meanlen"),
                           suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))

five.relation.rnd <- join_five(dbert.relation,baseline_rand.relation,bert.relation,xlnet.relation,w2v.relation,
                           by=c("n","relation","meanlen"),
                           suffixes=c(".DistilBERT",".random",".BERT",".XLNet",".Word2Vec"))

## same, only for arc-length ≥ 1

xlnet.relation.gt1<- prepare_by_relation(xlnet,length_greater_than = 1)
bert.relation.gt1 <- prepare_by_relation(bert,length_greater_than = 1)
xlm.relation.gt1 <-  prepare_by_relation(xlm,length_greater_than = 1)
bart.relation.gt1 <- prepare_by_relation(bart,length_greater_than = 1)
dbert.relation.gt1 <-prepare_by_relation(dbert,length_greater_than = 1)
# gpt2.relation.gt1 <- prepare_by_relation(gpt2,length_greater_than = 1)
w2v.relation.gt1 <-  prepare_by_relation(w2v, length_greater_than = 1)

lstm.relation.gt1        <- prepare_by_relation(lstm, length_greater_than = 1)
onlstm.relation.gt1      <- prepare_by_relation(onlstm, length_greater_than = 1)
onlstm_syd.relation.gt1  <- prepare_by_relation(onlstm_syd, length_greater_than = 1)

baseline_rand.relation.gt1 <- prepare_by_relation(baseline_random,length_greater_than = 1)

five.relation.gt1 <- join_five(dbert.relation.gt1,bart.relation.gt1,bert.relation.gt1,xlnet.relation.gt1,w2v.relation.gt1,
                               by=c("n","meanlen"),
                               suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))

five.relation.gt1.rnd <- join_five(dbert.relation.gt1,baseline_rand.relation.gt1,bert.relation.gt1,xlnet.relation.gt1,w2v.relation.gt1,
                               by=c("n","meanlen"),
                               suffixes=c(".DistilBERT",".random",".BERT",".XLNet",".Word2Vec"))


# A plot exploring accuracy by relation with respect to linear distance, model, and n
plotby_rel <- function(df, title="all arc lengths",ylabel=T,recoloring=NULL) {
  p <- df %>%  filter(n>60) %>%
    ggplot(aes(y=pct_acc, x=reorder(relation, meanlen))) +
    annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="grey") +
    geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="grey") +  # to print n
    annotate("text",x=Inf,y=-Inf, label="mean arclength", size=3, hjust=0, vjust=0) +
    geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3, alpha=0.2) +
    geom_line(aes(group=relation), colour="grey") +
    geom_point(aes(size=n, colour=model), alpha=0.6) +
    coord_flip(clip = "off") +
    theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
          plot.margin = ggplot2::margin(0, 50, 2, 2, "pt"),
          axis.ticks = element_blank()) +
    ylab(NULL) +
    ggtitle(title) +
    theme(plot.title = element_text(hjust = 0.5),
          legend.spacing = unit(0, 'cm'))
  if (ylabel) {p <- p + xlab("recall (# CPMI arc = gold arc)/(# gold arcs)")}
  else p <- p + xlab(NULL)
  if(is.null(recoloring)){
    return(p)
  } else {
    p <- p %>% recolor_plot(recoloring)
  }
}

#extract legend
#https://github.com/hadley/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

hbart <- plothist(bart.len,modelcols[[1]], "Bart")
hbert <- plothist(bert.len,modelcols[[2]], "BERT")
hdbert <-plothist(dbert.len,modelcols[[3]],"DistilBERT")
hw2v <- plothist(w2v.len,modelcols[[4]],"Word2Vec")
hxlm <-  plothist(xlm.len,modelcols[[5]],  "XLM")
hxlnet <-plothist(xlnet.len,modelcols[[6]],"XLNet")

rel.recolors <- hue_pal()(5)
rel.recolors[[1]] <-modelcols[[1]]
rel.recolors[[2]] <-modelcols[[2]]
rel.recolors[[3]] <-modelcols[[3]]
rel.recolors[[4]] <-modelcols[[4]]
rel.recolors[[5]] <-modelcols[[6]]

p.rel <- plotby_rel(five.relation, title="all arc lengths",recoloring = rel.recolors)
p.rel.gt1<- plotby_rel(five.relation.gt1, title="arc length > 1",ylabel=F,recoloring = rel.recolors)
p.rel.both <- grid.arrange(
  g_legend(p.rel.gt1),
  arrangeGrob(
    p.rel + theme(legend.position="none"),
    p.rel.gt1 + theme(legend.position="none"),
    nrow=1),
  nrow=2,heights=c(15, 100),
  top="Accuracy (recall) by gold label (only labels with n>60)",
  bottom="gold dependency label (ordered by mean arc length)")
ggsave("plots/acc_label.pdf",plot=p.rel.both, width = 9.7, height = 4.75, units = "in")


rel.recolors.rnd <- rel.recolors
rel.recolors.rnd[[1]] <- modelcols[[2]]
rel.recolors.rnd[[2]] <- modelcols[[3]]
rel.recolors.rnd[[3]] <- "darkgrey"

p.rel.rnd <- plotby_rel(five.relation.rnd, title="all arc lengths", recoloring = rel.recolors.rnd)
p.rel.gt1.rnd<- plotby_rel(five.relation.gt1.rnd, title="arc length > 1",ylabel=F, recoloring = rel.recolors.rnd)
p.rel.both.rnd <- grid.arrange(
  g_legend(p.rel.gt1.rnd),
  arrangeGrob(
    p.rel.rnd + theme(legend.position="none"),
    p.rel.gt1.rnd + theme(legend.position="none"),
    nrow=1),
  nrow=2,heights=c(15, 100),
  top="Accuracy (recall) by gold label (only labels with n>60)",
  bottom="gold dependency label (ordered by mean arc length)")
ggsave("plots/acc_label-rand.pdf",plot=p.rel.both.rnd, width = 9.7, height = 4.75, units = "in")
