library(tidyverse)
library(gridExtra)
theme_set(theme_minimal())
library(ggthemes)
#scale_colour_discrete <- scale_colour_colorblind
#setwd("/Users/j/McGill/PhD-miasma/pmi-dependencies/R")


## POS ###################################

make_simple_pos <- function(XPOS){
  return(
    case_when(
      XPOS %in% c("JJ", "JJR", "JJS")               ~ "Adjective",
      XPOS %in% c("NN", "NNS", "NNPS", "NNP")       ~ "Noun",
      XPOS %in% c("RB", "RBR", "RBS")               ~ "Adverb",
      XPOS %in% c("VB", "VBZ", "VBN", "VBG", "VBP", "VBD") ~ "Verb",
      XPOS %in% c("PRP$","WP$")                     ~ "Possessive",
      XPOS %in% c("IN","TO")                        ~ "Preposition", # needs work... TO isn't just preposition
      XPOS %in% c("DT")                             ~ "Determiner",
      XPOS %in% c("CD")                             ~ "Cardinal",
      XPOS %in% c("WDT")                            ~ "Complementizer",
      # ADD CLAUSAL COMPLEMENTIZERS
      TRUE ~ as.character(XPOS)))
}

## simple POS ####
prepare_POS <- function(df){
  df <- df %>%
    mutate(simple_POS_1 = make_simple_pos(XPOS1),
           simple_POS_2 = make_simple_pos(XPOS2),
           simple_POS_12 = paste(simple_POS_1,simple_POS_2,sep = '-'))
  df$UPOS12 <- factor(paste(df$UPOS1,df$UPOS2,sep = '-'))
  df$XPOS12 <- factor(paste(df$XPOS1,df$XPOS2,sep = '-'))
  return(df)
}


## open/closed class ####
open_xpos <-
  c("JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
    "NN", "NNS", "NNPS", "NNP", "VBN",
    "VB", "VBZ", "VBG", "VBD", "VBP", "FW")
closed_xpos <-
  c("DT", "CD", "CC",
    "IN", "RP",
    "PRP", "POS", "PRP$", "WP$",
    "PDT", "WDT", "WP", "WRB",
    "EX", "TO", "MD", "LS", "UH")

add_class_predictor <- function(df){
  df <- df %>%
    mutate(class1=if_else(XPOS1 %in% open_xpos,if_else(relation=="neg","CLOSED","OPEN"),
                          if_else(XPOS1 %in% closed_xpos,"CLOSED","?"))) %>%
    mutate(class2=if_else(XPOS2 %in% open_xpos,"OPEN",
                          if_else(XPOS2 %in% closed_xpos,"CLOSED","?")))
  df <- df %>%
    mutate(class1=case_when(
      XPOS1 %in% open_xpos   ~ if_else(relation=="neg","CLOSED","OPEN"),
      XPOS1 %in% closed_xpos ~ "CLOSED",
      TRUE                   ~ "?")) %>%
    mutate(class2=case_when(
      XPOS2 %in% open_xpos   ~ "OPEN",
      XPOS2 %in% closed_xpos ~ "CLOSED",
      TRUE                   ~ "?"))

  df$class12 <- factor(paste(df$class1,df$class2,sep = '-'))
  return(df)
}



# Prepare data with added columns ####
#
# with absolute value
bert <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=bert-large-cased_pad60*.csv"))
xlnet<- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=xlnet-base-cased_pad30*.csv"))
xlm  <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=xlm-mlm-en-2048_pad60*.csv"))
bart <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=bart-large_pad60*.csv"))
gpt2 <- read_csv(Sys.glob("by_wordpair/wordpair_gpt2_pad30*.csv"))
dbert<- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=distilbert-base-cased_pad60*.csv"))
w2v  <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=w2v*.csv"))

lstm  <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=lstm*.csv"))
onlstm <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=onlstm_pad*.csv"))
onlstm_syd <- read_csv(Sys.glob("by_wordpair/wordpair_abs-loaded=onlstm_syd*.csv"))

dbert$model <- "DistilBERT"
xlnet$model <- "XLNet"
bert$model  <- "BERT"
xlm$model   <- "XLM"
bart$model  <- "Bart"
gpt2$model  <- "GPT2"
w2v$model   <- "Word2Vec"

lstm$model <- "LSTM"
onlstm$model <- "ONLSTM"
onlstm_syd$model <- "ONLSTM-SYD"

prepare_df <- function(df){
  df <- df %>% mutate(acc=gold_edge==pmi_edge_sum)
  df$relation[is.na(df$relation)]<-"NONE"
  df <- df %>% prepare_POS() %>% add_class_predictor()
  return(df)
}

bert <- prepare_df(bert)
xlnet<- prepare_df(xlnet)
xlm  <- prepare_df(xlm)
bart <- prepare_df(bart)
gpt2 <- prepare_df(gpt2)
dbert<- prepare_df(dbert)
w2v  <- prepare_df(w2v)
lstm <- prepare_df(lstm)
onlstm  <- prepare_df(onlstm)
onlstm_syd  <- prepare_df(onlstm_syd)

all_bidir <- bind_rows(bert,xlnet,xlm,bart,dbert,w2v)
all_bidir$pmi_edge <- all_bidir$pmi_edge_sum
all_onedir <- bind_rows(gpt2,lstm,onlstm,onlstm_syd)
all_onedir$pmi_edge = all_onedir$pmi_edge_tril
all <- bind_rows(all_bidir,all_onedir)


## Exploratory analysis ####
# ######################## #

## Correlation between models

compare_onedir <- all_onedir %>%
  group_by(sentence_index, i1, i2, model) %>% summarise(pmi_edge) %>%
  pivot_wider(names_from = model, values_from = pmi_edge) %>% ungroup()

compare_all <- all %>%
  group_by(sentence_index, i1, i2, model) %>% summarise(pmi_edge) %>%
  pivot_wider(names_from = model, values_from = pmi_edge) %>% ungroup()

corrcol <- function(data, mapping, method="pearson", use="pairwise", ...){
  # grab data
  x <- eval_data_col(data, mapping$x)
  y <- eval_data_col(data, mapping$y)

  # calculate correlation
  corr <- cor(x, y, method=method, use=use)

  # calculate colour based on correlation value
  # Here I have set a correlation of minus one to blue,
  # zero to white, and one to red
  # Change this to suit: possibly extend to add as an argument
  colFn <- colorRampPalette(c("blue", "white", "red"), interpolate ='spline')
  fill <- colFn(100)[findInterval(corr, seq(-1, 1, length=100))]

  ggally_cor(data = data, mapping = mapping, ...) +
    theme_void() +
    theme(panel.background = element_rect(fill=fill))
}

library("GGally")
ggpairs(compare_onedir %>% select(-c("sentence_index","i1","i2")),
        upper = list(discrete= wrap(corrcol, colour = "black")))

ggpairs(compare_all %>% select(-c("sentence_index","i1","i2")),
        upper = list(discrete= wrap(corrcol, colour = "black")))


## Finding examples of bad correspondence ####

# gold
bert %>% filter(gold_edge==T) %>% group_by(w1,w2,lin_dist) %>% count() %>% arrange(desc(n))
bert %>%  filter(gold_edge==T) %>% group_by(simple_POS_1,simple_POS_2) %>% count() %>% arrange(desc(n))

w2v %>%  filter(gold_edge==T) %>% group_by(simple_POS_1,simple_POS_2) %>% count() %>% arrange(desc(n)) %>%
  ggplot(aes(x=simple_POS_1,y=simple_POS_2,size=n)) + geom_point() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1,vjust=0.5))

# wrong
w2v %>% filter(gold_edge==F,pmi_edge_sum==T) %>% group_by(w1,w2,lin_dist) %>% count() %>% arrange(desc(n))
w2v %>% filter(gold_edge==F,pmi_edge_sum==T) %>% group_by(simple_POS_12,lin_dist) %>% count() %>% arrange(desc(n))

# right
w2v %>%  filter(gold_edge==T,pmi_edge_sum==T) %>% group_by(w1,w2,lin_dist) %>% count() %>% arrange(desc(n))

## Harmonic word order ####

count_proportion_in_order <- function(dataframe, POSpair){
  #' Gets the proportion of edges of the type POS1-POS2 in that order, versus in either order in the data.
  #' Note: It doesn't matter what model dataframe you use, and doesn't matter what the gold arcs are either.
  #'       Result is just about the sentence data.
  dataframe = dataframe %>% filter((simple_POS_1 == POSpair[1] & simple_POS_2 == POSpair[2]) | (simple_POS_1 == POSpair[2] & simple_POS_2 == POSpair[1]))
  denom = count(dataframe)[[1]]
  dataframe = dataframe %>% filter(simple_POS_1 == POSpair[1] & simple_POS_2 == POSpair[2])
  num = count(dataframe)[[1]]
  return(denom)
}
count_proportion_in_order_edge <- function(dataframe, edge, POSpair){
  #' Gets the proportion of edges of the type POS1-POS2 in that order, versus in either order, that are edges
  dataframe = dataframe %>% filter((simple_POS_1 == POSpair[1] & simple_POS_2 == POSpair[2]) | (simple_POS_1 == POSpair[2] & simple_POS_2 == POSpair[1]))
  denom = sum(dataframe[[edge]])
  dataframe = dataframe %>% filter(simple_POS_1 == POSpair[1]  & simple_POS_2 == POSpair[2])
  num = sum(dataframe[[edge]])
  return(denom)
}
proportion_in_order <- function(dataframe, POSpair){
  #' Gets the proportion of edges of the type POS1-POS2 in that order, versus in either order in the data.
  #' Note: It doesn't matter what model dataframe you use, and doesn't matter what the gold arcs are either.
  #'       Result is just about the sentence data.
  dataframe = dataframe %>% filter((simple_POS_1 == POSpair[1] & simple_POS_2 == POSpair[2]) | (simple_POS_1 == POSpair[2] & simple_POS_2 == POSpair[1]))
  denom = count(dataframe)[[1]]
  dataframe = dataframe %>% filter(simple_POS_1 == POSpair[1] & simple_POS_2 == POSpair[2])
  num = count(dataframe)[[1]]
  return(num/denom)
}
proportion_in_order_edge <- function(dataframe, edge, POSpair){
  #' Gets the proportion of edges of the type POS1-POS2 in that order, versus in either order, that are edges
  dataframe = dataframe %>% filter((simple_POS_1 == POSpair[1] & simple_POS_2 == POSpair[2]) | (simple_POS_1 == POSpair[2] & simple_POS_2 == POSpair[1]))
  denom = sum(dataframe[[edge]])
  dataframe = dataframe %>% filter(simple_POS_1 == POSpair[1]  & simple_POS_2 == POSpair[2])
  num = sum(dataframe[[edge]])
  return(num/denom)
}

POSpairs = list(
  c("Adjective","Noun"),
  c("Determiner","Noun"),
  # c("Preposition","Noun"),
  # c("Adverb","Noun"),
  c("Adverb","Verb"),
  # c("Determiner","Cardinal"),
  c("Cardinal","Noun"),
  c("Possessive","Noun"),
  c("Complementizer", "Verb"))

POSpair_num.df <- tibble(
  "POSpair" = sapply(POSpairs, paste, collapse = "-"),
  "baseline"= sapply(POSpairs, count_proportion_in_order, dataframe = bert),
  "gold" = sapply(POSpairs, count_proportion_in_order_edge, dataframe = bert, edge = "gold_edge"),
  "BERT" = sapply(POSpairs, count_proportion_in_order_edge, dataframe = bert, edge = "pmi_edge_sum"),
  "XLNet"= sapply(POSpairs, count_proportion_in_order_edge, dataframe = xlnet, edge = "pmi_edge_sum"),
  "XLM"  = sapply(POSpairs, count_proportion_in_order_edge, dataframe = xlm, edge = "pmi_edge_sum"),
  # "GPT2"  = sapply(POSpairs, count_proportion_in_order_edge, dataframe = gpt2, edge = "pmi_edge_sum"),
  "Bart"  = sapply(POSpairs, count_proportion_in_order_edge, dataframe = bart, edge = "pmi_edge_sum"),
  "DistilBERT" = sapply(POSpairs, count_proportion_in_order_edge, dataframe = dbert, edge = "pmi_edge_sum")) %>%
  pivot_longer(-c(POSpair), names_to = "arctype", values_to = "num")%>%
  mutate(arctype = fct_relevel(factor(arctype),"baseline","gold","BERT","XLNet","XLM","Bart"))


POSpair_proportion.df <- tibble(
  "POSpair" = sapply(POSpairs, paste, collapse = "-"),
  "baseline"= sapply(POSpairs, proportion_in_order, dataframe = bert),
  "gold" = sapply(POSpairs, proportion_in_order_edge, dataframe = bert, edge = "gold_edge"),
  "BERT" = sapply(POSpairs, proportion_in_order_edge, dataframe = bert, edge = "pmi_edge_sum"),
  "XLNet"= sapply(POSpairs, proportion_in_order_edge, dataframe = xlnet, edge = "pmi_edge_sum"),
  "XLM"  = sapply(POSpairs, proportion_in_order_edge, dataframe = xlm, edge = "pmi_edge_sum"),
  # "GPT2"  = sapply(POSpairs, proportion_in_order_edge, dataframe = gpt2, edge = "pmi_edge_sum")
  "Bart"  = sapply(POSpairs, proportion_in_order_edge, dataframe = bart, edge = "pmi_edge_sum"),
  "DistilBERT" = sapply(POSpairs, proportion_in_order_edge, dataframe = dbert, edge = "pmi_edge_sum")) %>%
  pivot_longer(-c(POSpair), names_to = "arctype", values_to = "proportion")%>%
  mutate(arctype = fct_relevel(factor(arctype),"baseline","gold","BERT","XLNet","XLM","Bart"))

joined <-
  full_join(POSpair_proportion.df,POSpair_num.df,by=c("POSpair","arctype"))

joined  %>%
  ggplot(aes(x=reorder(POSpair,-proportion),y=proportion,fill=arctype)) +
  geom_bar(stat='identity', position='dodge') +
  geom_text(aes(x = reorder(POSpair,-proportion), y = -Inf, colour=arctype, label=num), show.legend = F,
            position = position_dodge(width = 1), vjust = 0, hjust=0.5, size = 2, angle=45) +
  coord_cartesian(clip = "off") + annotate("text",x=Inf,y=-Inf, label="n total", size=2.5, hjust=0, vjust=0, colour="grey") +
  # geom_text(data=POSpair_num.df, aes(x=POSpair,label=n,y=Inf), hjust=0, size=3) +
  labs(x="Type of POS pair", y="Proportion of edges in given order", fill="Arc type") +
  ggtitle("Proportion of edges in given order,  by model, for selected types of POS pairs")

## Exploring open/closed class effects ####
make_classdf_gold <- function(df,name){
  return(df %>% filter(gold_edge==TRUE) %>%
           group_by(class12,acc) %>% summarise(n = n()) %>%
           pivot_wider(names_from = acc, values_from = n, values_fill = list(n = 0)) %>% mutate(n=`FALSE`+`TRUE`, acc = `TRUE`/n, model=name) %>% ungroup())
}
make_classdf <- function(df,name,condition){
  return(df %>% filter(eval(parse(text=condition))) %>%
           group_by(class12,acc) %>% summarise(n = n()) %>%
           pivot_wider(names_from = acc, values_from = n, values_fill = list(n = 0)) %>% mutate(n=`FALSE`+`TRUE`, acc = `TRUE`/n, model=name) %>% ungroup())
}


# point plots
four_class_pairs = c("CLOSED-CLOSED","OPEN-CLOSED","CLOSED-OPEN","OPEN-OPEN")

five.acc_by_class <- function(
  title_suffix = "(all arclen)",
  condition = "lin_dist>0"
){
  acc_by_class <- function(df, show.legend, ylabel){
    plot <- df %>% filter(class12 %in% four_class_pairs) %>% filter() %>%
      mutate(class12 = factor(class12, levels=four_class_pairs)) %>%
      ggplot(aes(x=model,y=acc,fill=model)) +
      geom_point(aes(colour=model, size=n),
                 show.legend = show.legend) +
      coord_cartesian(clip = "off") +
      ylim(0,1) +
      facet_wrap(~class12,ncol=4) +
      theme(plot.margin = ggplot2::margin(2, 30, 2, 2, "pt")) +
      ylab(paste(ylabel[[1]], ylabel[[2]])) +
      xlab("")+
      theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust=0.5)) +
      ggtitle(paste("Accuracy (", ylabel[[1]], ") by word-class pair ", title_suffix, sep = ""))
    if(ylabel[[1]]=="precision"){
      plot <- plot + geom_text(aes(label=paste("n =",n),y=acc, colour=model), alpha=0.75,
                               show.legend = F,
                               vjust = 0.5, hjust=-.5, size = 3, angle=90)
    }
    if(ylabel[[1]]=="recall"){
      plot <- plot + geom_text(aes(x="DistilBERT",label=paste("n =",n),y=0.95), colour="darkgrey",
                               show.legend = F, size = 3)
    }
    return(plot)
  }

  five.classdf.gold <-
    bind_rows(map2(
      map(list(dbert,bart,bert,xlnet,w2v),filter,eval(parse(text=condition))),
      list("DistilBERT","Bart","BERT","XLNet","Word2Vec"),
      make_classdf,condition="gold_edge==TRUE")) %>% arrange(class12)
  five.classdf.pmi <-
    bind_rows(map2(
      map(list(dbert,bart,bert,xlnet,w2v),filter,eval(parse(text=condition))),
      list("DistilBERT","Bart","BERT","XLNet","Word2Vec"),
      make_classdf,condition="pmi_edge_sum==TRUE")) %>% arrange(class12)

  p.class.gold <- acc_by_class(
    five.classdf.gold, show.legend = F,
    ylabel = c("recall","(# PMI arc = gold arc)/(# gold arcs)"))
  p.class.pmi  <- acc_by_class(
    five.classdf.pmi,  show.legend = F,
    ylabel = c("precision","(# PMI arc = gold arc)/(# PMI arcs)"))

  gridExtra::grid.arrange(p.class.gold,p.class.pmi,ncol=2)
}

five.acc_by_class(title_suffix = "(all arclen)", condition = "lin_dist>0")
five.acc_by_class(title_suffix = "(arclen=1)", condition = "lin_dist==1")
five.acc_by_class(title_suffix = "(arclen>1)", condition = "lin_dist>1")





leftover<-bert %>% filter(class12=="CLOSED-?") %>% group_by(simple_POS_12) %>% count() %>% arrange(desc(n))

bert %>% ggplot(aes(x=reorder(class12,class12,length))) + geom_bar() + coord_flip()

# joined  %>%
#   ggplot(aes(x=reorder(POSpair,-proportion),y=proportion,fill=arctype)) +
#   geom_bar(stat='identity', position='dodge') +
#   geom_text(aes(x = reorder(POSpair,-proportion), y = -Inf, colour=arctype, label=num), show.legend = F,
#             position = position_dodge(width = 1), vjust = 0, hjust=0.5, size = 2, angle=45) +
#   coord_cartesian(clip = "off") + annotate("text",x=Inf,y=-Inf, label="n total", size=2.5, hjust=0, vjust=0, colour="grey") +
#   # geom_text(data=POSpair_num.df, aes(x=POSpair,label=n,y=Inf), hjust=0, size=3) +
#   labs(x="Type of POS pair", y="Proportion of edges in given order", fill="Arc type") +
#   ggtitle("Proportion of edges in given order,  by model, for selected types of POS pairs")

## Sentence length histogram ####
bert %>% filter(gold_edge==T) %>% group_by(sentence_index) %>% summarise(len=max(i2)) %>% ungroup() %>%
  ggplot(aes(x=len)) + geom_histogram(binwidth = 1)

bert %>% filter(gold_edge==T) %>% group_by(i1,sentence_index) %>% summarise(n=n()) %>% arrange(desc(n)) %>% .$n %>% sd

bert %>% filter(gold_edge) %>% select(relation) %>% as.factor()

## Looking at the number of valency of 'hubs': max dependencies per word per sentence ####
maxhub<-function(df,name){
  i1 <- df %>% group_by(i1,sentence_index) %>% summarise(n=n()) %>% arrange(desc(sentence_index)) %>% ungroup()
  i2 <- df %>% group_by(i2,sentence_index) %>% summarise(n=n()) %>% arrange(desc(sentence_index)) %>% ungroup()
  dfout <- bind_rows(i1,i2) %>% group_by(sentence_index) %>%
    summarise(max_hub=max(n)) %>% arrange(desc(max_hub))
  dfout$model = name
  return(dfout)
}
maxhub.df<-bind_rows(purrr::map2(
  list(bert %>% filter(gold_edge==T),
       bert %>% filter(pmi_edge_sum==T),
       dbert %>% filter(pmi_edge_sum==T),
       xlm %>% filter(pmi_edge_sum==T),
       xlnet %>% filter(pmi_edge_sum==T),
       bart %>% filter(pmi_edge_sum==T),
       #gpt2 %>% filter(pmi_edge_sum==T),
       w2v %>% filter(pmi_edge_sum==T)
  ),
  list("gold","Bart","BERT","DistilBERT","Word2Vec","XLM","XLNet" #,"GPT2"
  ),
  maxhub))
maxhub.df$model <- maxhub.df$model %>% fct_relevel(levels=c("gold","Bart","BERT","DistilBERT","Word2Vec","XLM","XLNet" #,"GPT2",
))
p.maxhub<-maxhub.df %>% filter(model!="gold") %>%
  ggplot(aes(x=max_hub)) + geom_histogram(binwidth = 1, aes(fill=model),show.legend=F) +
  xlab("max arity in sentence") + scale_y_continuous(limits = c(0,900)) +
  scale_x_continuous(breaks = c(2,3,4,5,6,7,8,9),limits = c(1.5,9.5)) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank()) +
  facet_wrap(model~.,nrow=2,strip.position = "top",scales = "free_x")
p.maxhubgold<-maxhub.df %>% filter(model=="gold") %>%
  ggplot(aes(x=max_hub)) + geom_histogram(binwidth = 1, fill="darkgrey",show.legend=F) +
  xlab("max arity in sentence") + scale_y_continuous(limits = c(0,900)) +
  scale_x_continuous(breaks = c(2,3,4,5,6,7,8,9),limits = c(1.5,9.5)) +
  theme(axis.title.y = element_blank(),
        axis.title.x = element_blank()) +
  facet_wrap(model~.,nrow=2,strip.position = "top", scales = "free_x")

grid.arrange(arrangeGrob(p.maxhubgold,
                         heights=unit(0.5,"npc")),
             p.maxhub, nrow=1,widths=c(12,30),
             top="Max arity histograms",
             bottom="max arity in sentence")

# maxhub_hist<-function(df,fill){
#   i1 <- df %>% group_by(i1,sentence_index) %>% summarise(n=n()) %>% arrange(desc(sentence_index)) %>% ungroup()
#   i2 <- df %>% group_by(i2,sentence_index) %>% summarise(n=n()) %>% arrange(desc(sentence_index)) %>% ungroup()
#   bind_rows(i1,i2) %>% group_by(sentence_index) %>%
#     summarise(max_hub=max(n)) %>% arrange(desc(max_hub)) %>%
#     ggplot(aes(x=max_hub)) + geom_histogram(binwidth = 1, fill=fill) +
#     scale_y_continuous(limits = c(0,900)) + xlab("") +
#     scale_x_continuous(breaks = c(2,3,4,5,6,7,8,9,10,11,12,13,14,15),limits = c(0,15))
# }
# grid.arrange(maxhub_hist(bert  %>% filter(gold_edge==T),"darkgrey")         + ggtitle("gold")       +theme(axis.text.x = element_blank()),
#              maxhub_hist(bart  %>% filter(pmi_edge_sum==T),modelcols[[1]])  + ggtitle("Bart")+theme(axis.text.x = element_blank()),
#              maxhub_hist(bert  %>% filter(pmi_edge_sum==T),modelcols[[2]])  + ggtitle("BERT")+theme(axis.text.x = element_blank()),
#              maxhub_hist(dbert %>% filter(pmi_edge_sum==T),modelcols[[3]])  + ggtitle("DistilBERT")+theme(axis.text.x = element_blank()),
#              maxhub_hist(w2v   %>% filter(pmi_edge_sum==T),modelcols[[4]])  + ggtitle("Word2Vec")+theme(axis.text.x = element_blank()),
#              maxhub_hist(xlm   %>% filter(pmi_edge_sum==T),modelcols[[5]])  + ggtitle("XLM")+theme(axis.text.x = element_blank()),
#              maxhub_hist(xlnet %>% filter(pmi_edge_sum==T),modelcols[[6]])  + ggtitle("XLNet") + xlab("max arity in sentence") ,
#              # maxhub_hist(gpt2  %>% filter(pmi_edge_sum==T),modelcols[[7]])  + ggtitle("GPT2"),
#              ncol=1,
#              top="Maximum arity histograms")
#
# maxhub_hist_all<-function(df){
#   i1 <- df %>% group_by(i1,sentence_index,model) %>% summarise(n=n()) %>% arrange(desc(sentence_index)) %>% ungroup()
#   i2 <- df %>% group_by(i2,sentence_index,model) %>% summarise(n=n()) %>% arrange(desc(sentence_index)) %>% ungroup()
#   bind_rows(i1,i2) %>% group_by(sentence_index,model) %>%
#     summarise(max_hub=max(n)) %>% arrange(desc(max_hub)) %>%
#     ggplot(aes(x=max_hub)) + geom_histogram(binwidth = 1, fill=model) +
#     scale_y_continuous(limits = c(0,900)) + xlab("") +
#     scale_x_continuous(breaks = c(2,3,4,5,6,7,8,9,10,11,12,13,14,15),limits = c(0,15))
# }

## Accuracy ####

#Linear baseline accuracy
nrow(filter(bert, lin_dist==1,gold_edge==T))/nrow(filter(bert,lin_dist==1))
#Overall accuracy
acc<-function(x){nrow(filter(x, gold_edge==T, pmi_edge_sum==T)) / nrow(filter(x, pmi_edge_sum==T))}

acc(dbert)
acc(xlnet)
acc(bert)
acc(xlm)
acc(bart)
acc(gpt2)
acc(w2v)
#Sentence avg accuracy
sentacc<-function(x){group_by(x,sentence_index) %>% group_map(~acc(.x)) %>% unlist %>% mean}

# sentence averaged are a little higher,
# since gives less weight to edges in longer sentences
# but the difference is only like 0.01--0.03
sentacc(dbert) - acc(dbert)# 0.010
sentacc(xlnet) - acc(xlnet)# 0.011
sentacc(bert)  - acc(bert) # 0.014
sentacc(xlm)   - acc(xlm)  # 0.016
sentacc(bart)  - acc(bart) # 0.018
sentacc(gpt2)  - acc(gpt2) # 0.028

sentacc(w2v)   - acc(w2v)  # 0.018




## Relation #####################


prepare_by_relation <- function(dataframe,length_greater_than=0){
  #' Prepare csv as df data grouped by 'relation'
  relation_len = dataframe %>% filter(gold_edge==T,
                                      lin_dist>length_greater_than) %>%
    group_by(relation) %>% summarise(medlen=median(lin_dist), meanlen=mean(lin_dist), n=n(),
                                     meanpmi=mean(pmi_sum), varpmi=var(pmi_sum))
  dataframe = dataframe %>% filter(gold_edge==T,
                                   lin_dist>length_greater_than) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(relation,acc) %>% summarise(n=n(), medlen=median(lin_dist), meanlen=mean(lin_dist)) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n,medlen,meanlen), values_fill = list(n = 0)) %>%
    left_join(relation_len, by="relation") %>% mutate(pct_acc = n_pmiTRUE/n)
  return(dataframe)
}

# xlnet.relation <-prepare_by_relation(read_csv("by_wordpair/wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
# bert.relation <- prepare_by_relation(read_csv("by_wordpair/wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
# xlm.relation <-  prepare_by_relation(read_csv("by_wordpair/wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))
# bart.relation <- prepare_by_relation(read_csv("by_wordpair/wordpair_bart-large_pad60_2020-04-27-01-20.csv"))
# dbert.relation <-prepare_by_relation(read_csv("by_wordpair/wordpair_distilbert-base-cased_pad60_2020-04-29-19-35.csv"))
# gpt2.relation <- prepare_by_relation(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"))
# w2v.relation  <- prepare_by_relation(read_csv("by_wordpair/wordpair_w2v_pad0_2020-05-17-00-47.csv"))

xlnet.relation <-prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=xlnet-base-cased_pad30_2020-07-05-17-41.csv"))
bert.relation <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=bert-large-cased_pad60_2020-07-05-16-29.csv"))
xlm.relation <-  prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=xlm-mlm-en-2048_pad60_2020-07-05-17-29.csv"))
bart.relation <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=bart-large_pad60_2020-07-05-16-06.csv"))
dbert.relation <-prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=distilbert-base-cased_pad60_2020-07-05-17-05.csv"))
# gpt2.relation <- prepare_by_relation(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"))
w2v.relation  <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=w2v_pad0_2020-07-05-17-17.csv"))


# All three models in one df
join_three <- function(df1, df2, df3,
                       #' to full_join three data frames
                       by=c("n","meanlen"),
                       suffixes=c(".BERT",".XLNet",".XLM")){
  return(
    full_join(df1,df2,by=by,suffix=c(".BERT",".XLNet")) %>%
      full_join(rename_at(df3, vars(-by), function(x){paste0(x,suffixes[3])}), by=by) %>%
      pivot_longer(cols = -by, names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)"))
}

three.relation <- join_three(bert.relation,xlnet.relation,xlm.relation,
                             by=c("n","relation","meanlen"),
                             suffixes=c(".BERT",".XLNet",".XLM"))

# All four models in one df
join_four <- function(df1, df2, df3, df4,
                      #' to full_join four data frames
                      by=c("n","meanlen"),
                      suffixes=c(".Bart",".BERT",".XLNet",".XLM")){
  return(
    full_join(df1,df2,by=by,suffix=suffixes[1:2]) %>%
      full_join(rename_at(df3, vars(-by), function(x){paste0(x,suffixes[3])}), by=by) %>%
      full_join(rename_at(df4, vars(-by), function(x){paste0(x,suffixes[4])}), by=by) %>%
      pivot_longer(cols = -by, names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)"))
}

four.relation <- join_four(bart.relation,bert.relation,xlnet.relation,xlm.relation,
                           by=c("n","relation","meanlen"),
                           suffixes=c(".Bart",".BERT",".XLNet",".XLM"))
# All five models in one df
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

five.relation <- join_five(dbert.relation,bart.relation,bert.relation,xlnet.relation,w2v.relation,
                           by=c("n","relation","meanlen"),
                           suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))


# A plot exploring accuracy by relation with respect to linear distance, model, and n
p.rel <-
  five.relation %>%  filter(n>50) %>%
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
  ylab("recall (# CPMI arc = gold arc)/(# gold arcs)") +
  xlab("gold dependency label (ordered by mean arc length)") +
  ggtitle("all arc lengths") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.spacing = unit(0, 'cm'))


## same, only for arc-length ≥ 1 ####

# xlnet.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"),length_greater_than = 1)
# bert.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"),length_greater_than = 1)
# xlm.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"),length_greater_than = 1)
# bart.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_bart-large_pad60_2020-04-27-01-20.csv"),length_greater_than = 1)
# dbert.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_distilbert-base-cased_pad60_2020-04-29-19-35.csv"),length_greater_than = 1)
# gpt2.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"),length_greater_than = 1)
# w2v.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_w2v_pad0_2020-05-17-00-47.csv"), length_greater_than = 1)

xlnet.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=xlnet-base-cased_pad30_2020-07-05-17-41.csv"),length_greater_than = 1)
bert.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=bert-large-cased_pad60_2020-07-05-16-29.csv.csv"),length_greater_than = 1)
xlm.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=xlm-mlm-en-2048_pad60_2020-07-05-17-29.csvv"),length_greater_than = 1)
bart.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=bart-large_pad60_2020-07-05-16-06.csv"),length_greater_than = 1)
dbert.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=distilbert-base-cased_pad60_2020-07-05-17-05.csv"),length_greater_than = 1)
# gpt2.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"),length_greater_than = 1)
w2v.relation.gt1 <- prepare_by_relation(read_csv("by_wordpair/wordpair_abs-loaded=w2v_pad0_2020-07-05-17-17.csv"), length_greater_than = 1)


three.relation.gt1 <- join_three(bert.relation.gt1,xlnet.relation.gt1,xlm.relation.gt1,
                                 by=c("n","meanlen"),
                                 suffixes=c(".BERT",".XLNet",".XLM"))
four.relation.gt1 <- join_four(bart.relation.gt1,bert.relation.gt1,xlnet.relation.gt1,xlm.relation.gt1,
                               by=c("n","meanlen"),
                               suffixes=c(".Bart",".BERT",".XLNet",".XLM"))
five.relation.gt1 <- join_five(dbert.relation.gt1,bart.relation.gt1,bert.relation.gt1,xlnet.relation.gt1,w2v.relation.gt1,
                               by=c("n","meanlen"),
                               suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))
p.rel.gt1<-
  five.relation.gt1 %>%  filter(n>50) %>%
  ggplot(aes(y=pct_acc, x=reorder(relation, meanlen))) +
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="grey") +
  geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="grey") +  # to print n
  annotate("text",x=Inf,y=-Inf, label="mean arclength", size=3, hjust=0.5, vjust=0) +
  geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3, alpha=0.2) +
  geom_line(aes(group=relation), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.6) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
        plot.margin = ggplot2::margin(0, 50, 2, 2, "pt"),
        axis.ticks = element_blank()) +
  ylab("recall (# CPMI arc = gold arc)/(# gold arcs)") +
  xlab("gold dependency label (ordered by mean arc length)") +
  ggtitle("arc length > 1") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.spacing = unit(0, 'cm'))
p.rel.gt1 +ggtitle("Accuracy (recall) by gold label (n>50, arc length > 1)")

grid.arrange(p.rel,p.rel.gt1,ncol=2,top="Accuracy (recall) by gold label (n>50)")

# PMI value ####
#

five.relation %>%  filter(n>200) %>%
  ggplot(aes(y=meanpmi, x=reorder(relation,pct_acc))) +
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="grey") +
  geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="grey") +  # to print n
  annotate("text",x=Inf,y=-Inf, label="mean\narclength", size=3, hjust=0, vjust=0) +
  geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3) +
  geom_line(aes(group=relation), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  coord_flip(clip = "off") +
  theme(legend.position="top", plot.margin = ggplot2::margin(2, 50, 2, 2, "pt"),
        axis.ticks = element_blank()) +
  ylab("mean PMI for arclabel") +
  xlab("gold dependency label (ordered by mean accuracy)") +
  ggtitle("Mean cpmi by gold label (n>50)")

five.relation %>% filter(n>50) %>%
  ggplot(aes(y=pct_acc, x=meanpmi)) +
  stat_smooth(aes(colour=model),geom="line",alpha=0.5,size=1) +
  geom_smooth(aes(colour=model),alpha=0.1,size=0) +
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="blue") +
  geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="blue") +  # to print n
  # annotate("text",x=Inf,y=-Inf, label="mean\narclength", size=3, hjust=0.5, vjust=0) +
  # geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3) +
  geom_line(aes(group=relation), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  geom_text(aes(label=relation), hjust=0.5, size=2) +
  coord_flip(clip = "off") +
  theme(legend.position="top", plot.margin = ggplot2::margin(2, 50, 2, 2, "pt"),
        axis.ticks = element_blank()) +
  ylab("recall (# CPMI arc = gold arc)/(# gold arcs)") +
  xlab("mean CPMI value per dependency label") +
  ggtitle("Comparing mean pmi value with accuracy in Bart, BERT, XLNet, XLM, by gold label(n>50)")


# len / lin_dist ####

# quick histograms
gold.len <- bert %>% filter(gold_edge==T) %>% group_by(lin_dist) %>% count
bert.len <- bert %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
xlnet.len <- xlnet %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
xlm.len <- xlm %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
gpt2.len <- gpt2 %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
bart.len <- bart %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
dbert.len <- dbert %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
w2v.len <- w2v %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count

ggcolhue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
modelcols<-ggcolhue(6)
goldcol<-ggcolhue(5)[[2]]

bert %>% filter(gold_edge==T) %>% ggplot(aes(x=lin_dist)) + geom_histogram() + scale_x_log10()

plothist<-function(df.len,fill,xlab,ylabel=T){
  yval = c(5, 10, 15, 20, 25)
  p<-df.len %>%  ggplot(aes(x=lin_dist,y=n)) + geom_col(fill=fill,colour=fill) +
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

hgold <- gold.len %>%  ggplot(aes(x=lin_dist,y=n)) + geom_col()  + scale_x_continuous(breaks = c(1,2,3,4,5,6,7,8,9), limits=c(0,10)) + xlab("gold")
hgold <- plothist(gold.len,"darkgrey", "gold")
hbart <- plothist(bart.len,modelcols[[1]], "Bart")
hbert <- plothist(bert.len,modelcols[[2]], "BERT")
hdbert <-plothist(dbert.len,modelcols[[3]],"DistilBERT")
hw2v <- plothist(w2v.len,modelcols[[4]],"Word2Vec")
hxlm <-  plothist(xlm.len,modelcols[[5]],  "XLM")
hxlnet <-plothist(xlnet.len,modelcols[[6]],"XLNet")
hgpt2 <- plothist(gpt2.len,modelcols[[7]], "GPT2")

grid.arrange(arrangeGrob(hgold,
                         heights=unit(0.5,"npc")),
             arrangeGrob(hbart, hbert,  hdbert,
                         hw2v,  hxlnet, hxlm,
                         nrow=2),
             nrow=1,widths=c(1,3),
             top="Dependency arc length histograms")
# sun   sum(abs)
gold.len[1,]$n/(bert %>% filter(gold_edge==T) %>%  count())      # 0.478 0.478
bart.len[1,]$n/(xlm %>% filter(pmi_edge_sum==T) %>%  count())    # 0.576 0.546
bert.len[1,]$n/(bert %>% filter(pmi_edge_sum==T) %>%  count())   # 0.709 0.737
dbert.len[1,]$n/(dbert %>% filter(pmi_edge_sum==T) %>%  count()) # 0.601 0.636
xlm.len[1,]$n/(xlm %>% filter(pmi_edge_sum==T) %>%  count())     # 0.498 0.505
xlnet.len[1,]$n/(xlnet %>% filter(pmi_edge_sum==T) %>%  count()) # 0.535 0.552
gpt2.len[1,]$n/(gpt2 %>% filter(pmi_edge_sum==T) %>%  count())   # 0.767 0.767
w2v.len[1,]$n/(w2v %>% filter(pmi_edge_sum==T) %>%  count())     # 0.389 0.161

# GOLD LEN
prepare_by_len_gold <- function(dataframe){
  #' Prepare csv as df data grouped by 'lin_dist'
  len = dataframe %>% filter(!is.na(relation)) %>%
    group_by(lin_dist) %>% summarise(meanpmi=mean(pmi_sum), varpmi=var(pmi_sum), n=n())
  dataframe = dataframe %>% filter(!is.na(relation)) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(lin_dist,acc) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(len, by="lin_dist") %>%
    mutate(pct_acc = pmiTRUE/n)
  return(dataframe)
}

# xlnet.len.gold <- prepare_by_len_gold(read_csv("by_wordpair/wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
# bert.len.gold <-  prepare_by_len_gold(read_csv("by_wordpair/wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
# xlm.len.gold <-   prepare_by_len_gold(read_csv("by_wordpair/wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))
# bart.len.gold <-  prepare_by_len_gold(read_csv("by_wordpair/wordpair_bart-large_pad60_2020-04-27-01-20.csv"))
# gpt2.len.gold <-  prepare_by_len_gold(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"))
# dbert.len.gold <- prepare_by_len_gold(read_csv("by_wordpair/wordpair_distilbert-base-cased_pad60_2020-04-29-19-35.csv"))
# w2v.len.gold <-   prepare_by_len_gold(read_csv("by_wordpair/wordpair_w2v_pad0_2020-05-17-00-47.csv"))

xlnet.len.gold <- prepare_by_len_gold(read_csv("by_wordpair/wordpair_abs-loaded=xlnet-base-cased_pad30_2020-07-05-17-41.csv"))
bert.len.gold <-  prepare_by_len_gold(read_csv("by_wordpair/wordpair_abs-loaded=bert-large-cased_pad60_2020-07-05-16-29.csv"))
xlm.len.gold <-   prepare_by_len_gold(read_csv("by_wordpair/wordpair_abs-loaded=xlm-mlm-en-2048_pad60_2020-07-05-17-29.csv"))
bart.len.gold <-  prepare_by_len_gold(read_csv("by_wordpair/wordpair_abs-loaded=bart-large_pad60_2020-07-05-16-06.csv"))
# gpt2.len.gold <-  prepare_by_len_gold(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"))
dbert.len.gold <- prepare_by_len_gold(read_csv("by_wordpair/wordpair_abs-loaded=distilbert-base-cased_pad60_2020-07-05-17-05.csv"))
w2v.len.gold <-   prepare_by_len_gold(read_csv("by_wordpair/wordpair_abs-loaded=w2v_pad0_2020-07-05-17-17.csv"))

# All three models in one df
three.len.gold <- join_three(bert.len.gold,xlnet.len.gold,xlm.len.gold,
                             by = c("n","lin_dist"),
                             suffixes=c(".BERT",".XLNet",".XLM"))
# All four models in one df
four.len.gold <- join_four(bart.len.gold,bert.len.gold,xlnet.len.gold,xlm.len.gold,
                           by = c("n","lin_dist"),
                           suffixes=c(".Bart",".BERT",".XLNet",".XLM"))
# All five models in one df
five.len.gold <- join_five(dbert.len.gold,bart.len.gold,bert.len.gold,xlnet.len.gold,w2v.len.gold,
                           by = c("n","lin_dist"),
                           suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))

# A plot exploring accuracy by lin_dist
p.lin_dist <-
  five.len.gold %>% filter(n>25) %>%
  ggplot(aes(y=pct_acc, x=lin_dist)) +
  geom_text(aes(label=n, y=Inf), hjust=0, size=2.5, colour="grey") +
  annotate("text",x=Inf,y=Inf, label="n", size=2.5, hjust=0, vjust=0, colour="grey") +
  geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.7) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
        plot.margin = ggplot2::margin(0, 50, 0, 2, "pt"),
        legend.spacing = unit(0, 'cm')
  ) + scale_x_continuous(trans="identity",breaks = seq(1, 23, by = 2), minor_breaks = seq(1, 23, by = 1)) +
  ylab("recall (# CPMI arc = gold arc)/(# gold arcs)") +
  xlab("arc length") +
  ggtitle("Accuracy (recall) by arc length (n>25)")

# PRED PMI LEN
prepare_by_len_pred <- function(dataframe){
  #' Prepare csv as df data grouped by 'lin_dist'
  len = dataframe %>% filter(pmi_edge_sum==T) %>%
    group_by(lin_dist) %>% summarise(meanpmi=mean(pmi_sum), varpmi=var(pmi_sum), n=n())
  dataframe = dataframe %>% filter(pmi_edge_sum==T) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(lin_dist,acc) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(len, by="lin_dist") %>%
    mutate(pct_acc = pmiTRUE/n)
  return(dataframe)
}

xlnet.len.pred <- prepare_by_len_pred(read_csv("by_wordpair/wordpair_abs-loaded=xlnet-base-cased_pad30_2020-07-05-17-41.csv"))
bert.len.pred <-  prepare_by_len_pred(read_csv("by_wordpair/wordpair_abs-loaded=bert-large-cased_pad60_2020-07-05-16-29.csv"))
xlm.len.pred <-   prepare_by_len_pred(read_csv("by_wordpair/wordpair_abs-loaded=xlm-mlm-en-2048_pad60_2020-07-05-17-29.csv"))
bart.len.pred <-  prepare_by_len_pred(read_csv("by_wordpair/wordpair_abs-loaded=bart-large_pad60_2020-07-05-16-06.csv"))
# gpt2.len.pred <-  prepare_by_len_pred(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"))
dbert.len.pred <- prepare_by_len_pred(read_csv("by_wordpair/wordpair_abs-loaded=distilbert-base-cased_pad60_2020-07-05-17-05.csv"))
w2v.len.pred <-   prepare_by_len_pred(read_csv("by_wordpair/wordpair_abs-loaded=w2v_pad0_2020-07-05-17-17.csv"))

# All three models in one df
three.len.pred <- join_three(bert.len.pred,xlnet.len.pred,xlm.len.pred,
                             by = c("n","lin_dist"),
                             suffixes=c(".BERT",".XLNet",".XLM"))
# All four models in one df
four.len.pred <- join_four(bart.len.pred,bert.len.pred,xlnet.len.pred,xlm.len.pred,
                           by = c("n","lin_dist"),
                           suffixes=c(".Bart",".BERT",".XLNet",".XLM"))
# All five models in one df
five.len.pred <- join_five(dbert.len.pred,bart.len.pred,bert.len.pred,xlnet.len.pred,w2v.len.pred,
                           by = c("n","lin_dist"),
                           suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".Word2Vec"))


# A plot exploring accuracy by lin_dist
p.lin_dist.pred <-
  five.len.pred %>%  filter(n>25) %>%
  ggplot(aes(y=pct_acc, x=lin_dist)) +
  #geom_text(aes(label=n, y=Inf), hjust=0, size=3, colour="grey") +
  #annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0, colour="grey") +
  #geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical", legend.margin = margin(),
        plot.margin = ggplot2::margin(0, 20, 2, 0, "pt"),
        legend.spacing = unit(0, 'cm')
  ) + scale_x_continuous(trans="identity", breaks = seq(1, 35, by = 4), minor_breaks = seq(1, 35, by = 1)) +
  ylab("precision (# CPMI arc = gold arc)/(# CPMI arcs)") +
  xlab("arc length") +
  ggtitle("Accuracy (precision) by arc length (n>25)")

grid.arrange(p.lin_dist,p.lin_dist.pred,ncol=2,widths=c(10,9))


## ALL edges
##

prepare_by_len_all <- function(dataframe){
  #' Prepare csv as df data grouped by 'lin_dist'
  len = dataframe %>% #filter(pmi_edge_sum==T) %>%
    group_by(lin_dist) %>% summarise(meanpmi=mean(pmi_sum), varpmi=var(pmi_sum), n=n())
  dataframe = dataframe %>% #filter(pmi_edge_sum==T) %>%
    mutate(acc=gold_edge==pmi_edge_sum) %>%
    group_by(lin_dist,acc) %>% summarise(n=n()) %>%
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n), values_fill = list(n = 0)) %>%
    left_join(len, by="lin_dist") %>%
    mutate(pct_acc = pmiTRUE/n)
  return(dataframe)
}

xlnet.len.all <- prepare_by_len_all(read_csv("by_wordpair/wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.len.all <-  prepare_by_len_all(read_csv("by_wordpair/wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.len.all <-   prepare_by_len_all(read_csv("by_wordpair/wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))
bart.len.all <-  prepare_by_len_all(read_csv("by_wordpair/wordpair_bart-large_pad60_2020-04-27-01-20.csv"))
gpt2.len.all <-  prepare_by_len_all(read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv"))
dbert.len.all <- prepare_by_len_all(read_csv("by_wordpair/wordpair_distilbert-base-cased_pad60_2020-04-29-19-35.csv"))
three.len.all <- join_three(bert.len.all,xlnet.len.all,xlm.len.all,
                            by = c("n","lin_dist"),
                            suffixes=c(".BERT",".XLNet",".XLM"))
five.len.all <- join_five(dbert.len.all,bart.len.all,bert.len.all,xlnet.len.all,xlm.len.all,
                          by = c("n","lin_dist"),
                          suffixes=c(".DistilBERT",".Bart",".BERT",".XLNet",".XLM"))

# A plot exploring accuracy by lin_dist
five.len.all %>%  filter(n>200) %>%
  ggplot(aes(y=pct_acc, x=lin_dist)) +
  #geom_text(aes(label=n, y=Inf), hjust=0, size=3, colour="grey") +
  #annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0, colour="grey") +
  #geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical", legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 2, 2, 2, "pt")
  ) + #scale_x_continuous(breaks = seq(1, 35, by = 4), minor_breaks = seq(1, 35, by = 1)) +
  ylab("precision (# PMI arc = gold arc)/(# PMI arcs)") +
  xlab("arc length") +
  ggtitle("Precision by arc length (n>200)")

##
## PMI value vs accuracy, grouped by length
##

bert %>% filter(gold_edge==T,lin_dist<25) %>%
  ggplot(aes(x=lin_dist,y=pmi_sum)) + geom_point(alpha=0.1) + geom_smooth(method=lm)
bert %>% filter(pmi_edge_sum==T) %>%
  ggplot(aes(x=lin_dist,y=pmi_sum)) + geom_point(alpha=0.1) + geom_smooth(method=lm)

summary(lm(pmi_sum~lin_dist, data = bert %>% filter(gold_edge==T)))
summary(lm(pmi_sum~lin_dist, data = bert %>% filter(pmi_edge_sum==T)))
#PMI and lin_dist are correlated p<.001...
#effect negative, size nonnegligible for gold and pmi edges...


p.pmi.gold <- five.len.gold %>%  filter(lin_dist<15) %>%
  ggplot(aes(y=pct_acc, x=meanpmi)) +
  geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n,colour=model), alpha=0.8) +
  # geom_text(aes(label=lin_dist,colour=model,x=9-lin_dist/3), hjust=0, size=3) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical", legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 50, 2, 2, "pt")
  ) + scale_x_continuous(breaks = seq(-3, 23, by = 1)) +
  ylab("recall (# PMI arc = gold arc)/(# gold arcs)") +
  xlab("mean PMI value") +
  ggtitle("Accuracy (recall) by mean PMI (arc length < 15)")
p.pmi.pred <- five.len.pred %>%  filter(lin_dist<15)%>%
  ggplot(aes(y=pct_acc, x=meanpmi)) +
  geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n,colour=model), alpha=0.8) +
  # geom_text(aes(label=lin_dist, colour=model,x=8.5-lin_dist/3), hjust=0, size=3) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical", legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 50, 2, 2, "pt")
  ) + scale_x_continuous(breaks = seq(-1, 23, by = 1)) +
  ylab("precision (# PMI arc = gold arc)/(# PMI arcs)") +
  xlab("mean PMI value") +
  ggtitle("Accuracy (precision) by mean PMI (arc length < 15)")

grid.arrange(p.pmi.gold,p.pmi.pred,ncol=2)





# OTHER IDEAS #########

{# plot common xpos pairs
  XPOS_common_pairs = (bert %>% group_by(XPOS12) %>% summarise(n=n()) %>% filter(n>2000))[[1]]
  bert %>% filter(XPOS12 %in% XPOS_common_pairs) %>%
    ggplot(aes(x=reorder(XPOS12,class12,length), fill=class12)) + geom_bar() + coord_flip()
}
{#plot common simple pairs
  simple_common_pairs = (bert %>% group_by(simple_POS_12) %>% summarise(n=n()) %>% filter(n>2000))[[1]]
  bert %>% filter(simple_POS_12 %in% simple_common_pairs) %>%  ggplot(aes(x=reorder(simple_POS_12,class12,length), fill=class12)) + geom_bar() + coord_flip()
}

bert %>% filter(gold_edge==TRUE) %>%  ggplot(aes(x=reorder(XPOS12,XPOS12,length),fill=acc)) + geom_bar() + coord_flip()


bert %>% group_by(UPOS12) %>% summarise(n=n()) %>% filter(n>2000) %>%
  ggplot(aes(x=reorder(UPOS12,n),y=n)) + geom_bar(stat="identity") + coord_flip() +
  ggtitle("Most common (n>2000) UPOS pairs")
bert %>% group_by(XPOS12) %>% summarise(n=n()) %>% filter(n>2000) %>%
  ggplot(aes(x=reorder(XPOS12,n),y=n)) + geom_bar(stat="identity") + coord_flip() +
  ggtitle("Most common (n>2000) XPOS pairs")


bert %>% group_by(simple_POS_12) %>% summarise(n=n()) %>% arrange(desc(n))

# Phase edge predictor.
# WDT is wh-determiner, when "which" and "that" are POS1,
# when they are not under the "dobj" relation, when w1 == Verb
bert %>% filter(gold_edge==T, simple_POS_1 == "Complementizer", simple_POS_2=="Verb", relation!="dobj") %>%
  group_by(XPOS1,XPOS2,relation) %>% summarise(n=n(),mean_dist=mean(lin_dist))

bert %>% filter(gold_edge==T, simple_POS_2=="Adjective") %>%
  group_by(XPOS1,XPOS2,relation) %>% summarise(n=n(),mean_dist=mean(lin_dist)) %>% arrange(desc(n))



#########################################
# Running a random forest classifier ####
#########################################

# library(caret)
library(ranger)
library(ggpubr)

gg_varimp <- function(ranger,title) {
  #' just a ggplot2 version of dotplot varimp.
  #' input ranger model object, to plot variable importance
  ggplot(stack(ranger$variable.importance),
         aes(x=reorder(ind,values), y=values))+
    geom_point() +
    coord_flip() +
    ylab("Information value (Gini corrected)")+
    xlab("Variable")+
    ggtitle(paste(title,"OOB err = ",
                  round(ranger$prediction.error, digits=3),
                  sep="")) #+
  # guides(fill=F) + scale_fill_gradient(low="red", high="blue")
}

all.traindf <- all %>% filter(gold_edge==F) %>%
  mutate(y = as.factor(model),
         class_pair=class12,
         simple_POS_pair=simple_POS_12,
         POS1 = XPOS1,
         POS2 = XPOS2,
         POS_pair = XPOS12) %>%
  select(c(y #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
           ,lin_dist,simple_POS_1,simple_POS_2,i1,i2,w1,w2
           ,class1,class2
           ,relation
           #,pmi_tril,pmi_triu,pmi_sum
           #,UPOS12,UPOS1,UPOS2
           #,class12,simple_POS_12,XPOS1,XPOS2,XPOS12 # these ones are just renamed
           ,class_pair,simple_POS_pair#,POS1,POS2,POS_pair # as these
  ))
library(pheatmap)
{# RF on discerning which
  ptm <- proc.time()
  n = 9800
  training.df = all.traindf %>% sample_n(n)
  rf <- ranger(
    y ~ ., local.importance = FALSE,
    mtry = 3, num.trees = 2000, min.node.size = 1, splitrule = "gini",
    importance = "impurity_corrected", save.memory = TRUE,
    data = training.df,
  )
  print(proc.time() - ptm)

  cm <- rf$confusion.matrix
  print(cm)
  pheatmap(cm, display_numbers = T, number_format = "%d", cluster_rows = F, cluster_cols = F, legend = F)
  gg_varimp(rf,"discern model for wrong edge ")
}





maketraindf_gold <- function(df){
  df %>% filter(gold_edge==T) %>%
    mutate(y = as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(y #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
             ,lin_dist,simple_POS_1,simple_POS_2,i1,i2,w1,w2
             ,class1,class2
             ,relation
             ,pmi_tril,pmi_triu,pmi_sum
             #,UPOS12,UPOS1,UPOS2
             #,class12,simple_POS_12,XPOS1,XPOS2,XPOS12 # these ones are just renamed
             ,class_pair,simple_POS_pair#,POS1,POS2,POS_pair # as these
    ))
}
maketraindf_pmi <- function(df){
  df %>% filter(pmi_edge_sum==T) %>%
    mutate(y = as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(y #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
             ,lin_dist,simple_POS_1,simple_POS_2,i1,i2,w1,w2
             ,class1,class2
             #,relation
             ,pmi_tril,pmi_triu,pmi_sum
             #,UPOS12,UPOS1,UPOS2
             #,class12,simple_POS_12,XPOS1,XPOS2,XPOS12 # these ones are just renamed
             ,class_pair,simple_POS_pair#,POS1,POS2,POS_pair # as these
    ))
}
maketraindf_all <- function(df){
  df %>%
    mutate(y = as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(y #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
             ,lin_dist,simple_POS_1,simple_POS_2,i1,i2,w1,w2
             ,class1,class2
             ,relation
             ,pmi_tril,pmi_triu,pmi_sum
             #,UPOS12,UPOS1,UPOS2
             #,class12,simple_POS_12,XPOS1,XPOS2,XPOS12 # these ones are just renamed
             ,class_pair,simple_POS_pair#,POS1,POS2,POS_pair # as these
    ))
}
maketraindf_norelation <- function(df){
  df %>%
    mutate(y = as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(y #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
             ,lin_dist,simple_POS_1,simple_POS_2,i1,i2,w1,w2
             ,class1,class2
             #,relation
             ,pmi_tril,pmi_triu,pmi_sum
             #,UPOS12,UPOS1,UPOS2
             #,class12,simple_POS_12,XPOS1,XPOS2,XPOS12 # these ones are just renamed
             ,class_pair,simple_POS_pair#,POS1,POS2,POS_pair # as these
    ))
}

model_list=list(bert,
                xlnet,
                xlm ,
                dbert,
                bart,#gpt2,
                w2v )
model_list1=map(model_list,filter,lin_dist>1)

plotvarimp<-function(input.df){
  ptm <- proc.time()
  n = 9800
  df = maketraindf_pmi(input.df)
  training.df = df %>% sample_n(n)
  rf <- ranger(
    y ~ ., local.importance = FALSE,
    mtry = 3, num.trees = 2000, min.node.size = 1, splitrule = "gini",
    importance = "impurity_corrected", save.memory = TRUE,
    data = training.df,
  )
  print(proc.time() - ptm)

  cm <- rf$confusion.matrix
  print(cm)
  return(gg_varimp(rf,paste(input.df$model[[1]], "")))
}
plotvarimp(bert %>% filter(lin_dist>1))


p.vi <-exec("ggarrange",plotlist=map(model_list,plotvarimp),
            ncol=3,nrow=2, common.legend = TRUE, legend="bottom") %>%
  annotate_figure(bottom=text_grob("Variable importance - PMI subset"))
p.vi1<-exec("ggarrange",plotlist=map(model_list1,plotvarimp),
            ncol=3,nrow=2, common.legend = TRUE, legend="bottom") %>%
  annotate_figure(bottom=text_grob("Variable importance - PMI subset, len>1"))

pdp<-function(input.df){
  #' Partial dependence plot
  ptm <- proc.time()
  n = 9000
  df = maketraindf_pmi(input.df)
  training.df = df %>% sample_n(n)
  rf <- ranger(
    y ~ ., local.importance = FALSE,
    mtry = 3, num.trees = 2000, min.node.size = 1, splitrule = "gini",
    importance = "none", save.memory = TRUE, probability = FALSE,
    data = training.df,
  )
  print(proc.time() - ptm)

  cm <- rf$confusion.matrix
  print(cm)

  test.df = df #%>% sample_n(n)
  prediction = predict(rf, data = test.df)
  test.df$prediction <- prediction$predictions#['TRUE']  # if probability = TRUE
  # p<-test.df %>% ggplot(aes(x=pmi_sum,y=prediction)) + geom_point(alpha=0.5) +
  #   ggtitle(input.df$model[[1]])

  p<-test.df %>% select(y,prediction,lin_dist,pmi_sum) %>%
    ggplot(aes(x=lin_dist,y=pmi_sum,colour=prediction)) +
    scale_x_continuous(trans='identity',limits = c(0.5,6.5))+
    scale_y_continuous(trans='identity',limits = c(-5,30))+
    geom_jitter(alpha=0.05) +
    labs(x="arc length",y="CPMI")+
    ggtitle(input.df$model[[1]])
  return(p)
  # partial(rf, pred.var="lin_dist", plot = TRUE, rug = TRUE, plot.engine = "ggplot2")
  # pdp <- autoplot(pd)
  # return(gg_varimp(rf,"BERT testing"))
}
theme_set(theme_minimal())
theme_update()
# Ex. see the difference:
# pdp(bert)
# pdp(bert %>% filter(lin_dist>1))

p.pdp <-exec("ggarrange",plotlist=map(model_list,pdp),
             ncol=3,nrow=2, common.legend = TRUE, legend="bottom") %>%
  annotate_figure(bottom=text_grob("Random forest predictions"))
p.pdp1<-exec("ggarrange",plotlist=map(model_list1,pdp),
             ncol=3,nrow=2, common.legend = TRUE, legend="bottom") %>%
  annotate_figure(bottom=text_grob("Random forest predictions, len>1"))

pdp.comparison<-function(input.df){
  #' Plot to compare with partial dependence plot
  n = 9000
  df = maketraindf_pmi(input.df)
  test.df = df #%>% sample_n(n)
  p<-test.df %>% select(y,lin_dist,pmi_sum) %>%
    ggplot(aes(x=lin_dist,y=pmi_sum,colour=y)) +
    scale_x_continuous(trans='identity',limits = c(0.5,6.5))+
    scale_y_continuous(trans='identity',limits = c(-5,30))+
    geom_jitter(alpha=0.05) +
    labs(x="arc length",y="CPMI",colour="true label")+
    ggtitle(input.df$model[[1]])
  return(p)
  # partial(rf, pred.var="lin_dist", plot = TRUE, rug = TRUE, plot.engine = "ggplot2")
  # pdp <- autoplot(pd)
  # return(gg_varimp(rf,"BERT testing"))
}

p.pdpc <-exec("ggarrange",!!!map(model_list,pdp.comparison),
              ncol=3,nrow=2, common.legend = TRUE, legend="bottom") %>%
  annotate_figure(bottom=text_grob("Actual values"))
p.pdpc1<-exec("ggarrange",!!!map(model_list1,pdp.comparison),
              ncol=3,nrow=2, common.legend = TRUE, legend="bottom") %>%
  annotate_figure(bottom=text_grob("Actual values, len>1"))

ggsave("pdp.pdf",  plot=p.pdp,  width=14,height=9,units="in")
ggsave("pdp1.pdf", plot=p.pdp1, width=14,height=9,units="in")
ggsave("pdpc.pdf", plot=p.pdpc, width=14,height=9,units="in")
ggsave("pdpc1.pdf",plot=p.pdpc1,width=14,height=9,units="in")

# pdp.comparison(bert)























#####
#####

runrf <- function(df,name,mtry=3,min.node.size=1){
  ptm <- proc.time()
  training.df = df
  # ranger.fit = train(
  #   y = training.df$acc,
  #   x = training.df %>% select(-acc),
  #   method = 'ranger',
  #   num.trees = 500,
  #   tuneGrid = expand.grid(mtry = c(3,4,5,6),
  #                          splitrule = c("gini"),
  #                          min.node.size = c(1,2)),
  #   trControl = trainControl(
  #     search = "grid", method = "cv", number = 5, verboseIter = TRUE)
  # )
  # print(ranger.fit)
  # print(ranger.fit$bestTune)
  # Run best fit model
  rf <- ranger(
    y ~ ., local.importance = FALSE,
    mtry = mtry, num.trees = 2000, min.node.size = min.node.size, splitrule = "gini",
    importance = "impurity_corrected", save.memory = TRUE,
    data = training.df,
  )
  cm <- rf$confusion.matrix
  print(proc.time() - ptm)
  # print(importance_pvalues(rf, method = "janitza"))
  print(cm)
  return(gg_varimp(rf,name))
}


p.bert.gold <- runrf(maketraindf_gold(bert), "gold subset, BERT\n" ,mtry=3,min.node.size=1)
p.xlnet.gold <- runrf(maketraindf_gold(xlnet),"gold subset, XLNet\n",mtry=3,min.node.size=1)
p.xlm.gold <- runrf(maketraindf_gold(xlm),  "gold, XLM\n"         ,mtry=3,min.node.size=1)
grid.arrange(p.bert.gold,p.xlnet.gold,p.xlm.gold,nrow=1)

p.bert.pmi <- runrf(maketraindf_pmi(bert), "PMI subset, BERT\n"  ,mtry=3,min.node.size=1)
p.xlnet.pmi <- runrf(maketraindf_pmi(xlnet),"PMI subset, XLNet\n" ,mtry=3,min.node.size=1)
p.xlm.pmi <- runrf(maketraindf_pmi(xlm),  "PMI subset, XLM\n"   ,mtry=3,min.node.size=1)
grid.arrange(p.bert.pmi,p.xlnet.pmi,p.xlm.pmi,nrow=1)

p.bert.all <- runrf(maketraindf_all(bert),  "all data, BERT\n"  ,mtry=3,min.node.size=1)
p.xlnet.all <- runrf(maketraindf_all(xlnet), "all data, XLNet\n",mtry=3,min.node.size=1)
p.xlm.all <- runrf(maketraindf_all(xlm),   "all data, XLM\n"    ,mtry=3,min.node.size=1)
grid.arrange(p.bert.all,p.xlnet.all,p.xlm.all,nrow=1)

p.bert.norelation  <- runrf(maketraindf_norelation(bert),  "all data, BERT\n"   ,mtry=3,min.node.size=1)
p.xlnet.norelation <- runrf(maketraindf_norelation(xlnet), "all data, XLNet\n"  ,mtry=3,min.node.size=1)
p.xlm.norelation   <- runrf(maketraindf_norelation(xlm),   "all data, XLM\n"    ,mtry=3,min.node.size=1)
grid.arrange(p.bert.norelation, p.xlnet.norelation, p.xlm.norelation,nrow=1)




























## Experimenting with the symmetric difference, subsetting the data only for pairs where acc=FALSE,
## predicting which model

symdif <- function(df){
  df <- df %>% mutate(acc=gold_edge==pmi_edge_sum)
  df.gold <- df %>% filter(gold_edge==T)
  df.pmi <- df %>% filter(pmi_edge_sum==T)
  df.union <- rbind(df.gold, df.pmi)
  df.symdif <- df.union %>% filter(!(gold_edge==T & pmi_edge_sum==T)) %>%
    mutate(which = factor(
      case_when(gold_edge == T    ~ "gold",
                pmi_edge_sum == T ~ "pmi",
                TRUE ~ "error!")))
  return(df.symdif)
}

xlnet.symdif <- symdif(xlnet)
bert.symdif <- symdif(bert)
xlm.symdif <- symdif(xlm)
gpt2.symdif <- symdif(gpt2)
dbert.symdif <- symdif(dbert)

maketraindf_norelation_nopmi <- function(df){
  df %>%
    mutate(y = as.factor(which),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(y #,gold_edge,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
             ,sentence_index
             ,lin_dist,simple_POS_1,simple_POS_2,i1,i2,w1,w2
             ,class1,class2
             #,relation
             #,pmi_tril,pmi_triu,pmi_sum
             #,UPOS12,UPOS1,UPOS2
             #,class12,simple_POS_12,XPOS1,XPOS2,XPOS12 # these ones are just renamed
             ,class_pair,simple_POS_pair#,POS1,POS2,POS_pair # as these
    ))
}

p.bert.symdif <- runrf(maketraindf_norelation_nopmi(bert.symdif),"Sym diff, BERT\n"  ,mtry=3,min.node.size=1)
p.xlnet.symdif <- runrf(maketraindf_norelation_nopmi(xlnet.symdif),"Sym diff, XLNet\n"  ,mtry=3,min.node.size=1)
p.xlm.symdif <- runrf(maketraindf_norelation_nopmi(xlm.symdif),"Sym diff, XLM\n"  ,mtry=3,min.node.size=1)
p.gpt2.symdif <- runrf(maketraindf_norelation_nopmi(gpt2.symdif),"Sym diff, GPT2\n"  ,mtry=3,min.node.size=1)
p.dbert.symdif <- runrf(maketraindf_norelation_nopmi(dbert.symdif),"Sym diff, DistilBERT\n"  ,mtry=3,min.node.size=1)

grid.arrange(p.bert.symdif, p.dbert.symdif,
             p.xlnet.symdif, p.xlm.symdif,
             nrow=2)


symdif.df <- bind_rows(bert.symdif %>% mutate(model="BERT"),
                       xlnet.symdif %>% mutate(model="XLNet"),
                       xlm.symdif %>% mutate(model="XLM"),
                       gpt2.symdif %>% mutate(model="GPT2"),
                       dbert.symdif %>% mutate(model="DistilBERT"))
symdif.df %>% # sanity check: pmi is a good predictor of pmi-dependencyhood, as expected
  ggplot(aes(x=factor(which),y=pmi_sum,colour=model)) + geom_boxplot()


## Validation of RF ####

importance_pvalues(rf, formula = acc ~ ., method = "altmann",
                   num.permutations = 50, # should be 50-100 for reliable results
                   data = training.df)
importance_pvalues(rf, method = "janitza")


# Logistic regression ####
library(detectseparation)
lm.dist <- function(df){return(glm(acc ~ lin_dist, data = df, family=binomial("logit"), method="detect_separation"))}
lm.pmi <-  function(df){return(glm(acc ~ pmi_sum,  data = df, family=binomial("logit"), method="detect_separation"))}

bert.lm_gold.dist  <- lm.dist(bert %>% filter(gold_edge==T))
xlnet.lm_gold.dist <- lm.dist(xlnet %>% filter(gold_edge==T))
xlm.lm_gold.dist   <- lm.dist(xlm %>% filter(gold_edge==T))

bert.lm_gold.pmi   <- lm.pmi(bert %>% filter(gold_edge==T))
xlnet.lm_gold.pmi  <- lm.pmi(xlnet %>% filter(gold_edge==T))
xlm.lm_gold.pmi    <- lm.pmi(xlm %>% filter(gold_edge==T))

bert.lm_pmi.dist   <- lm.dist(bert %>% filter(pmi_edge_sum==T))
xlnet.lm_pmi.dist  <- lm.dist(xlnet  %>% filter(pmi_edge_sum==T))
xlm.lm_pmi.dist    <- lm.dist(xlm  %>% filter(pmi_edge_sum==T))

bert.lm_pmi.pmi    <- lm.pmi(bert  %>% filter(pmi_edge_sum==T))
xlnet.lm_pmi.pmi   <- lm.pmi(xlnet  %>% filter(pmi_edge_sum==T))
xlm.lm_pmi.pmi     <- lm.pmi(xlm  %>% filter(pmi_edge_sum==T))

dflist <- list(
  bert.lm_gold.dist,
  xlnet.lm_gold.dist,
  xlm.lm_gold.dist,
  bert.lm_gold.pmi,
  xlnet.lm_gold.pmi,
  xlm.lm_gold.pmi,
  bert.lm_pmi.dist,
  xlnet.lm_pmi.dist,
  xlm.lm_pmi.dist,
  bert.lm_pmi.pmi,
  xlnet.lm_pmi.pmi,
  xlm.lm_pmi.pmi)

r2s <- sapply(dflist,function(x){unname(r2_nagelkerke(x))})
coeffs <- sapply(dflist,coef)
allcoeffs <- sapply(dflist,function(x){coef(summary(x))}) # linearized. one per column
# for example, the first column is the linearized version of:
coef(summary(bert.lm_gold.dist))

lm.distpmi <- function(df){return(glm(acc ~ lin_dist*pmi_sum, data = df, family=binomial("logit"),method="detect_separation"))}
bert.lm_gold.distpmi  <- lm.distpmi(bert %>% filter(gold_edge==T))
xlnet.lm_gold.distpmi <- lm.distpmi(xlnet %>% filter(gold_edge==T))
xlm.lm_gold.distpmi   <- lm.distpmi(xlm %>% filter(gold_edge==T))
bert.lm_pmi.distpmi   <- lm.distpmi(bert %>% filter(pmi_edge_sum==T))
xlnet.lm_pmi.distpmi  <- lm.distpmi(xlnet  %>% filter(pmi_edge_sum==T))
xlm.lm_pmi.distpmi    <- lm.distpmi(xlm  %>% filter(pmi_edge_sum==T))

dflist.with_interaction <- list(
  bert.lm_gold.distpmi,
  xlnet.lm_gold.distpmi,
  xlm.lm_gold.distpmi,
  bert.lm_pmi.distpmi,
  xlnet.lm_pmi.distpmi,
  xlm.lm_pmi.distpmi  )

r2s.2 <- sapply(dflist2,function(x){unname(r2_nagelkerke(x))})
coeffs.2 <- sapply(dflist2,coef)
regrtab <- sapply(dflist2,function(x){coef(summary(x))})
rownames(regrtab) <- c("coef      (Intercept)","coef      lin_dist.","coef      pmi_sum.", "coef      lin_dist:pmi_sum",
                       "Std.Error (Intercept)","Std.Error lin_dist","Std.Error  pmi_sum.", "Std.Error lin_dist:pmi_sum",
                       "pmi_sum   (Intercept)","pmi_sum   lin_dist.","pmi_sum    z value", "pmi_sum   lin_dist:pmi_sum",
                       "((Intercept)","lin_dist:pmi_sum .","pmi_sum.", "lin_dist:pmi_sumPr(>|z|))")
colnames(regrtab) <-
  c("bert.lm_gold.distpmi",
    "xlnet.lm_gold.distpmi",
    "xlm.lm_gold.distpmi",
    "bert.lm_pmi.distpmi",
    "xlnet.lm_pmi.distpmi",
    "xlm.lm_pmi.distpmi")
regrtab
coef(summary(bert.lm_gold.distpmi))

xlnet %>% filter(pmi_edge_sum==T) %>% count()

