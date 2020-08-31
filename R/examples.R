library(tidyverse)
library(gridExtra)
theme_set(theme_minimal())
#setwd("/Users/j/McGill/PhD-miasma/pmi-dependencies/R")


## POS ###################################

make_simple_pos <- function(XPOS){
  return(
    case_when(
      XPOS %in% c("JJ", "JJR", "JJS")               ~ "Adjective",
      XPOS %in% c("NN", "NNS", "NNPS", "NNP",
                  "PRP", "WP", "EX")                ~ "Noun", # Personal and wh- pronouns and expletive 'there' included
      XPOS %in% c("RB", "RBR", "RBS","WRB")         ~ "Adverb",
      XPOS %in% c("VB", "VBZ", "VBN",
                  "VBG", "VBP", "VBD",
                  "MD")                             ~ "Verb", # Modal verbs included
      XPOS %in% c("POS","PRP$","WP$")               ~ "Possessive",
      XPOS %in% c("IN","TO")                        ~ "Preposition",
      XPOS %in% c("DT","PDT")                       ~ "Determiner",
      XPOS %in% c("CD")                             ~ "Cardinal",
      XPOS %in% c("WDT")                            ~ "Complementizer",
      XPOS %in% c("RP")                             ~ "Particle",
      XPOS %in% c("CC")                             ~ "Conjunction",
      TRUE ~ "other")
    )
}
## simple POS ####
prepare_POS <- function(df){
  df <- df %>%
    mutate(
      simple_POS_2 = make_simple_pos(XPOS2),
      simple_POS_1 = case_when(XPOS1=="TO" & simple_POS_2=="Verb" ~ "Particle",
                               TRUE ~ make_simple_pos(XPOS1)),
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
    "VB", "VBZ", "VBG", "VBD", "VBP", "FW", "RP", "WRB")
closed_xpos <-
  c("DT", "CD", "CC", "IN",
    "PRP", "POS", "PRP$", "WP$",
    "PDT", "WDT", "WP", "EX", "TO", "MD", "LS", "UH")

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
bert <- read_csv("by_wordpair/wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv")
xlnet<- read_csv("by_wordpair/wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv")
xlm  <- read_csv("by_wordpair/wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv")
bart <- read_csv("by_wordpair/wordpair_bart-large_pad60_2020-04-27-01-20.csv")
gpt2 <- read_csv("by_wordpair/wordpair_gpt2_pad30_2020-04-24-13-45.csv")
dbert<- read_csv("by_wordpair/wordpair_distilbert-base-cased_pad60_2020-04-29-19-35.csv")
w2v  <- read_csv("by_wordpair/wordpair_w2v_pad0_2020-05-17-00-47.csv")

dbert$model <- "DistilBERT"
xlnet$model <- "XLNet"
bert$model  <- "BERT"
xlm$model   <- "XLM"
bart$model  <- "Bart"
gpt2$model  <- "GPT2"
w2v$model   <- "Word2Vec"

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
all <- bind_rows(bert, xlnet,xlm,  bart, gpt2, dbert,w2v)


# Exploratory analysis
## Finding examples of bad correspondence ####

# gold
bert %>% filter(gold_edge==T) %>% group_by(w1, w2, lin_dist) %>% count() %>% arrange(desc(n))
bert %>%  filter(gold_edge==T) %>% group_by(simple_POS_1, simple_POS_2) %>% count() %>% arrange(desc(n))

# wrong
w2v %>% filter(gold_edge==F,pmi_edge_sum==T) %>% group_by(w1,w2,lin_dist) %>% count() %>% arrange(desc(n))
w2v %>% filter(gold_edge==F,pmi_edge_sum==T) %>% group_by(simple_POS_12,lin_dist) %>% count() %>% arrange(desc(n))

# right
w2v %>%  filter(gold_edge==T,pmi_edge_sum==T) %>% group_by(w1,w2,lin_dist) %>% count() %>% arrange(desc(n))


# PMI edges in distribution
pospos<-function(model,simple_POS_1, simple_POS_2){
  model %>%  filter(pmi_edge_sum==T) %>% group_by(simple_POS_1, simple_POS_2) %>% count() %>% ungroup() %>%
    mutate(simple_POS_1 = fct_reorder(simple_POS_1, n),
           simple_POS_2 = fct_reorder(simple_POS_2, n, .desc = T)) %>%
    ggplot(aes(x=simple_POS_2, y=simple_POS_1, size=n)) + geom_point() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust=0.5)) +
    ggtitle(model$model[[1]])
}
pospos(xlm, simple_POS_1, simple_POS_2)
