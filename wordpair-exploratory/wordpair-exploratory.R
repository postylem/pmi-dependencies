library(tidyverse)
# setwd("/Users/j/McGill/PhD-miasma/pmi-dependencies/wordpair-exploratory/")

## Exploratory Plotting ####

prepare_by_relation <- function(dataframe,length_greater_than=0){
  #' Prepare csv as df data grouped by 'relation'
  relation_len = dataframe %>% filter(!is.na(relation), lin_dist>length_greater_than) %>% 
    group_by(relation) %>% summarise(medlen=median(lin_dist), meanlen=mean(lin_dist), n=n())
  dataframe = dataframe %>% filter(!is.na(relation), lin_dist>length_greater_than) %>% 
    mutate(acc=gold_edge==pmi_edge_sum) %>% 
    group_by(relation,acc) %>% summarise(n=n(), medlen=median(lin_dist), meanlen=mean(lin_dist)) %>% 
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n,medlen,meanlen), values_fill = list(n = 0)) %>% 
    left_join(relation_len, by="relation") %>% mutate(pct_acc = n_pmiTRUE/n) 
  return(dataframe)
}

xlnet.relation <- prepare_by_relation(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.relation <- prepare_by_relation(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.relation <- prepare_by_relation(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))

# By-model basic plots
# bert.relation %>% ggplot(aes(x=reorder(relation,pct_acc), y=pct_acc)) + coord_flip() +
#   geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("BERT large")
# xlnet.relation %>%  ggplot(aes(x=reorder(relation,pct_acc), y=pct_acc)) + coord_flip() +
#   geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("XLNet base")
# xlm.relation %>%  ggplot(aes(x=reorder(relation,pct_acc), y=pct_acc)) + coord_flip() +
#   geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("XLM MLM EN 2048")


# All three models in one df
join_three <- function(df1, df2, df3, 
                       by=c("n","relation","medlen","meanlen"), 
                       suffixes=c(".BERT",".XLNet",".XLM")){
  return( 
    full_join(df1,df2,by=by,suffix=c(".BERT",".XLNet")) %>% 
    full_join(rename_at(df3, vars(-by), function(x){paste0(x,suffixes[3])}), by=by) %>%  
    pivot_longer(cols = -by, names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)"))
}

three.relation <- join_three(bert.relation,xlnet.relation,xlm.relation)

# three.relation <- 
#   full_join(bert.relation,xlnet.relation,
#             by=c("n","relation","medlen","meanlen"),
#             suffix=c(".BERT",".XLNet")) %>% 
#   full_join(rename_at(xlm.relation, vars(-c(n,relation,medlen,meanlen)), function(x){paste0(x,".XLM")}),
#             by=c("relation","n","medlen","meanlen")) %>%  
#   pivot_longer(cols = -c(n,relation,medlen,meanlen), 
#                names_to = c(".value", "model"), names_pattern = "(.*)\\.(.*)")


# A plot exploring accuracy by relation with respect to linear distance, model, and n
three.relation %>%  filter(n>50) %>% 
  ggplot(aes(y=pct_acc, x=reorder(relation, desc(pct_acc)))) + 
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="blue") +
  geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="blue") +  # to print n
  annotate("text",x=Inf,y=-Inf, label="mean\narclength", size=3, hjust=0.5, vjust=0) +
  geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3) +  
  geom_line(aes(group=relation), colour="grey") + 
  geom_point(aes(size=n, colour=model), alpha=0.8) + 
  coord_flip(clip = "off") +
  theme(legend.position="top", plot.margin = ggplot2::margin(2, 50, 2, 2, "pt"),
        axis.ticks = element_blank()) +
  ylab("percent PMI arc = gold arc") + 
  xlab("gold dependency label (ordered by mean accuracy)") + 
  ggtitle("Comparing % acc of XLNet base, BERT large, and XLM, by gold label (n>50)") 


## same, only for arc-length ≥ 1 ####

xlnet.relation.gt1 <- prepare_by_relation(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"),
                                          length_greater_than = 1)
bert.relation.gt1 <- prepare_by_relation(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"),
                                         length_greater_than = 1)
xlm.relation.gt1 <- prepare_by_relation(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"),
                                        length_greater_than = 1)
three.relation.gt1 <- join_three(bert.relation.gt1,xlnet.relation.gt1,xlm.relation.gt1)

three.relation.gt1 %>%  filter(n>49) %>% 
  ggplot(aes(y=pct_acc, x=reorder(relation, meanlen))) + 
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="blue") +
  geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="blue") +  # to print n
  annotate("text",x=Inf,y=-Inf, label="mean\narclength", size=3, hjust=0.5, vjust=0) +
  geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3) +  
  geom_line(aes(group=relation), colour="grey") + 
  geom_point(aes(size=n, colour=model), alpha=0.8) + 
  coord_flip(clip = "off") +
  theme(legend.position="top", plot.margin = ggplot2::margin(2, 50, 2, 2, "pt"),
        axis.ticks = element_blank()) +
  ylab("percent PMI arc = gold arc") + 
  xlab("gold dependency label (ordered by mean accuracy)") + 
  ggtitle("Comparing % acc of XLNet base, BERT large, and XLM, by gold label (n≥50, arclen>1)") 


## len ####

prepare_by_len <- function(dataframe){
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

xlnet.len <- prepare_by_len(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.len <- prepare_by_len(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.len <- prepare_by_len(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))

# By-model basic plots
bert.len %>% ggplot(aes(x=meanpmi, y=pct_acc)) + scale_x_log10() +
  geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("BERT large")
xlnet.len %>% ggplot(aes(x=meanpmi, y=pct_acc)) + scale_x_log10() +
  geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("XLNet base")
xlm.len %>% ggplot(aes(x=meanpmi, y=pct_acc)) + scale_x_log10() +
  geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("XLM MLM EN 2048")

# All three models in one df
three.len <- join_three(bert.len,xlnet.len,xlm.len, by = c("n","lin_dist"))

# A plot exploring accuracy by lin_dist
three.len %>%  filter(n>25) %>% 
  ggplot(aes(y=pct_acc, x=lin_dist)) + 
  geom_text(aes(label=n, y=Inf), hjust=0, size=3, colour="blue") +
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0, colour="blue") +
  geom_line(aes(group=lin_dist), colour="grey") + 
  geom_point(aes(size=n, colour=model), alpha=0.8) + 
  coord_flip(clip = "off") +
  theme(legend.position="top", plot.margin = ggplot2::margin(2, 50, 2, 2, "pt")
        ) + scale_x_continuous(breaks = seq(1, 23, by = 1)) +
  ylab("percent PMI arc = gold arc") + 
  xlab("arc length") + 
  ggtitle("Comparing pct acc of XLNet base, BERT large, and XLM, by arc length (n>25)") 



#########################################
#########################################
# Running a random forest classifier ####
#########################################
library(ranger)

gg_varimp <- function(ranger) {
  #' ggplot2 version of dotplot varimp.
  #' input ranger model object, 
  #' to plot variable importance
  ggplot(stack(ranger$variable.importance), 
         aes(x=reorder(ind,values), y=values, fill=values))+ 
    geom_point() +
    coord_flip() +
    ylab("Information value")+
    xlab("")+
    ggtitle(paste("Variable Importance, ",paste("n_obs = ",ranger$num.samples)))+
    guides(fill=F)+
    scale_fill_gradient(low="red", high="blue")
}

bert.raw <- read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv") %>%
  mutate(acc=gold_edge==pmi_edge_sum) %>% select(-c(gold_edge))
bert.raw$relation[is.na(bert.raw$relation)]<-"NONE"
bert.raw$UPOS12 <- factor(paste(bert.raw$UPOS1,bert.raw$UPOS2,sep = '-'))
bert.raw$XPOS12 <- factor(paste(bert.raw$XPOS1,bert.raw$XPOS2,sep = '-'))

## Making open/closed category feature #
open_xpos <- 
  c("JJ", "JJR", "JJS", "RB", "RBR", "RBS", "NN", "NNS", "NNPS", "NNP", "VBN", "VB", "VBZ", "VBG", "VBD", "VBP", "FW")

closed_xpos <- 
  c("DT", "CD", "CC", "IN", "PRP", "POS", "PRP$", "WP$", "PDT", "WDT", "WP")

bert.raw <- bert.raw %>% 
  mutate(cat1=if_else(XPOS1 %in% open_xpos,if_else(relation=="neg","CLOSED","OPEN"),
                      if_else(XPOS1 %in% closed_xpos,"CLOSED","?"))) %>% 
  mutate(cat2=if_else(XPOS2 %in% open_xpos,"OPEN",
                      if_else(XPOS2 %in% closed_xpos,"CLOSED","?")))
bert.raw$cat12 <- factor(paste(bert.raw$cat1,bert.raw$cat2,sep = '-'))
# Note, almost all occurences of neg have the first word as the negation (closed cat) word. 
bert.raw %>% filter(relation=="neg") %>% group_by(w1) %>% count
# We could refine this if we needed to, but it should be a good approximation...

##

bert.raw %>% ggplot(aes(x=cat12)) + geom_bar() + coord_flip()
XPOS_common_pairs = (bert.raw %>% group_by(XPOS12) %>% summarise(n=n()) %>% filter(n>2000))[[1]]
bert.raw %>% filter(XPOS12 %in% XPOS_common_pairs) %>%  ggplot(aes(x=reorder(XPOS12,cat12), fill=cat12)) + geom_bar() + coord_flip()


bert.raw %>% group_by(UPOS12) %>% summarise(n=n()) %>% filter(n>2000) %>% 
ggplot(aes(x=reorder(UPOS12,n),y=n)) + geom_bar(stat="identity") + coord_flip() +
  ggtitle("Most common (n>2000) UPOS pairs")
bert.raw %>% group_by(XPOS12) %>% summarise(n=n()) %>% filter(n>2000) %>% 
  ggplot(aes(x=reorder(XPOS12,n),y=n)) + geom_bar(stat="identity") + coord_flip() + 
  ggtitle("Most common (n>2000) XPOS pairs")


bert.raw %>% group_by(UPOS12) %>% summarise(n=n()) %>% filter(n>7000) 

bert.rf.no_pmiedge <- ranger(
  acc ~ .,
  importance = "permutation", save.memory = TRUE,
  data = bert.raw %>% select(-c(sentence_index,
                                pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu,
                                UPOS12,XPOS12
                                )))
sort(bert.rf.no_pmiedge$variable.importance, decreasing = T)
# relation     pmi_sum      pmi_tril     pmi_triu     lin_dist     i2 
# 0.1129536633 0.0677521808 0.0247886574 0.0227068496 0.0191836310 0.0040428731 
# UPOS2           i1        XPOS2        XPOS1        UPOS1 
# 0.0033115500 0.0027982239 0.0025109101 0.0020280266 0.0020246610
# cat2        cat12           w2           w1         cat1 
# 0.0018665519 0.0015147889 0.0012748560 0.0009567042 0.0008086439
gg_varimp(bert.rf.no_pmiedge)
#fourfoldplot(bert.rf.no_pmiedge$confusion.matrix,conf.level = 0)
#bert.rf.no_pmiedge$confusion.matrix

bert.rf.no_relation <- ranger(
  acc ~ .,
  importance = "permutation", save.memory = TRUE,
  data = bert.raw %>% select(-c(sentence_index,relation)))

bert.rf.no_relation2 <- ranger(
  acc ~ .,
  importance = "permutation", save.memory = TRUE,
  data = bert.raw %>% select(-c(sentence_index,relation,
                                pmi_edge_none,pmi_edge_tril,pmi_edge_triu,
                                pmi_tril,pmi_triu)))
sort(bert.rf.no_pmiedge$variable.importance, decreasing = T)
gg_varimp(bert.rf.no_relation2)

bert.rf.just_relation <- ranger(
  acc ~ relation,
  importance = "permutation", save.memory = TRUE,
  data = bert.raw %>% select(-c(sentence_index)))
gg_varimp(bert.rf.best2)
fourfoldplot(bert.rf.best2$confusion.matrix,conf.level = 0)
bert.rf.just_relation$variable.importance
bert.rf.just_relation$confusion.matrix
