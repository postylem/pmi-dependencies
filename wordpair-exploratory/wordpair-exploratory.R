library(tidyverse)
library(gridExtra)
#setwd("/Users/j/McGill/PhD-miasma/pmi-dependencies/wordpair-exploratory/")

## Exploratory Plotting ####
##
##
bert <- read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv")
xlnet<- read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv")
xlm  <- read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv")


theme_set(theme_minimal())
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

xlnet.relation <- prepare_by_relation(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.relation <- prepare_by_relation(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.relation <- prepare_by_relation(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))



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


# A plot exploring accuracy by relation with respect to linear distance, model, and n
p.rel <-
three.relation %>%  filter(n>20) %>%
  ggplot(aes(y=pct_acc, x=reorder(relation, desc(meanlen)))) +
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="grey") +
  geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="grey") +  # to print n
  annotate("text",x=Inf,y=-Inf, label="mean\narclength", size=3, hjust=0, vjust=0) +
  geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3) +
  geom_line(aes(group=relation), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 50, 2, 2, "pt"),
        axis.ticks = element_blank()) +
  ylab("recall (# PMI arc = gold arc)/(# gold arcs)") +
  xlab("gold dependency label (ordered by mean arc length)") +
  ggtitle("Accuracy (recall) by gold label (n>20)")


## same, only for arc-length ≥ 1 ####

xlnet.relation.gt1 <- prepare_by_relation(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"),
                                          length_greater_than = 1)
bert.relation.gt1 <- prepare_by_relation(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"),
                                         length_greater_than = 1)
xlm.relation.gt1 <- prepare_by_relation(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"),
                                        length_greater_than = 1)
three.relation.gt1 <- join_three(bert.relation.gt1,xlnet.relation.gt1,xlm.relation.gt1,
                                 by=c("n","meanlen"),
                                 suffixes=c(".BERT",".XLNet",".XLM"))
p.rel.gt1<-
three.relation.gt1 %>%  filter(n>20) %>%
  ggplot(aes(y=pct_acc, x=reorder(relation, desc(meanlen)))) +
  annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0,colour="grey") +
  geom_text(aes(label=paste("",n,sep=""),y=Inf), hjust=0, size=3, colour="grey") +  # to print n
  annotate("text",x=Inf,y=-Inf, label="mean\narclength", size=3, hjust=0.5, vjust=0) +
  geom_text(aes(label=round(meanlen, digits=1), y=-Inf), hjust=0, size=3) +
  geom_line(aes(group=relation), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 50, 2, 2, "pt"),
        axis.ticks = element_blank()) +
  ylab("recall (# PMI arc = gold arc)/(# gold arcs)") +
  xlab("gold dependency label (ordered by mean arc length)") +
  ggtitle("Accuracy (recall) by gold label (n>20, arclen>1)")

grid.arrange(p.rel,p.rel.gt1,ncol=2)

# PMI value ####
#

three.relation %>%  filter(n>200) %>%
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
  ggtitle("Comparing mean pmi of edges in XLNet base, BERT large, and XLM, by gold label (n>50)")

three.relation %>% filter(n>50) %>%
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
  ylab("recall (# PMI arc = gold arc)/(# gold arcs)") +
  xlab("mean PMI value per dependency label") +
  ggtitle("Comparing mean pmi value with accuracy in XLNet base, BERT large, and XLM, by gold label(n>50)")


## len ####

# quick histograms
gold.len <- bert %>% filter(gold_edge==T) %>% group_by(lin_dist) %>% count
bert.len <- bert %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
xlnet.len <- xlnet %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count
xlm.len <- xlm %>% filter(pmi_edge_sum==T) %>% group_by(lin_dist) %>% count

ggcolhue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
modelcols<-ggcolhue(3)
goldcol<-ggcolhue(5)[[2]]

bert %>% filter(gold_edge==T) %>% ggplot(aes(x=lin_dist)) + geom_histogram() + scale_x_log10()
hgold <- gold.len %>% ggplot(aes(x=lin_dist,y=n)) + geom_col(fill=goldcol) + xlim(0,15) + xlab("gold arc length")
hbert <- bert.len %>% ggplot(aes(x=lin_dist,y=n)) + geom_col(fill=modelcols[[1]]) + xlim(0,15) + xlab("PMI arc length (BERT)")
hxlnet <- xlnet.len %>% ggplot(aes(x=lin_dist,y=n)) + geom_col(fill=modelcols[[2]]) + xlim(0,15) + xlab("PMI arc length (XLNet)")
hxlm <- xlm.len %>% ggplot(aes(x=lin_dist,y=n)) + geom_col(fill=modelcols[[3]]) + xlim(0,15) + xlab("PMI arc length (XLM)")
#
# hgold <- bert %>% filter(gold_edge==T) %>% ggplot(aes(x=lin_dist)) + geom_histogram(breaks=seq(0,32)) + scale_x_continuous(trans = "log2")
# hbert <- bert %>% filter(pmi_edge_sum==T) %>% ggplot(aes(x=lin_dist)) + geom_histogram(breaks=seq(0,32)) + scale_x_continuous(trans = "log2")
# hxlnet <- xlnet %>% filter(pmi_edge_sum==T) %>% ggplot(aes(x=lin_dist)) +  geom_histogram(breaks=seq(0,32))  + scale_x_continuous(trans = "log2")
# hxlm <- xlm %>% filter(pmi_edge_sum==T) %>% ggplot(aes(x=lin_dist)) +  geom_histogram(breaks=seq(0,32))  + scale_x_continuous(trans = "log2")
grid.arrange(hgold,hbert,hxlnet,hxlm,ncol=4)

gold.len[1,]$n/(bert %>% filter(gold_edge==T) %>%  count())
bert.len[1,]$n/(bert %>% filter(pmi_edge_sum==T) %>%  count())
xlnet.len[1,]$n/(xlnet %>% filter(pmi_edge_sum==T) %>%  count())
xlm.len[1,]$n/(xlm %>% filter(pmi_edge_sum==T) %>%  count())


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

xlnet.len.gold <- prepare_by_len_gold(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.len.gold <- prepare_by_len_gold(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.len.gold <- prepare_by_len_gold(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))

# All three models in one df
three.len.gold <- join_three(bert.len.gold,xlnet.len.gold,xlm.len.gold,
                             by = c("n","lin_dist"),
                             suffixes=c(".BERT",".XLNet",".XLM"))

# A plot exploring accuracy by lin_dist
p.lin_dist <- three.len.gold %>%  filter(n>25) %>%
  ggplot(aes(y=pct_acc, x=lin_dist)) +
  geom_text(aes(label=n, y=Inf), hjust=0, size=2.5, colour="grey") +
  annotate("text",x=Inf,y=Inf, label="n", size=2.5, hjust=0, vjust=0, colour="grey") +
  geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical",legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 50, 2, 2, "pt")
  ) + scale_x_continuous(breaks = seq(1, 23, by = 2), minor_breaks = seq(1, 23, by = 1)) +
  ylab("recall (# PMI arc = gold arc)/(# gold arcs)") +
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

xlnet.len.pred <- prepare_by_len_pred(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.len.pred <- prepare_by_len_pred(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.len.pred <- prepare_by_len_pred(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))

# All three models in one df
three.len.pred <- join_three(bert.len.pred,xlnet.len.pred,xlm.len.pred,
                             by = c("n","lin_dist"),
                             suffixes=c(".BERT",".XLNet",".XLM"))

# A plot exploring accuracy by lin_dist
p.lin_dist.pred <- three.len.pred %>%  filter(n>25) %>%
  ggplot(aes(y=pct_acc, x=lin_dist)) +
  #geom_text(aes(label=n, y=Inf), hjust=0, size=3, colour="grey") +
  #annotate("text",x=Inf,y=Inf, label="n", size=3, hjust=0, vjust=0, colour="grey") +
  #geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n, colour=model), alpha=0.8) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical", legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 2, 2, 2, "pt")
  ) + scale_x_continuous(breaks = seq(1, 35, by = 4), minor_breaks = seq(1, 35, by = 1)) +
  ylab("precision (# PMI arc = gold arc)/(# PMI arcs)") +
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

xlnet.len.all <- prepare_by_len_all(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.len.all <- prepare_by_len_all(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.len.all <- prepare_by_len_all(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))
three.len.all <- join_three(bert.len.all,xlnet.len.all,xlm.len.all,
                            by = c("n","lin_dist"),
                            suffixes=c(".BERT",".XLNet",".XLM"))

# A plot exploring accuracy by lin_dist
three.len.all %>%  filter(n>200) %>%
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


p.pmi.gold <- three.len.gold %>%  filter(lin_dist<15) %>%
  ggplot(aes(y=pct_acc, x=meanpmi)) +
  geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n,colour=model), alpha=0.8) +
  # geom_text(aes(label=lin_dist,colour=model,x=9-lin_dist/3), hjust=0, size=3) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical", legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 50, 2, 2, "pt")
  ) + scale_x_continuous(breaks = seq(1, 23, by = 1)) +
  ylab("recall (# PMI arc = gold arc)/(# gold arcs)") +
  xlab("mean PMI value") +
  ggtitle("Accuracy (recall) by mean PMI (arc length < 15)")
p.pmi.pred <- three.len.pred %>%  filter(lin_dist<15)%>%
  ggplot(aes(y=pct_acc, x=meanpmi)) +
  geom_line(aes(group=lin_dist), colour="grey") +
  geom_point(aes(size=n,colour=model), alpha=0.8) +
  # geom_text(aes(label=lin_dist, colour=model,x=8.5-lin_dist/3), hjust=0, size=3) +
  coord_flip(clip = "off") +
  theme(legend.position="top", legend.box="vertical", legend.margin = margin(),
        plot.margin = ggplot2::margin(2, 50, 2, 2, "pt")
  ) + scale_x_continuous(breaks = seq(1, 23, by = 1)) +
  ylab("precision (# PMI arc = gold arc)/(# PMI arcs)") +
  xlab("mean PMI value") +
  ggtitle("Accuracy (precision) by mean PMI (arc length < 15)")

grid.arrange(p.pmi.gold,p.pmi.pred,ncol=2)




# POS ###################################

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

prepare_proportion <- function(df){
  df <- df %>%
    mutate(acc=gold_edge==pmi_edge_sum,
           simple_POS_1 = make_simple_pos(XPOS1),
           simple_POS_2 = make_simple_pos(XPOS2),
           simple_POS_12 = paste(simple_POS_1,simple_POS_2,sep = '-'))
  df$relation[is.na(df$relation)]<-"NONE"
  df$UPOS12 <- factor(paste(df$UPOS1,df$UPOS2,sep = '-'))
  df$XPOS12 <- factor(paste(df$XPOS1,df$XPOS2,sep = '-'))
  return(df)
}

bert <- prepare_proportion(bert)
xlnet<- prepare_proportion(xlnet)
xlm  <- prepare_proportion(xlm)

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
  "XLM"  = sapply(POSpairs, count_proportion_in_order_edge, dataframe = xlm, edge = "pmi_edge_sum")) %>%
  pivot_longer(-c(POSpair), names_to = "arctype", values_to = "num")%>%
  mutate(arctype = fct_relevel(factor(arctype),"baseline","gold","BERT","XLNet","XLM"))


POSpair_proportion.df <- tibble(
  "POSpair" = sapply(POSpairs, paste, collapse = "-"),
  "baseline"= sapply(POSpairs, proportion_in_order, dataframe = bert),
  "gold" = sapply(POSpairs, proportion_in_order_edge, dataframe = bert, edge = "gold_edge"),
  "BERT" = sapply(POSpairs, proportion_in_order_edge, dataframe = bert, edge = "pmi_edge_sum"),
  "XLNet"= sapply(POSpairs, proportion_in_order_edge, dataframe = xlnet, edge = "pmi_edge_sum"),
  "XLM"  = sapply(POSpairs, proportion_in_order_edge, dataframe = xlm, edge = "pmi_edge_sum")) %>%
  pivot_longer(-c(POSpair), names_to = "arctype", values_to = "proportion")%>%
  mutate(arctype = fct_relevel(factor(arctype),"baseline","gold","BERT","XLNet","XLM"))

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

## Making open/closed class feature #
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

bert <-add_class_predictor(bert)
xlnet<-add_class_predictor(xlnet)
xlm  <-add_class_predictor(xlm)

make_classdf_gold <- function(df,name){
  return(df %>% filter(gold_edge==TRUE) %>% group_by(class12,acc) %>% summarise(n = n()) %>%
    pivot_wider(names_from = acc, values_from = n, values_fill = list(n = 0)) %>% mutate(n=`FALSE`+`TRUE`, acc = `TRUE`/n, model=name) %>% ungroup())
}
make_classdf_pmi <- function(df,name){
  return(df %>% filter(pmi_edge_sum==TRUE) %>% group_by(class12,acc) %>% summarise(n = n()) %>%
           pivot_wider(names_from = acc, values_from = n, values_fill = list(n = 0)) %>% mutate(n=`FALSE`+`TRUE`, acc = `TRUE`/n, model=name) %>% ungroup())
}

three.classdf.gold <-
  bind_rows(purrr::map2(
    list(bert,xlnet,xlm),
    list("BERT","XLNet","XLM"),
    make_classdf_gold)) %>% arrange(class12)
three.classdf.pmi <-
  bind_rows(purrr::map2(
    list(bert,xlnet,xlm),
    list("BERT","XLNet","XLM"),
    make_classdf_pmi)) %>% arrange(class12)

# barplots
four_class_pairs = c("CLOSED-CLOSED","OPEN-CLOSED","CLOSED-OPEN","OPEN-OPEN")

p.class.gold <- three.classdf.gold %>% filter(class12 %in% four_class_pairs) %>%
  mutate(class12 = factor(class12, levels=four_class_pairs)) %>%
  ggplot(aes(x=class12,y=acc,fill=model)) +
  geom_bar(stat='identity', position='dodge') + coord_cartesian(clip = "off") +
  geom_text(aes(label=n,y=-Inf), colour="grey",show.legend = F,
            vjust = 0, hjust=0.5, size = 3, angle=0) +
  annotate("text",x=Inf,y=-Inf, label="n total", size=3, hjust=0, vjust=0, colour="grey") +
  theme(plot.margin = ggplot2::margin(2, 30, 2, 2, "pt")) +
  labs(x="Class pair type", y = "recall (# PMI arc = gold arc)/(# gold arcs)") +
  ggtitle("Accuracy (recall) by word-class pair")
p.class.pmi <- three.classdf.pmi %>% filter(class12 %in% c("OPEN-CLOSED","OPEN-OPEN","CLOSED-CLOSED","CLOSED-OPEN")) %>%
  mutate(class12 = factor(class12, levels=four_class_pairs)) %>%
  ggplot(aes(x=class12,y=acc,fill=model)) +
  geom_bar(stat='identity', position='dodge',show.legend = F) + coord_cartesian(clip = "off") +
  geom_text(aes(label=n,y=-Inf, colour=model), show.legend = F,
            position = position_dodge(width = 1), vjust = 0, hjust=0.5, size = 3, angle=0) +
  annotate("text",x=Inf,y=-Inf, label="n total", size=3, hjust=0, vjust=0, colour="grey") +
  theme(plot.margin = ggplot2::margin(2, 30, 2, 2, "pt")) +
  labs(x="Class pair type", y = "precision (# PMI arc = gold arc)/(# PMI arcs)") +
  ggtitle("Accuracy (precision) by word-class pair")
gridExtra::grid.arrange(p.class.gold,p.class.pmi,ncol=2,widths=c(24,22))



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
# Gridsearch for best model
library(caret)
library(ranger)


maketraindf_gold <- function(df){
  df %>% filter(gold_edge==T) %>%
    mutate(acc=as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(acc #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
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
    mutate(acc=as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(acc #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
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
    mutate(acc=as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(acc #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
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
    mutate(acc=as.factor(acc),
           class_pair=class12,
           simple_POS_pair=simple_POS_12,
           POS1 = XPOS1,
           POS2 = XPOS2,
           POS_pair = XPOS12) %>%
    select(c(acc #,gold_edge,sentence_index,pmi_edge_sum,pmi_edge_none,pmi_edge_tril,pmi_edge_triu
             ,lin_dist,simple_POS_1,simple_POS_2,i1,i2,w1,w2
             ,class1,class2
             #,relation
             ,pmi_tril,pmi_triu,pmi_sum
             #,UPOS12,UPOS1,UPOS2
             #,class12,simple_POS_12,XPOS1,XPOS2,XPOS12 # these ones are just renamed
             ,class_pair,simple_POS_pair#,POS1,POS2,POS_pair # as these
    ))
}

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
    acc ~ ., local.importance = FALSE,
    mtry = mtry, num.trees = 2000, min.node.size = min.node.size, splitrule = "gini",
    importance = "impurity_corrected", save.memory = TRUE,
    data = training.df,
  )
  library(gridExtra)
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

rf$variable.importance

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

