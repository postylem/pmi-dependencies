library(tidyverse)

## Exploratory Plotting ####

prepare_by_relation <- function(dataframe){
  #' Prepare csv as df data grouped by 'relation'
  relation_len = dataframe %>% filter(!is.na(relation)) %>% 
    group_by(relation) %>% summarise(medlen=median(lin_dist), meanlen=mean(lin_dist), n=n())
  dataframe = dataframe %>% filter(!is.na(relation)) %>% 
    mutate(acc=gold_edge==pmi_edge_sum) %>% 
    group_by(relation,acc) %>% summarise(n=n(), medlen=median(lin_dist), meanlen=mean(lin_dist)) %>% 
    pivot_wider(names_from = acc, names_prefix = "pmi", values_from = c(n,medlen,meanlen), values_fill = list(n = 0)) %>% 
    left_join(relation_len, by="relation") %>% mutate(pct_acc = n_pmiTRUE/n) 
  return(dataframe)
}

xlnet.relation <- prepare_by_relation(read_csv("wordpair_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
bert.relation <- prepare_by_relation(read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv"))
xlm.relation <- prepare_by_relation(read_csv("wordpair_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))

# Some by-model basic plots
bert.relation %>% ggplot(aes(x=reorder(relation,pct_acc), y=pct_acc)) + coord_flip() +
  geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("BERT large")
xlnet.relation %>%  ggplot(aes(x=reorder(relation,pct_acc), y=pct_acc)) + coord_flip() +
  geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("XLNet base")
xlm.relation %>%  ggplot(aes(x=reorder(relation,pct_acc), y=pct_acc)) + coord_flip() +
  geom_point(aes(size=n)) + ylab("percent PMI arc = gold arc") + ggtitle("XLM MLM EN 2048")

# All three models in one df
three.relation <- 
  full_join(bert.relation,xlnet.relation,
            by=c("n","relation","medlen","meanlen"),
            suffix=c(".BERT",".XLNet")) %>% 
  full_join(rename_at(xlm.relation, vars(-c(n,relation,medlen,meanlen)), function(x){paste0(x,".XLM")}),
            by=c("relation","n","medlen","meanlen")) %>% 
  select(c(relation, n, medlen, meanlen, pct_acc.BERT, pct_acc.XLNet, pct_acc.XLM))

# Here's a plot exploring accuracy by relation 
# with respect to mean linear distance, model, and n
three.relation %>%  filter(n>100) %>% 
  pivot_longer(cols = c(pct_acc.BERT, pct_acc.XLNet, pct_acc.XLM), 
               values_to = "pct_acc", names_to = "model", names_prefix = "pct_acc.") %>% 
  ggplot(aes(y=pct_acc, x=reorder(relation, meanlen))) + 
  # geom_text(aes(label=paste("",n,sep=""),y=0), hjust=1, size=3, colour="blue") +  # to print n
  geom_text(aes(label=round(meanlen, digits=1), y=Inf), hjust=0, size=3) +  
  annotate("text",x=Inf,y=Inf, label="mean arclength", size=3, hjust=0.5, vjust=0) +
  geom_line(aes(group=relation), colour="grey") + 
  geom_point(aes(size=n, colour=model), alpha=0.8) + 
  coord_flip(clip = "off") +
  theme(legend.position="top", plot.margin = ggplot2::margin(2, 50, 2, 2, "pt")) +
  ylab("percent PMI arc = gold arc") + 
  xlab("gold dependency label (ordered by mean arc length)") + 
  ggtitle("Comparing pct acc of XLNet base, BERT large, and XLM, by gold label") 

# Fitting a random forest ####

# preparing the data, just using BERT for now
bert.raw <- read_csv("wordpair_bert-large-cased_pad60_2020-04-09-13-57.csv") %>% head(3000) %>% 
  mutate(acc=gold_edge==pmi_edge_sum) %>% select(-c(gold_edge))
# make some text/boolean predictors and response into factors, 
# so the random forest can use them
bert.raw$acc <- factor(bert.raw$acc)
bert.raw$UPOS1 <- factor(bert.raw$UPOS1)
bert.raw$UPOS2 <- factor(bert.raw$UPOS2)
bert.raw$XPOS1 <- factor(bert.raw$XPOS1)
bert.raw$XPOS2 <- factor(bert.raw$XPOS2)
bert.raw$relation <- factor(bert.raw$relation)

# Using party... ####
library(party)

bert.cforest = cforest(
  acc ~ i1 + i2 + lin_dist + pmi_sum + pmi_edge_sum + UPOS1 + UPOS2,
  data=bert.raw)
bert.cforest
bert.cforest.varimp = varimp(bert.cforest, conditional=TRUE)
dotplot(sort(bert.cforest.varimp))
bert.cforest.varimp

# Using randomForest... ####
library(randomForest)
bert.randomForest <- randomForest(
  acc ~ i1 + i2 + lin_dist + pmi_sum + pmi_edge_sum + UPOS1 + UPOS2,
  data=bert.raw)
bert.randomForest
varImpPlot(bert.randomForest)
