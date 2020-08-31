library(tidyverse)
theme_set(theme_minimal())
#setwd("/Users/j/McGill/PhD-miasma/pmi-dependencies/R")

## Exploratory plotting ####

prepareppl <- function(csv){
  #' Prepare the raw data
  df = csv %>%
    mutate(sentence_logperplexity=-pseudo_loglik/sentence_length) %>%
    select(sentence_index, sentence_length, number_edges, pseudo_loglik, sentence_logperplexity,
           baseline_linear, baseline_random_proj, baseline_random_nonproj,
           # too many types of scores makes it messy. just choosing the 'sum' types
           projective.uuas.sum, projective.uuas.tril, projective.uuas.triu)  %>%
    pivot_longer(
      cols = -c(sentence_index, sentence_length, number_edges, pseudo_loglik, sentence_logperplexity),
      names_to = "score_method", values_to = "uuas")
  return(df)
}

# Main ones
# scores_bert_large_60 <-  prepareppl(read.csv("by_sentence/scores_bert-large-cased_pad60_2020-04-09-13-57.csv"))
# scores_xlnet_base_30 <-  prepareppl(read_csv("by_sentence/scores_xlnet-base-cased_pad30_2020-04-09-19-11.csv"))
# scores_xlm_60 <-         prepareppl(read_csv("by_sentence/scores_xlm-mlm-en-2048_pad60_2020-04-09-20-43.csv"))
# scores_bart_60 <-        prepareppl(read_csv("by_sentence/scores_bart-large_pad60_2020-04-27-01-20.csv")) # note tril is best.
# scores_dbert_60 <-        prepareppl(read_csv("by_sentence/scores_distilbert-base-cased_pad60_2020-04-29-19-35.csv"))
# scores_w2v <-            prepareppl(read_csv("by_sentence/scores_w2v_pad0_2020-05-17-00-47.csv"))

# scores_bert_base_60 <-  prepareppl(read.csv("by_sentence/scores_bert-base-cased_pad60_2020-03-31-17-30.csv"))
# scores_xlnet_large_30 <- prepareppl(read_csv("by_sentence/scores_xlnet-large-cased_pad30_2020-04-01-12-43.csv"))
# scores_gpt2_30 <-        prepareppl(read_csv("by_sentence/scores_gpt2_pad30_2020-04-24-13-45.csv"))      # note triu is best.

# With absolute value
scores_bert_large_60 <-  prepareppl(read.csv("by_sentence/scores_abs-loaded=bert-large-cased_pad60_2020-07-05-16-29.csv"))
scores_xlnet_base_30 <-  prepareppl(read_csv("by_sentence/scores_abs-loaded=xlnet-base-cased_pad30_2020-07-05-17-41.csv"))
scores_xlm_60 <-         prepareppl(read_csv("by_sentence/scores_abs-loaded=xlm-mlm-en-2048_pad60_2020-07-05-17-29.csv"))
scores_bart_60 <-        prepareppl(read_csv("by_sentence/scores_abs-loaded=bart-large_pad60_2020-07-05-16-06.csv"))
scores_dbert_60 <-        prepareppl(read_csv("by_sentence/scores_abs-loaded=distilbert-base-cased_pad60_2020-07-05-17-05.csv"))
scores_w2v <-            prepareppl(read_csv("by_sentence/scores_abs-loaded=w2v_pad0_2020-07-05-17-17.csv"))

scores_bert_base_30 <-  prepareppl(read.csv("by_sentence/scores_abs-loaded=bert-base-cased_pad30_2020-07-05-16-17.csv"))
scores_xlnet_large_30 <- prepareppl(read_csv("by_sentence/scores_abs-loaded=xlnet-large-cased_pad30_2020-07-05-17-53.csv"))


scores_bert_base_60$model <- 'BERT base'
scores_bert_base_30$model <- 'BERT base'
scores_bert_large_60$model <- 'BERT large'
scores_xlnet_base_30$model <- 'XLNet base'
scores_xlnet_large_30$model <- 'XLNet large'
scores_xlm_60$model <- 'XLM'
scores_bart_60$model <- 'Bart'
scores_gpt2_30$model <- 'GPT2'
scores_dbert_60$model <- 'DistilBERT'
scores_w2v$model <- 'Word2Vec'

scores_models = rbind(scores_dbert_60,
                      scores_bert_base_30,
                      scores_bert_large_60,
                      scores_xlnet_base_30,
                      scores_xlnet_large_30,
                      scores_xlm_60,
                      scores_bart_60,
                      scores_gpt2_30,
                      scores_w2v)
scores_models$model <- factor(scores_models$model,
                              levels = c("DistilBERT",
                                         "BERT base",
                                         "BERT large",
                                         "XLNet base",
                                         "XLNet large",
                                         "XLM",
                                         "Bart",
                                         "GPT2",
                                         "Word2Vec"))

scores_models %>% filter(sentence_index == 0,score_method == 'projective.uuas.sum')


scores_models %>% filter(sentence_length %in% 4:61,score_method == 'projective.uuas.sum') %>%
  ggplot(mapping = aes(x=sentence_length)) +
  scale_x_continuous(trans='log10') +
  geom_smooth(aes(y=uuas, colour=model), alpha = 1/5) +
  # ylim(0,1) + #facet_grid(model~.) +
  ggtitle("Accuracy vs sentence length (5-60)") +
  labs(x="sentence length",
       y="PMI dependency accuracy")

selected_models <- c('DistilBERT','BERT large','BERT base', 'XLNet large','XLNet base', 'XLM', 'Bart', 'Word2Vec')

library(stringr)
baselines_violin <- scores_models %>%
  filter(model=="XLM", str_detect(score_method,"^baseline")) %>%
  ggplot(aes(x=score_method,y=uuas)) +
  geom_violin() + geom_boxplot(width=0.1) +
  scale_x_discrete(labels=c('linear','random','random proj.')) +
  labs(x="baseline", y="dependency accuracy (UUAS)") +
  ggtitle("Baselines")
w2v_violin <- scores_models %>%
  filter(score_method == 'projective.uuas.sum', model=="Word2Vec") %>%
  # filter(str_detect(score_method,"^proj")) %>%
  ggplot(aes(x=model,
             # colour=score_method,
             y=uuas)) +
  geom_violin() + geom_boxplot(width=0.1) + # facet_wrap(~model,nrow = 1) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) + xlab("") +
  ggtitle("")
models_violin <- scores_models %>%
  filter(score_method == 'projective.uuas.sum', model %in% setdiff(selected_models,c("Word2Vec"))) %>%
  # filter(str_detect(score_method,"^proj")) %>%
  ggplot(aes(x=model,
             # colour=score_method,
             y=uuas)) +
  geom_violin() + geom_boxplot(width=0.1) + # facet_wrap(~model,nrow = 1) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) + xlab("model") +
  ggtitle("Contextualized PMI models")
library(gridExtra)
grid.arrange(arrangeGrob(baselines_violin,w2v_violin,models_violin,widths=c(3,1,7)))

## exploring the pseudo loglikelihood measure
##
scores_models <- scores_models %>%
  mutate(model = relevel(model, "Bart")) %>% mutate(model = relevel(model, "GPT2"))


scores_models %>% filter(model %in% selected_models) %>% filter(model!="Word2Vec") %>%
  filter(score_method %in% c("projective.uuas.sum"),
         sentence_length > 4,
         sentence_logperplexity<4*exp(1)
        ) %>%
  ggplot(aes(x=sentence_logperplexity,y=uuas)) + theme(legend.position = "top") +
  geom_point(alpha=.15) + geom_smooth(method = "lm", formula = y~x) +
  #geom_boxplot(aes(x=0),alpha=0.15) +
  scale_y_continuous(breaks=c(0,.5,1)) +
  facet_grid(model~.) +
  ggpmisc::stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~~")),
               label.x = "right", label.y = 1,
               formula = y~x, parse = TRUE, size = 3) +
  labs(x="log(pseudo-perplexity)", y="CPMI dependency accuracy") +
  ggtitle("Accuracy vs LM performance")


scores_models %>% filter(model!="Word2Vec") %>%
  filter(score_method %in% c("projective.uuas.sum"),
         sentence_length > 4,
         sentence_logperplexity<10*exp(1)
  ) %>%
  ggplot(aes(x=sentence_logperplexity,y=uuas)) + theme(legend.position = "top") +
  geom_point(alpha=.15) + geom_smooth(alpha=0.2, method = "lm", formula = y~x) +
  facet_grid(model~.) +
  ggpmisc::stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~~")),
                        label.x = "right", label.y = 1,
                        formula = y~x, parse = TRUE, size = 3) +
  labs(x="log(sentence perplexity)", y="PMI dependency accuracy") +
  ggtitle("PMI dependency accuracy vs LM performance")

permm<-lmPerm::lmp(formula = uuas ~ sentence_logperplexity,
            data = scores_models %>% filter(score_method == "projective.uuas.sum"))
summary(permm)
plot(permm,which=2)

# Correlation is low for all models.
cor.test(compare_models$`sentence_logperplexity_BERT base`,compare_models$`uuas_BERT base`,method="pearson")
cor.test(compare_models$`sentence_logperplexity_BERT large`,compare_models$`uuas_BERT large`,method="pearson")
cor.test(compare_models$`sentence_logperplexity_XLNet base`,compare_models$`uuas_XLNet base`,method="pearson")
cor.test(compare_models$`sentence_logperplexity_XLNet large`,compare_models$`uuas_XLNet large`,method="pearson")
cor.test(compare_models$`sentence_logperplexity_XLM`,compare_models$`uuas_XLM`,method="pearson")


coin::independence_test(formula = uuas ~ sentence_logperplexity, alternative = "less",
                        data = scores_models %>% filter(score_method == "projective.uuas.sum"))

compare_models <- scores_models %>%  filter(score_method == "projective.uuas.sum") %>%
  group_by(sentence_index,model) %>% summarise(uuas, sentence_logperplexity) %>%
  pivot_wider(names_from = model, values_from = c(uuas, sentence_logperplexity)) %>% ungroup()

# some examples...
compare_models %>% ggplot(aes(x=`uuas_BERT large`,y=`uuas_Word2Vec`)) +
  geom_point(alpha=0.1) +geom_smooth(method="lm")
compare_models %>% ggplot(aes(x=`uuas_Bart`,y=`uuas_Word2Vec`)) +
  geom_point(alpha=0.1) +geom_smooth(method="lm")
compare_models %>% ggplot(aes(x=`sentence_logperplexity_BERT large`,y=`sentence_logperplexity_XLM`)) +
  geom_point(alpha=0.1) + geom_smooth(method="lm") +
  scale_x_log10() + scale_y_log10()
compare_models %>% ggplot(aes(x=`sentence_logperplexity_BERT large`,y=`sentence_logperplexity_Bart`)) +
  geom_point(alpha=0.1) + geom_smooth(method="lm") +
  scale_x_log10() + scale_y_log10()

cor.test(compare_models$`uuas_BERT base`,compare_models$`uuas_BERT large`,method="pearson")

compare_uuas <- compare_models %>% filter(model %in% selected_models) %>%
  select(starts_with("uuas")) %>% set_names(~str_replace_all(., "uuas_", ""))
compare_ppl  <- compare_models %>% filter(model %in% selected_models) %>% filter(model!="Word2Vec") %>%
  select(starts_with("sentence_logperplexity")) %>% set_names(~str_replace_all(., "sentence_logperplexity_", ""))


library("Hmisc")
Hcor <- rcorr(as.matrix(compare_uuas))

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    pearson_r = (cormat)[ut],
    P = pmat[ut]
  )
}

flattenCorrMatrix(Hcor$r, Hcor$P)

library("corrplot")
corrplot(cor(compare_uuas),method = "ellipse")

pointsmooth <- function(data, mapping, method="loess", ...){
  p <- ggplot(data = data, mapping = mapping) +
    geom_point(alpha=0.1) +
    geom_smooth(colour="red",method=method, ...)
  p
}
corrcol <- function(data, mapping, method="p", use="pairwise", ...){
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

PerformanceAnalytics::chart.Correlation(compare_uuas, histogram=TRUE)
library("GGally")
ggpairs(compare_uuas,
        upper = list(continuous = wrap(corrcol, colour = "black")),
        lower = list(continuous = pointsmooth),
        diag = list(continuous = wrap("densityDiag", alpha=0.5)),
        title = "PMI depedency accuracy correlogram"
        )
ggpairs(compare_ppl %>% log(),
        upper = list(continuous = wrap(corrcol, colour = "black")),
        lower = list(continuous = pointsmooth),
        title = "Sentence log perplexity correlogram (log scale)"
)

