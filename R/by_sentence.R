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
           # too many types of scores makes it messy. just choosing the 'projective' types
           projective.uuas.sum, projective.uuas.tril, projective.uuas.triu)  %>%
    pivot_longer(
      cols = -c(sentence_index, sentence_length, number_edges, pseudo_loglik, sentence_logperplexity),
      names_to = "score_method", values_to = "uuas")
  return(df)
}

# With absolute value
scores_bert_large <-  prepareppl(read.csv("by_sentence/scores_abs-loaded=bert-large-cased_pad60_2020-08-09-06-12.csv"))
scores_xlnet_base <-  prepareppl(read_csv("by_sentence/scores_abs-loaded=xlnet-base-cased_pad30_2020-08-09-07-13.csv"))
scores_xlm <-         prepareppl(read_csv("by_sentence/scores_abs-loaded=xlm-mlm-en-2048_pad60_2020-08-09-07-00.csv"))
scores_bart <-        prepareppl(read_csv("by_sentence/scores_abs-loaded=bart-large_pad60_2020-08-09-04-25.csv"))
scores_dbert <-       prepareppl(read_csv("by_sentence/scores_abs-loaded=distilbert-base-cased_pad60_2020-08-09-06-36.csv"))
# scores_w2v <-            prepareppl(read_csv("by_sentence/scores_abs-loaded=w2v_pad0_2020-08-09-06-48.csv"))
scores_w2v <-            prepareppl(read_csv("by_sentence/scores_w2v_pad0_2020-08-08-00-28.csv"))
scores_gpt2 <-        prepareppl(read_csv("by_sentence/scores_abs-loaded=gpt2_pad30_2020-08-25-00-19.csv"))

scores_lstm <-           prepareppl(read_csv("by_sentence/scores_abs-loaded=lstm_pad0_2020-08-31-10-17.csv"))
scores_onlstm <-         prepareppl(read_csv("by_sentence/scores_abs-loaded=onlstm_pad0_2020-08-31-10-29.csv"))
scores_onlstm_syd <-     prepareppl(read_csv("by_sentence/scores_abs-loaded=onlstm_syd_pad0_2020-08-31-10-41.csv"))

scores_bert_base <-  prepareppl(read.csv("by_sentence/scores_abs-loaded=bert-base-cased_pad30_2020-08-09-04-37.csv"))
scores_xlnet_large <- prepareppl(read_csv("by_sentence/scores_abs-loaded=xlnet-large-cased_pad30_2020-08-09-07-25.csv"))


scores_bert_base$model <- 'BERT base'
scores_bert_large$model <- 'BERT large'
scores_xlnet_base$model <- 'XLNet base'
scores_xlnet_large$model <- 'XLNet large'
scores_xlm$model <- 'XLM'
scores_bart$model <- 'Bart'
scores_gpt2$model <- 'GPT2'
scores_dbert$model <- 'DistilBERT'
scores_w2v$model <- 'Word2Vec'

scores_lstm$model       <- 'LSTM'
scores_onlstm$model     <- 'ONLSTM'
scores_onlstm_syd$model <- 'ONLSTM_SYD'

scores_lstms = rbind(
  scores_lstm,
  scores_onlstm,
  scores_onlstm_syd
)

scores_models = rbind(
  scores_dbert,
  scores_bert_base,
  scores_bert_large,
  scores_xlnet_base,
  scores_xlnet_large,
  scores_xlm,
  scores_bart,
  scores_gpt2,
  scores_w2v)

scores_models$model <- factor(
  scores_models$model,
  levels = c("DistilBERT",
             "BERT base",
             "BERT large",
             "XLNet base",
             "XLNet large",
             "XLM",
             "Bart",
             "GPT2",
             "Word2Vec"))

scores_models_lstms = rbind(scores_models, scores_lstms)


scores_models %>% filter(sentence_index == 0, score_method == 'projective.uuas.sum')
scores_lstms %>% filter(sentence_index==0, score_method == 'projective.uuas.sum')

`%notin%` <- Negate(`%in%`)
lstms <- c("LSTM","ONLSTM","ONLSTM_SYD")
scores_models_lstms_filtered <- scores_models_lstms %>% filter(
  (score_method == 'projective.uuas.sum' & model %notin% lstms) |
     (score_method == 'projective.uuas.tril' & model %in% lstms))



scores_models_lstms_filtered %>% filter(sentence_length %in% 4:61) %>%
  ggplot(mapping = aes(x=sentence_length)) +
  scale_x_continuous(trans='log10') +
  geom_smooth(aes(y=uuas, colour=model), alpha = 1/5) +
  # ylim(0,1) + #facet_grid(model~.) +
  ggtitle("Accuracy vs sentence length (5-60)") +
  labs(x="sentence length",
       y="PMI dependency accuracy")

selected_models <- c('DistilBERT','BERT large','BERT base',
                     # 'XLNet large','XLM', 'Bart',
                     'XLNet base', 'Word2Vec')
selected_models_lstms <- append(selected_models, lstms)


library(stringr)
baselines_violin <- scores_models %>%
  filter(model=="XLM", str_detect(score_method,"^baseline")) %>%
  ggplot(aes(x=score_method,y=uuas)) +
  geom_violin() + geom_boxplot(width=0.1) +
  scale_x_discrete(labels=c('random','random proj.','linear')) +
  labs(x="baseline", y="dependency accuracy (UUAS)") +
  ggtitle("Baselines")
w2v_violin <- scores_models_lstms_filtered %>%
  filter(model=="Word2Vec") %>%
  # filter(str_detect(score_method,"^proj")) %>%
  ggplot(aes(x=model,
             # colour=score_method,
             y=uuas)) +
  geom_violin() + geom_boxplot(width=0.1) + # facet_wrap(~model,nrow = 1) +
  ylim(0,1) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) + xlab("") +
  ggtitle("")
models_violin <- scores_models_lstms_filtered %>%
  filter(model %in% setdiff(selected_models,c("Word2Vec"))) %>%
  # filter(str_detect(score_method,"^proj")) %>%
  ggplot(aes(x=model,
             # colour=score_method,
             y=uuas)) +
  geom_violin() + geom_boxplot(width=0.1) + # facet_wrap(~model,nrow = 1) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) + xlab("model") +
  ggtitle("Selected contextualized PMI models")
library(gridExtra)
grid.arrange(arrangeGrob(baselines_violin,w2v_violin,models_violin,widths=c(3,1,length(selected_models)-1)))

## exploring the pseudo loglikelihood measure
##
scores_models_lstms_filtered <- scores_models_lstms_filtered %>%
  mutate(model = relevel(model, "Bart")) %>% mutate(model = relevel(model, "GPT2"))


scores_models_lstms_filtered %>% filter(model %in% selected_models) %>% filter(model!="Word2Vec") %>%
  filter(sentence_length > 4,
         sentence_logperplexity<4*exp(1)
        ) %>%
  ggplot(aes(x=sentence_logperplexity,y=uuas)) + theme(legend.position = "top") +
  geom_point(alpha=.15) + geom_smooth(method = "lm", formula = y~x) +
  # geom_boxplot(aes(x=0),alpha=0.15) +
  scale_y_continuous(breaks=c(0,.5,1)) +
  facet_grid(model~.) +
  ggpmisc::stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~~")),
               label.x = "right", label.y = 1,
               formula = y~x, parse = TRUE, size = 3) +
  labs(x="log(pseudo-perplexity)", y="CPMI dependency accuracy") +
  ggtitle("Accuracy vs LM performance")

scores_models_lstms_filtered %>% filter(model %in% selected_models_lstms) %>% filter(model %in% lstms) %>%
  filter(sentence_length > 4,
         sentence_logperplexity<4*exp(1)
  ) %>%
  ggplot(aes(x=sentence_logperplexity,y=uuas)) + theme(legend.position = "top") +
  geom_point(alpha=.15) + geom_smooth(method = "lm", formula = y~x) +
  # geom_boxplot(aes(x=0),alpha=0.15) +
  scale_y_continuous(breaks=c(0,.5,1)) +
  facet_grid(model~.) +
  ggpmisc::stat_poly_eq(aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~~")),
                        label.x = "right", label.y = 1,
                        formula = y~x, parse = TRUE, size = 3) +
  labs(x="log(pseudo-perplexity)", y="CPMI dependency accuracy") +
  ggtitle("Accuracy vs LM performance, LSTM models")


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

compare_models <- scores_models_lstms_filtered %>%
  group_by(sentence_index, model) %>% summarise(uuas, sentence_logperplexity) %>%
  pivot_wider(names_from = model, values_from = c(uuas, sentence_logperplexity)) %>% ungroup()

# some examples...
# compare_models %>% ggplot(aes(x=`uuas_BERT large`,y=`uuas_Word2Vec`)) +
#   geom_point(alpha=0.1) +geom_smooth(method="lm")
# compare_models %>% ggplot(aes(x=`uuas_Bart`,y=`uuas_Word2Vec`)) +
#   geom_point(alpha=0.1) +geom_smooth(method="lm")
# compare_models %>% ggplot(aes(x=`uuas_ONLSTM`,y=`uuas_ONLSTM_SYD`)) +
#   geom_point(alpha=0.1) +geom_smooth(method="lm")
# compare_models %>% ggplot(aes(x=`sentence_logperplexity_ONLSTM`,y=`sentence_logperplexity_ONLSTM_SYD`)) +
#   geom_point(alpha=0.1) +geom_smooth(method="lm")
# compare_models %>% ggplot(aes(x=`sentence_logperplexity_BERT large`,y=`sentence_logperplexity_XLM`)) +
#   geom_point(alpha=0.1) + geom_smooth(method="lm") +
#   scale_x_log10() + scale_y_log10()
# compare_models %>% ggplot(aes(x=`sentence_logperplexity_BERT large`,y=`sentence_logperplexity_Bart`)) +
#   geom_point(alpha=0.1) + geom_smooth(method="lm") +
#   scale_x_log10() + scale_y_log10()
#
# cor.test(compare_models$`uuas_BERT base`,compare_models$`uuas_BERT large`,method="pearson")

compare_uuas <- compare_models %>%
  select(starts_with("uuas")) %>% set_names(~str_replace_all(., "uuas_", ""))
compare_ppl  <- compare_models %>%
  select(starts_with("sentence_logperplexity")) %>% set_names(~str_replace_all(., "sentence_logperplexity_", ""))

#
# library("Hmisc")
# Hcor <- rcorr(as.matrix(compare_uuas))
#
# flattenCorrMatrix <- function(cormat, pmat) {
#   ut <- upper.tri(cormat)
#   data.frame(
#     row = rownames(cormat)[row(cormat)[ut]],
#     column = rownames(cormat)[col(cormat)[ut]],
#     pearson_r = (cormat)[ut],
#     P = pmat[ut]
#   )
# }
#
# flattenCorrMatrix(Hcor$r, Hcor$P)
#
# library("corrplot")
# corrplot(cor(compare_uuas),method = "ellipse")

pointsmooth <- function(data, mapping, method="loess", ...){
  p <- ggplot(data = data, mapping = mapping) +
    geom_point(alpha=0.1) +
    geom_smooth(colour="red",method=method, ...)
  p
}
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

PerformanceAnalytics::chart.Correlation(compare_uuas, histogram=TRUE)
library("GGally")
ggpairs(compare_uuas,
        upper = list(continuous = wrap(corrcol, colour = "black")),
        lower = list(continuous = pointsmooth),
        # diag = list(continuous = wrap("densityDiag", alpha=0.5)),
        title = "PMI depedency accuracy (by sentence, pearson) correlogram "
        )
ggpairs(compare_ppl %>% log(),
        upper = list(continuous = wrap(corrcol, colour = "black")),
        lower = list(continuous = pointsmooth),
        title = "Sentence log perplexity (by sentence, pearson) correlogram (log scale)"
)

