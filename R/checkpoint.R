library(tidyverse)
theme_set(theme_minimal())
#setwd("/Users/j/McGill/PhD-miasma/pmi-dependencies/R")

#bertckpt <- read_csv("bert-checkpoint500.csv")
bertckpt <- read_csv("abs-bert-checkpoint.csv")
bertckpt <- filter(bertckpt,`training steps`!='off-shelf')
bertckpt$`training steps` <- fct_relevel(bertckpt$`training steps`, c("10","50","100","500","1000","1500"))


ggcolhue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
ggcolhue(2)[[1]]


coeff <- 20
bertckpt %>%
  rename(nproj = nproj_sum, proj = proj_sum) %>%
  pivot_longer(cols = -c(`training steps`,meanppl),
               names_to = "method",
               values_to = "accuracy") %>%
  filter(method %in% c("nproj","proj")) %>%
  ggplot(aes(x=`training steps`,y=accuracy * coeff,colour=method)) +
  geom_path(aes(y = meanppl,group=rep(c("log PPL"))), color="#888888") +
  geom_point() + geom_line(aes(group=method)) +
  geom_hline(yintercept=.26 * coeff, linetype="dashed", color = ggcolhue(2)[[2]]) +
  geom_hline(yintercept=.13 * coeff, linetype="dashed", color = ggcolhue(2)[[1]]) +
  #geom_hline(yintercept=.5  * coeff, linetype="dashed") +
  xlab("BERT training steps (thousands)") +
  scale_y_continuous(
    name = "mean log perplexity",
    sec.axis = sec_axis(~./coeff, name="accuracy")
  ) +
  theme( axis.ticks.y.left = element_line(color = "#666666"),
         axis.text.y.left = element_text(color =  "#666666"),
         axis.title.y.left = element_text(color = "#666666")) +
  scale_color_discrete(name="CPMI\ndependency\nmethod") +
  ggtitle("Accuracy and perplexity during training")


bertckpt %>%
  rename(nproj = nproj_sum, proj = proj_sum) %>%
  ggplot() +
  geom_point(aes(x=`training steps`,y=meanppl)) +
  xlab("training steps (bert-base-uncased)") +
  ylab("Avg sentence log perplexity") +
  ggtitle("Perplexity at checkpoints during training")


get_lgppl<-function(csvfile){
  mean_ppl<-read_csv(csvfile) %>%
    mutate(sentence_logperplexity=-pseudo_loglik/sentence_length) %>%
    select(sentence_index,sentence_logperplexity) %>%
    summarize(mean_ppl=mean(sentence_logperplexity))
  return(mean_ppl[[1]])
}

# c(
# get_lgppl("../results-azure/bert-base-uncased.ckpt-10k(500)_pad60_2020-05-16-18-23/scores_bert-base-uncased.ckpt-10k(500)_pad60_2020-05-16-18-23.csv"),
# get_lgppl("../results-azure/bert-base-uncased.ckpt-50k(500)_pad60_2020-05-16-17-56/scores_bert-base-uncased.ckpt-50k(500)_pad60_2020-05-16-17-56.csv"),
# get_lgppl("../results-azure/bert-base-uncased.ckpt-100k(500)_pad60_2020-05-16-17-28/scores_bert-base-uncased.ckpt-100k(500)_pad60_2020-05-16-17-28.csv"),
# get_lgppl("../results-azure/bert-base-uncased.ckpt-500k(500)_pad60_2020-05-16-17-01/scores_bert-base-uncased.ckpt-500k(500)_pad60_2020-05-16-17-01.csv"),
# get_lgppl("../results-azure/bert-base-uncased.ckpt-1000k(500)_pad60_2020-05-16-16-34/scores_bert-base-uncased.ckpt-1000k(500)_pad60_2020-05-16-16-34.csv"),
# get_lgppl("../results-azure/bert-base-uncased.ckpt-1500k(500)_pad60_2020-05-16-23-46/scores_bert-base-uncased.ckpt-1500k(500)_pad60_2020-05-16-23-46.csv"),
# get_lgppl("../results-azure/bert-base-uncased(500)_pad60_2020-05-16-16-06/scores_bert-base-uncased(500)_pad60_2020-05-16-16-06.csv"))

# ==> 8.861502 8.400238 5.984933 5.657154 5.292113 5.605134 3.041858

c(
get_lgppl("../results-clean/bert-base-uncased.ckpt-10k_pad30_2020-07-06-00-45/scores_bert-base-uncased.ckpt-10k_pad30_2020-07-06-00-45.csv"),
get_lgppl("../results-clean/bert-base-uncased.ckpt-50k_pad30_2020-07-06-01-48/scores_bert-base-uncased.ckpt-50k_pad30_2020-07-06-01-48.csv"),
get_lgppl("../results-clean/bert-base-uncased.ckpt-100k_pad30_2020-07-06-02-51/scores_bert-base-uncased.ckpt-100k_pad30_2020-07-06-02-51.csv"),
get_lgppl("../results-clean/bert-base-uncased.ckpt-500k_pad30_2020-07-06-03-53/scores_bert-base-uncased.ckpt-500k_pad30_2020-07-06-03-53.csv"),
get_lgppl("../results-clean/bert-base-uncased.ckpt-1000k_pad30_2020-07-06-04-56/scores_bert-base-uncased.ckpt-1000k_pad30_2020-07-06-04-56.csv"),
get_lgppl("../results-clean/bert-base-uncased.ckpt-1500k_pad30_2020-07-06-05-58/scores_bert-base-uncased.ckpt-1500k_pad30_2020-07-06-05-58.csv"),
get_lgppl("../results-clean/bert-base-uncased_pad30_2020-07-04-12-15/scores_bert-base-uncased_pad30_2020-07-04-12-15.csv"))

# ==> 8.972310 8.515644 6.195083 5.938785 5.753842 6.092419 3.302884

