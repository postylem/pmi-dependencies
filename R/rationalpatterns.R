library(tidyverse)

# this is an interesting emergent pattern to explain...

randdf <- data.frame(replicate(2, sample(35:60, 10000, rep = TRUE)))
variance <- sample(0:25, 10000, rep = TRUE)
n <- 60 + variance
randdf <- randdf %>% transform(Q1 = X1 / n, Q2 = X2 / n)
randdf %>% ggplot(mapping = aes(x = Q1, y = Q2, colour = n)) +
    geom_point(alpha = 0.1)
