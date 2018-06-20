library(MEMSS)
library(lme4)

Rm1ML <- lmer(travel ~ 1 + (1 | Rail), Rail, REML = FALSE, verbose = TRUE);
summary(Rm1ML)