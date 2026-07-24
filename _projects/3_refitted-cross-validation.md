---
layout: page
title: Refitted Cross-Validation
description: A regression-adjustment method for high-dimensional causal effect estimation
img: assets/img/refitted-cross-validation.svg
importance: 2
category: work
related_publications: false
---

In a randomized trial, treatment assignment is independent of potential outcomes by design, so a naive comparison of means between treatment and control is already an unbiased estimate of the causal effect &mdash; there's no confounding to correct for. Covariate adjustment doesn't fix bias here, because there isn't any to fix; its role is to soak up predictable variance in the outcome, tightening the estimate and improving power without touching its unbiasedness. That's a different justification than the usual "adjust for covariates to remove confounding" story in observational causal inference &mdash; here it's about efficiency, not bias.

**The problem.** Once the number of covariates (p) exceeds the number of observations (n), standard linear regression breaks down outright. The natural fix is to first select a smaller set of variables (e.g. via lasso), then refit a regression using just those &mdash; but that introduces overfitting: because the data was used to pick the variables, refitting on the same data understates the true variance of the estimates, which overstates the significance of the results.

**Approach.** Refitted cross-validation is a two-stage procedure: variable selection (lasso and other approaches), followed by cross-fitting to correct for the variance-understatement that comes from selecting and refitting on the same data. The goal is for this to be a competitor to double machine learning for causal effect estimation in high-dimensional settings.

**Tooling.** Built an R Shiny dashboard that breaks each simulation down by algorithm stage rather than just reporting a final bias/variance number. For the variable-selection stage, it reports false positive/negative rates and plots which variables were selected. For the refitting stage, it shows how well the refitted model fits the held-out fold &mdash; making it possible to see exactly how the method behaves at each step, not just whether the end result looks good.

Currently simulation-only; a quantitative comparison against standard adjustment and double machine learning is ongoing.

**Skills:** causal inference, regression modeling, high-dimensional statistics, simulation study design, R (tidyverse), R Shiny.
