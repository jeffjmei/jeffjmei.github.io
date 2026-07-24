---
layout: page
title: Equivariant Covariance Estimation
description: A correlation estimator for non-stationary time series, robust to mean shifts
img: assets/img/circular-lag-graph.png
importance: 1
category: work
related_publications: false
---

Standard correlation estimators break down under non-stationarity: if the mean of a series shifts partway through, classical Pearson correlation can register a spurious correlation with another series even when there's no real association between them. This is a common failure mode in real time series &mdash; a market regime change, a policy shift, a sensor recalibration &mdash; and it silently corrupts downstream analysis that assumes stationarity.

**Approach.** The estimator works off lagged differences, which gives an unbiased correlation estimate when there are no mean shifts. That same lagged-difference estimator becomes biased once a shift is introduced, so we correct for it with a regression-based debiasing step. This keeps the estimator valid even in the presence of a shift. "Equivariant" isn't just descriptive &mdash; it's a technical property we define and prove for this estimator, characterizing how it behaves under the class of mean-shift transformations. As far as we're aware, no existing method handles this case.

**Validation.** In simulation, the estimator achieves nominal Type I error control and holds up across a range of misspecified scenarios, where the actual data-generating process deviates from what the estimator assumes. Beyond simulation, we validated it against real datasets chosen so that domain knowledge could sanity-check the result:

- **Air pollution (ozone vs. temperature):** atmospheric chemistry predicts a negative correlation &mdash; a known-sign expectation to check the estimator's output against.
- **Major wildfire events:** tested whether the estimator handled a large, real mean-shift better than standard methods.
- **FRED manufacturing employment:** total manufacturing employment is directly decomposable into durables and non-durables employment, giving a case with guaranteed correlation to confirm the estimator was powerful enough to detect it.

**Tooling.** Built an R Shiny dashboard to run and visualize the Type I error / power simulation studies backing the estimator, making it easy to explore behavior across parameter regimes without rerunning scripts by hand.

A manuscript is currently in preparation.

**Skills:** time-series analysis, simulation study design, hypothesis testing, R (tidyverse), R Shiny.
