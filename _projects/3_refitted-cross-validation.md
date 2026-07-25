---
layout: page
title: Refitted Cross-Validation
description: A regression-adjustment method for high-dimensional causal effect estimation
img: assets/img/refitted-cross-validation.svg
importance: 2
category: work
related_publications: false
---

## Motivation

**RCTs estimate the treatment effect.** In a randomized controlled trial, patients are split into a control arm and a treatment arm. Under this design, a simple t-test is an unbiased estimator of treatment effect. This determines whether a treatment is efficacious or not.

**Covariate adjustment improves estimation.** The t-test can be made more powerful with covariate adjustment. We can incorporate covariates (e.g. race, sex, age) into a regression model to make the treatment effect estimation more precise without sacrificing unbiasedness.

**High-dimensional covariates break OLS.** Medical imaging, genetic panels, and wearable devices all provide a rich set of covariates to adjust on. However, once the number of covariates exceeds the number of patients, ordinary least squares no longer produces a unique solution.

**Reusing data for selection and refitting overfits.** We can apply variable selection (e.g. via lasso), and refit the OLS with the selected variables. This produces a unique solution, but overstates significance, because the same data used to select the covariates is reused to estimate the effect. *Refitted cross-validation* (RCV) fixes this by keeping the two steps on separate data.

## Approach

<div class="row d-flex justify-content-center text-center">
    <div class="col-sm mt-3 mt-md-0" style="max-width: 600px;" >
        {% include figure.liquid loading="eager" path="assets/img/refitted-cross-validation.svg" title="refitted cross-validation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Selection and estimation are split across folds so neither uses the same data as the other.
</div>

Assume the model $$Y = \tau A + X\beta + \varepsilon$$, where $$A$$ is the treatment assignment, $$X$$ is a high-dimensional vector of covariates, and $$\tau$$ is the treatment effect of interest. Refitted cross-validation splits the data into $$K=2$$ balanced folds and separates variable selection from estimation:

1. **Select.** On fold $$-k$$, fit a lasso regression of $$Y$$ on $$A$$ and $$X$$, and keep the covariates with nonzero coefficients as the selected set $$\hat{M}_{(k)}$$.
2. **Refit.** On the held-out fold $$k$$, refit an ordinary least squares regression of $$Y$$ on $$A$$ and $$X_{\hat{M}_{(k)}}$$ to get a fold-specific treatment effect estimate $$\hat{\tau}_{(k)}$$.
3. **Aggregate.** Average the two fold estimates:

$$
\hat{\tau} = \frac{\hat{\tau}_{(1)} + \hat{\tau}_{(2)}}{2}.
$$

Because each fold's estimate is computed from data that played no role in selecting its own variables, the overfitting from variable selection is broken. Empirically, the standard errors between folds are close to independent, so they combine as

$$
SE(\hat{\tau}) = \sqrt{Var(\hat{\tau}_{(1)}) + Var(\hat{\tau}_{(2)})}.
$$
