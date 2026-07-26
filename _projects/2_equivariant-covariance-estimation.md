---
layout: page
title: Correlation Estimation with Mean Shifts
description: Correlation underlies many methods, but classical correlation estimation requires i.i.d. data. This is a problem, because time series data are often neither independent nor identically distributed.
img: assets/img/circular-lag-graph.png
importance: 1
category: work
related_publications: false
---

## Motivation

**Correlation underlies many methods.** It's one of the most fundamental statistical quantities, so it's unsurprising that its estimation is necessary for many statistical methods.

1. **Principal Component Analysis (PCA).** To apply PCA, we estimate the covariance matrix $$\Sigma$$, and then apply an eigendecomposition on it.
2. **Causal Discovery.** To identify the causal skeleton, we rely on the precision matrix (the inverse of the covariance matrix).
3. **Portfolio Volatility.** The variance of a stock explains how volatile it is, while the covariance between stocks explains how diversified the portfolio is.

**Correlation requires i.i.d. data.** Classical correlation estimation assumes the data is <u>independent</u>, so there is no autocorrelation, and it also assumes the data is <u>identically distributed</u>, so the mean and variance do not change over time.

**Time series are often not i.i.d.** As data collection technologies improve, we're collecting data at higher frequencies, so <u>autocorrelation is becoming more ubiquitous</u>. Time series are also characterized by <u>trends, seasonality, and change points</u>, all of which violate the identically distributed assumption. In short, time series are neither independent, nor identically distributed. <span style="color: red;">Applying classical correlation estimators on non-i.i.d. data will produce spurious correlation.</span>

**Example.** Consider two uncorrelated sequences of data that share the same mean. Take

$$
X_i \sim N(\theta_i, 1) \qquad Y_i \sim N(\theta_i, 1), \qquad
\theta_i = \begin{cases} 1 & i \le 3 \\ -1 & i > 3 \end{cases}
$$

with $$\text{Cov}(X_i, Y_i) = 0, \, \forall i$$. If there is a mean shift, then they will become spuriously correlated.

<div class="row d-flex justify-content-center text-center">
    <div class="col-sm mt-3 mt-md-0 figure-frame" style="max-width: 600px;" >
        {% include figure.liquid loading="eager" path="assets/img/ece-spurious-correlation.png" title="spurious correlation from a mean shift" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption text-center">
    A mean shift in two otherwise-unrelated series (left) inflates their apparent correlation (right).
</div>

## Approach

Assume the model $$X = \theta_X + \varepsilon_X$$ and $$Y = \theta_Y + \varepsilon_Y$$, where $$\theta_X, \theta_Y$$ are piecewise-constant mean functions and $$\varepsilon_X, \varepsilon_Y$$ are mean-zero errors with $$\text{Cov}(\varepsilon_X, \varepsilon_Y) = \sigma_{XY}$$. Under this model, we can define the circular $$k$$-lag statistic:

$$
T_k(X, Y) = \frac{1}{2n} \sum_{i=1}^{n} (X_i - X_{i+k})(Y_i - Y_{i+k}),
$$

Taking the expectation, we get

$$
\mathbb{E}\left[T_k(X, Y)\right] = \textcolor{red}{\sigma_{XY}} + k \, T_1(\theta).
$$

The desired covariance pops out, but is biased by $$T_1(\theta)$$ &mdash; a function of the mean $$\theta$$, which is unknown in practice.

<div class="row d-flex justify-content-center text-center">
    <div class="col-sm mt-3 mt-md-0 figure-frame" style="max-width: 300px;" >
        {% include figure.liquid loading="eager" path="assets/img/ece-regression.png" title="regression-based debiasing" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption text-center">
    Extrapolating the lag-k statistic back to k=0 debiases the estimate.
</div>

We debias the circular $$k$$-lag statistic through a regression scheme. Notice that $$\mathbb{E}[T_k(X,Y)]$$ is linear in $$k$$. Then, we can interpret the expectation as a linear equation: it has an intercept of $$\sigma_{XY}$$ and a slope of $$T_1(\theta)$$. Therefore, we can compute $$T_1(X,Y)$$ and $$T_2(X,Y)$$, and extrapolate the line back to the intercept to get an estimate of $$\sigma_{XY}$$. Then, we get the computable expression

$$
\boxed{\hat{\sigma}_{XY} = 2\,T_1(X, Y) - T_2(X, Y).}
$$

This is an unbiased covariance estimator in the presence of mean shifts.

## Applications

Labor productivity measures how much output a worker produces per hour worked, and it's a key indicator of economic health and long-run growth. During the 2008 market crash, several economic sectors suffered severe shocks to their labor productivity. We'll take a close look at two manufacturing sectors: durables and non-durables. Durable manufacturing includes things that last a long time (e.g. washing machines, refrigerators, cars), whereas non-durable manufacturing includes things that don't (e.g. clothes, food, paper products).

<div class="row d-flex justify-content-center text-center">
    <div class="col-sm mt-3 mt-md-0 figure-frame" style="max-width: 600px;" >
        {% include figure.liquid loading="eager" path="assets/img/DUR_NDUR_volatility.png" title="durable vs. non-durable manufacturing volatility" class="img-fluid rounded" %}
    </div>
</div>

Taking the Pearson correlation between these two time series, we find that they are strongly related ($$p < 0.001$$). However, our method shows that once we account for the mean shift, they may not be correlated after all ($$p = 0.37$$).

Mean shifts can clearly change the conclusions drawn from a correlation test. This underscores the need for methods that relax the stringent i.i.d. assumptions built into classical estimators like Pearson correlation.
