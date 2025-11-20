---
layout: post
title: The Double Descent Phenomenon
date: 2025-04-12 22:25:00
description: Modern machine learning models defy the bias-variance trade-off. A new theory of machine learning must be developed to explain the double descent phenomenon and the success of over-parameterized models. 
tags: ml theory tutorials
tikzjax: true
featured: false
---

# Double Descent
The discovery of the descent phenomenon (Belkin, 2019) has upended the classical understanding of bias-variance trade-off. Classical understanding suggests that increasing model complexity inevitably leads to overfitting. A long standing mystery has been why neural networks have such successful generalization performance even when it has been "overfit" on training data. The double descent phenomenon demonstrates that the bias-variance trade-off is incomplete. For low capacity models, the bias-variance trade-off successfully explains generalization performance. However, for high capacity models (e.g. neural networks, random forest, etc.), increasing model complexity often leads to improved generalization error. Consequently, studying over-parameterized models -- models that interpolate the training data to achieve perfect training error -- have been an active field of research. 

<img src="{{ site.baseurl }}/assets/img/double-descent-curve.png" alt="Double Descent Curve" style="width:100%; max-width:600px;">

Recent evidence suggests SGD behaves differently in these over-parameterized contexts than in classical scenarios (Ma, et al., 2018; Belkin, 2021). In particular, whereas SGD tends to reach local minima in classical scenarios, SGD tends to reach global minima in over-parameterized regimes. 

<img src="{{ site.baseurl }}/assets/img/sgd-under-over-parameterized.png" alt="SGD Over-Parameterized" style="width:100%; max-width:600px;">

Moreover, while SGD with fixed learning rate does not converge in classical scenarios, it converges exponentially in over-parameterized scenarios. 

<img src="{{ site.baseurl }}/assets/img/double-descent-comparisons.png" alt="Double Descent Comparison" style="width:100%; max-width:600px;">

All in all, machine learning theory has long lagged behind our empirical understanding. The double-descent phenomenon and the surprising advantages of over-parameterization are only the first insights into explaining the remarkable success of machine learning models. 

**References:**

1. **Belkin, M., Hsu, D., Ma, S., & Mandal, S.** (2019). *Reconciling modern machine-learning practice and the classical bias–variance trade-off.* Proceedings of the National Academy of Sciences, 116(32), 15849–15854. [https://doi.org/10.1073/pnas.1903070116](https://doi.org/10.1073/pnas.1903070116)

2. **Ma, S., Bassily, R., & Belkin, M.** (2018). *The power of interpolation: Understanding the effectiveness of SGD in modern over-parameterized learning.* Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80:3325–3334. [http://proceedings.mlr.press/v80/ma18a/ma18a.pdf](http://proceedings.mlr.press/v80/ma18a/ma18a.pdf)

3. **Belkin, M.** (2021). *Fit without fear: Remarkable mathematical phenomena of deep learning through the prism of interpolation.* Acta Numerica, 30, 203–248. [https://doi.org/10.1017/S0962492921000039](https://doi.org/10.1017/S0962492921000039)
