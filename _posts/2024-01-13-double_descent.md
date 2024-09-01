---
layout: post
title:  The Double Descent Phenomenon
date: 2024-01-13 23:25:00
description: 
tags: tutorials
categories: tutorials
tikzjax: true
featured: true
---


One of the most fundamental concepts in statistics is the bias-variance trade-off. 

The lasso uses bias-variance trade-off to determine the optimal number of features to use in prediction.

Cross-validation is one of the most common ways to assess the generalization error of models. 

It is typically said that if a model accomplishes perfect training error, we can expect the model to be over-fitted. The model is learning the noise rather than the systematic pattern, and is unlikely to generalize to unseen data. 

While over-fitting has long been understood, Breiman (????) posed the question: why do neural networks not over-fit? 

While common wisdom insisted perfect training error nearly guaranteed poor generalization error, neural networks seemed to defy this wisdom. Computer scientists ignored statisticians pleas to apply the principles of the bias-variance trade-off. 

This remained a mystery for two decades until the recent discovery of the double descent phenomenon (Belkin, 2017). The double descent phenomenon challenges the conventional view of bias-variance trade-off and generalizes it. 



The question as to why over-parameterized models behave so well still remains. How come some models overfit where others do not?  











