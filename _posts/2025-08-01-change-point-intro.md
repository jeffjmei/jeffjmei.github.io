---
layout: post
title: Change Point Analysis Introduction
date: 2025-08-01 12:25:00
description:  
tags: 
tikzjax: true
featured: false
thumbnail: /assets/img/DUR_NDUR_volatility.png
related_publications: false
---

<div class="row d-flex justify-content-center text-center">
    <div class="col-sm mt-3 mt-md-0" style="max-width: 500px;" >
        {% include figure.liquid loading="eager" path="assets/img/DUR_NDUR_volatility.png" title="labor-statistics" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Volatility of Labor Statistics.
</div>

When you collect data for a long enough time, the data-generating process will inevitably change. The mean might shift, the variance might shrink. _Change points_ are where these changes occur. 

Failing to incorporate _change points_ into an analysis risks producing false conclusions. Research in _change point analysis_ is focused on extending the statistical inference repertoire to handle distribution shifts. 

Change points are present in a wide variety of applications, but their primary use is with biomedical and economic/financial data. Stock prices are collected every second of every day, but market conditions change constantly. Over time, laws get passed and policies get enacted. Each of these can trigger a bull run or a recession. We cannot accurately study these time-series without considering the distribution shifts. 
