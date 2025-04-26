---
layout: post
title: Generalized Additive Models
date: 2025-04-15 22:25:00
description: Generalized additive models are powerful but can easily overfit in high dimensions. Techniques like COSSO and its extensions use soft-thresholding to automatically select important components, making these models more reliable and interpretable. 
tags: ml theory tutorials
tikzjax: true
featured: true
thumbnail: /assets/img/3dplot.png
related_publications: true
---

# Generalized Additive Models
Nonparametric additive models are flexible extensions of linear models. They can fit to complex smooth surfaces, but they suffer in high dimensions since they potentially fit to insignificant components. Therefore, a mechanism for filtering out insignificant components is of paramount importance. Since the advent of the LASSO {% cite tibshirani_regression_1996 %}, many procedures have applied regularization techniques to instill variable selection capabilities. In this literature review, we cover two approaches that apply soft-thresholding to filter out insignificant components in the context of additive models. Lin and Zhang propose the COSSO (COmponent Selection and Smoothing Operator), which generalizes the LASSO to the context of additive models in order to facilitate component selection {% cite lin_component_2006 %}. Building on this work, {% cite ravikumar_sparse_2008 %} proposes a variation that generalizes the COSSO and applies a backfitting procedure to facilitate variable selection.

Additive models were introduced as a generalization of linear models {% cite hastie_generalized_1986 %}. They take the form

$$
\begin{equation}
    Y_i = \sum_{j=1}^{p} f_j(X_{ij}) + \varepsilon_i
    \label{eq:additive_model}
\end{equation}
$$

where $$\varepsilon$$ is often gaussian noise. It is easy to see that if we let $$f(X) = X \beta$$, then we have the linear model again. In practice, it is common to let $$f \in \mathcal{S}^2[0, 1]$$, the second order Sobolev space on $$[0, 1]$$, defined as the set $$\left\{ f: f, f' \text{ abs. continuous}, \int_{0}^{1} (f''(x))^2 d{x} < \infty \right\}$$. This is an enormous class of functions that includes polynomials, $$\sin$$, $$\cos$$, $$\log$$, and many more functions, making additive models an incredibly flexible approach to data modeling.

It is worth noting that additive models can be extended to the class of Smoothing-Spline Analysis of Variance (SS-ANOVA) models if interactions are considered as well. That is, SS-ANOVA models take the form

$$
\begin{equation}
    Y_i = \sum_{j=1}^{p} f_j(X_{ij}) + \sum_{j < k} f_{jk}(X_{ij}) + ... + \varepsilon_i.
    \label{eq:ss_anova}
\end{equation}
$$

While additive models are incredibly flexible, one problem is that they suffer in high dimensions. Often times when $$p$$ is large, there are components in the model that make little to no contributions to the model performance. However, without filtering out these features, the model will end up fitting to the noise. Therefore,

In this literature review, we will provide an overview of methods used to overcome this issue. In particular we will consider the COmponent Selection and Smoothing Operator (COSSO) {% cite lin_component_2006 %} and the Sparse Additive Model (SpAM) {% cite ravikumar_sparse_2008 %}.


# Reproducing Kernel Hilbert Spaces

It is essential to understand the basics of Reproducing Kernel Hilbert Spaces (RKHS), as they play a significant role in both the COSSO and SpAM procedures.

RKHSs are useful tools in the smoothing spline literature. They offer a feasible approach to access a potentially infinite dimensional class of functions (e.g. 2nd order Sobolev space). For a more detailed approach, refer to {% cite gu_smoothing_2002 %} and {% cite nosedal-sanchez_reproducing_2012 %}. Here, we will provide an informal overview of RKHS theory.

A **functional** $$L$$ is defined as a mapping from a linear space $$\mathcal{V}$$ to the real numbers $$L: \mathcal{V} \to \mathbb{R}$$. A functional of particular interest is the **evaluation functional** $$[\cdot]$$, such that $$[x]f = f(x)$$. That is, the evaluation functional $$[x]$$ is equal to evaluating the function $$f$$ at $$x$$.

> **Riesz Representation Theorem**
>
> Let $$\mathcal{H}$$ be a Hilbert space with continuous functional $$L$$ defined on it. For any $$f \in \mathcal{H}$$, there exists a unique $$g \in \mathcal{H}$$ such that $$Lf = \langle f, g \rangle. $$

If we take $$[x]$$ to be the evaluation functional, it follows that for any continuous function $$f$$, there exists some representer $$R_x$$ such that $$\langle R_x, f \rangle = f(x)$$. We will call $$R_x$$ the **reproducing kernel** of the RKHS $$\mathcal{H}_R$$.

<div class="row d-flex justify-content-center text-center">
    <div class="col-sm mt-3 mt-md-0" style="max-width: 500px;" >
        {% include figure.liquid loading="eager" path="assets/img/additive-models/Riesz.png" title="riesz-representation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Riesz Representation Theorem.
</div>

The significance of this result is that it provides two different ways to evaluate a functional. We can evaluate it on the set $$X$$ with the kernel, or we can project the data onto a different space $$\mathcal{H}_R$$ and evaluate the functional using the inner product. Most often however, we will evaluate the functional using the kernel because it is typically more computationally convenient.

To find a RKHS, we note that there is a convenient relationship between non-negative definite functions and reproducing kernels.

> **RKHS - Non-Negative Definite Relationship**
>
> For every non-negative definite function $$R(x, y)$$ on $$X$$, there exists a unique RKHS $$\mathcal{H}_R$$ with the reproducing kernel $$R(x, y)$$. The converse is also true. For every RKHS $$\mathcal{H}_R$$, there exists a unique non-negative definite function $$R(x, y)$$ on $$X$$.

<div class="row d-flex justify-content-center text-center">
    <div class="col-sm mt-3 mt-md-0" style="max-width: 500px;" >
        {% include figure.liquid loading="eager" path="assets/img/additive-models/RKHS.png" title="rkhs" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Reproducing Kernel Hilbert Space.
</div>

For the remainder of this paper, we define the inner product to be

$$
\begin{equation}
    \langle f, g \rangle =
    \sum_{v = 0}^{m-1}
    \left(  \int_{0}^{1} f^{(v)} d{x}\right)
    \left(  \int_{0}^{1} g^{(v)} d{x}\right) +
    \int_{0}^{1} f^{(m)}g^{(m)}  d{x}.
    \label{eq:inner_prod}
\end{equation}
$$

One important consequence of the RKHS framework, is that we are able to represent functions in $$\mathcal{H}_R$$ by their kernel functions:

$$
\begin{equation}
    f(x) = \sum_{\alpha=1}^{p} \theta_\alpha R_\alpha  c + b \textbf{1}_n.
\label{eq:rkhs_function}
\end{equation}
$$

By representing functions in $$\mathcal{H}_R$$ by their kernels, we are able to obtain notions of "similarity" between functions in a richer space $$\mathcal{H}_R$$ without having to do any computations within that richer space. Instead, our computations remain on the original space $$X$$.


# COSSO: Component Selection and Smoothing

The COSSO is a flexible approach that performs component selection to filter out small to insignificant components. This is similar to the LASSO {% cite tibshirani_regression_1996 %}, but instead of penalizing on coefficient size, the COSSO penalizes, in an informal sense, "component size." Roughly speaking, size of a function is described as the squared integral over $$[0, 1]$$. To develop this fully, we must dive into the theory of Reproducing Kernel Hilbert Spaces (RKHS).

In particular, the problem that we are solving is

$$
\begin{equation}
    \hat{f} = \text{argmin}_f \sum_{i=1}^{n} (y_i - f(x_i))^2  + \lambda_n \sum_{j=1}^{p} \| P^j f \|
    \label{eq:cosso}
\end{equation}
$$

where the norm in the 2nd order Sobolev space is

$$
\begin{equation}
    \| f \| =
    \left(  \int_{0}^{1} f(t) d{t} \right)^2 +
    \left(  \int_{0}^{1} f'(t) d{t} \right)^2 +
    \int_{0}^{1} \left( f''(t) \right)^2 d{t}.
    \label{eq:norm}
\end{equation}
$$

The first term in (\ref{eq:cosso}) encourages the optimization problem to fit as closely to the data as possible, but it is penalized by the 2nd term, which penalizes by model complexity.

## Relationship to LASSO

The authors demonstrate that the COSSO and the LASSO are the same. It is important to note that while the COSSO generalizes the LASSO, the interpretation changes. Instead of penalizing on coefficient size, we are penalizing by component size.

## Algorithm

The authors demonstrate that the COSSO can be decomposed into a non-negative garrote and a smoothing spline problem. Since the algorithm is guaranteed to improve its estimate on every iteration, we can just alternate between the non-negative garrote solution and the smoothing spline solution.

The authors note that the original COSSO problem (\ref{eq:cosso}) can be reformulated into the more computationally tractable form

$$
\begin{equation}
    \frac{1}{n}
    \left( y - \sum_{\alpha=1}^{p} \theta_\alpha R_\alpha c - b \textbf{1}_n  \right)^T
    \left( y - \sum_{\alpha=1}^{p} \theta_\alpha R_\alpha c - b \textbf{1}_n  \right) +
    \lambda_0 \sum_{\alpha=1}^{p} \theta_\alpha c^T R_\alpha c +
    \lambda \sum_{\alpha=1}^{p} \theta_\alpha.
    \label{eq:cosso_algo}
\end{equation}
$$

As it turns out, (\ref{eq:cosso_algo}) can be broken down further into two constituent sub-algorithms. In particular, when we fix $$c$$ and $$b$$, (\ref{eq:cosso_algo}) is reduced into the ridge regression. Similarly, when $$\theta$$ is fixed, (\ref{eq:cosso_algo}) reduces into a non-negative garrote.

By fixing $$c, b$$, the COSSO reduces to the non-negative garrote, where components are selected:

$$
\begin{equation}
    \min_\theta (z - G \theta)^T (z - G \theta) + n \lambda \sum_{\alpha=1}^{p} \theta_\alpha
    \label{eq:cosso_garrote}
\end{equation}
$$

where $$\theta_\alpha \ge 0$$ and $$z = y - (1/2) n \lambda_0 c - b \textbf{1}_n. $$

Similarly, by fixing $$\theta$$, we get a problem equivalent to ridge regression, where the functions are smoothed:

$$
\begin{equation}
    \min_{c, b}
    (y - R_\theta c - b \textbf{1}_n)^T
    (y - R_\theta c - b \textbf{1}_n) +
    n \lambda_0 c^T R_\theta c.
    \label{eq:cosso_ridge}
\end{equation}
$$

Now, with the insight that the COSSO can be broken down into problems with known solutions, the proposed algorithm flips between fixing $$\theta$$ and fixing $$c \text{ and } b$$. In other words, the algorithm flips between the non-negative garrote and ridge regression. We continue until algorithm converges on a solution with a pre-specified error.

$$
\begin{aligned}
&\textbf{Algorithm: COSSO} \\
&\textbf{Initialize:} \text{fix } \theta_\alpha = 1, \alpha = 1, ..., p, \; g(\theta, b, c) = 0 \\
&\textbf{Repeat until convergence:} \\
&\quad 1. \quad \text{Fix } \theta, \text{apply ridge regression (\ref{eq:cosso_ridge})} \\
&\quad 2. \quad (c, b) \gets \text{argmin}_{c, b}
        \left[
        (y - R_\theta c - b \mathbf{1}_n)^T
        (y - R_\theta c - b \mathbf{1}_n) +
        n \lambda_0 c^T R_\theta c
        \right] \\
&\quad 3. \quad \text{Fix } b, c, \text{apply non-negative garrote (\ref{eq:cosso_garrote})} \\
&\quad 4. \quad \theta \gets \text{argmin}_\theta
        \left[
        (z - G \theta)^T (z - G \theta) + n \lambda \sum_{\alpha=1}^{p} \theta_\alpha
        \right] \\
&\quad 5. \quad g(\theta, b, c) \gets \min_\theta
        \left[
        (z - G \theta)^T (z - G \theta) + n \lambda \sum_{\alpha=1}^{p} \theta_\alpha
        \right]
\end{aligned}
$$

However, the authors note that the first iteration makes most of the way to a solution.

# Sparse Additive Models

Ravikumar et al. (2008) propose the Sparse Additive Model (SpAM), which is similar to the COSSO. Again, the model is proposed is another $$l_1$$ penalized approach to add sparsity to the additive model. However, SpAM applies an additional constraint to normalize function size. In doing so, SpAM decouples sparsity and smoothness. Recall that the COSSO penalizes by component sizes, which are functions of both complexity and magnitude. By decoupling sparsity and smoothness, SpAM is more flexible than COSSO.

The optimization problem for SpAM is

$$
\begin{align}
    &\min_{g_j \in H_j} \mathbb{E}\left[Y - \sum_{j=1}^{p} \beta_j g_j(X_j) \right]^2 \\
    &\text{s.t.: } \sum_{j=1}^{p} |\beta_j| \le L \\
                        & \qquad \mathbb{E}\left[g_j^2\right] = 1.
    \label{eq:spam}
\end{align}
$$

where $$Y$$ is an $$n \times 1$$ vector representing the outputs to be predicted, $$X$$ is an $$n \times p$$ data matrix, $$L \ge 0$$ is a penalty constraint, and $$\mathcal{H}_j$$ is a RKHS for $$j=1,...,p.$$

In the LASSO, we penalize the regression coefficients by taking the norm of the $$\beta$$ vector. Here, we take the same idea to encourage sparsity, in addition to adding an additional constraint of $$\mathbb{E}\left[g_j^2\right] = 1$$ to limit the set of functions to search.

We can rewrite the above constraint to be

$$
\begin{align*}
    \min_{f_j \in H_j} &\mathbb{E}\left[Y - \sum_{j=1}^{p} \beta_j g_j(X_j) \right]^2 \\
    \text{subject to: } & \sum_{j=1}^{p} \sqrt{\mathbb{E}\left[f_j^2(X_j)\right]} \le L.
\end{align*}
$$

Or equivalently

$$
\begin{equation}
    \mathcal{L}(f, \lambda) = \frac{1}{2} \mathbb{E}\left[Y - \sum_{j=1}^{p} f_j(X_j) \right]^2 + \lambda \sum_{j=1}^{p} \sqrt{\mathbb{E}\left[f_j^2(X_j)\right]}.
\label{eq:spam_lagrange}
\end{equation}
$$

Proof:

$$
\begin{align*}
    f_j(X_j) &= \beta_j g_j (X_j) \\
    g_j(X_j) &= f_j(X_j) / \beta_j \\
    \mathbb{E}\left[g_j(X_j)^2\right] &= \mathbb{E}\left[f_j(X_j)^2 / \beta_j^2\right] = 1 \\
    \beta_j^2 &= \mathbb{E}\left[f_j^2(X_j)\right]\\
    \beta_j &= \sqrt{\mathbb{E}\left[f_j^2(X_j)\right]} \\
    \sum_{j=1}^{p}  |\beta_j| &= \sum_{j=1}^{p}  \sqrt{\mathbb{E}\left[f_j^2(X_j)\right]} \le L .
\end{align*}
$$

The authors demonstrate that the minimizers can be expressed as the soft-thresholded projection

$$
\begin{equation}
    f_j = \left[ 1 -
            \frac{\lambda}{\sqrt{\mathbb{E}\left[P_j^2\right]}} \right]_+
            \mathbb{E}\left[R_j|X_j\right].
%            P_j.
    \label{eq:spam_minimizers}
\end{equation}
$$

Where residuals excluding the contribution of the $$j$$th component is $$R_j = Y - \sum_{k\neq j}^{} f_k(X_k) $$ and the projection from the residuals onto $$\mathcal{H}_j$$ is

$$
\begin{equation}
    P_j = \mathbb{E}\left[R_j|X_j\right].
    \label{eq:projection_pop}
\end{equation}
$$

Equation (\ref{eq:spam_minimizers}) illuminates the inner workings of SpAM. In particular, we can see that the population minimizer is a soft-thresholded projection onto $$\mathcal{H}_j$$ where the projection $$P_j$$ attempts to reconstruct the signal using information exclusively in the $$j$$th component.

## Algorithm:

The problem with our formulation in (\ref{eq:spam_minimizers}) is that it requires information on the population in $$E[P_j^2]$$ and $$E[R_j \mid X_j]$$. In most practical situations, we will not know the probability distributions and will thus be unable to obtain the expectations. To bridge this gap, we will produce estimates of the expectations.

We may represent projection of residuals onto $$\mathcal{H}_j$$ defined in (\ref{eq:projection_pop}) with the transformation of the residuals by the smoothing matrix $$\mathcal{S}_j$$:

$$
\begin{equation}
    \mathbb{E}\left[P_j\right]
    \approx
    \hat{P}_j
    =
    \mathcal{S}_j R_j.
    \label{eq:projection_data}
\end{equation}
$$

Consequently,

$$
\begin{equation}
    \sqrt{\mathbb{E}\left[P_j^2\right]}
    \approx
    \hat{s}_j
    =
    \frac{1}{\sqrt{n}} \| \hat{P}_j \|
    =
    \sqrt{\text{mean}(\hat{P}_j^2)}.  \\
    \label{eq:projection_data_error}
\end{equation}
$$

One natural algorithm to solve the problem (\ref{eq:spam_lagrange}) is the coordinate descent algorithm. The coordinate descent algorithm is guaranteed to find the global minimum if the function to be optimized can be decomposed into

$$
f(\beta_1, ..., \beta_p) = g(\beta_1, ..., \beta_p) +
\sum_{j=1}^{p} h_j(\beta_j)
$$

where $$g$$ is both convex and differentiable, and $$h_j$$ convex but not necessarily differentiable \cite{hastie_statistical_2016}. Obviously, additive models fit neatly within this framework. The authors call this method \textbf{backfitting}, which can be thought of as a functional version of coordinate descent.

# Discussion

It is unfortunate that the authors of SpAM do not make a direct comparison with the COSSO considering their great similarities. While the authors of both papers apply their methods to the Boston data set, the authors of COSSO only report the prediction error, whereas the authors of SpAM only report the selected model. The authors of SpAM appear to have been more interested in how well their method selected variables rather than its overall performance.

Without a direct comparison, it is difficult to determine where one method is superior to the other. The authors of the SpAM claim that the decoupling of sparsity and smoothing in their method provides flexibility than the COSSO. What cost is incurred by adding this flexibility? Perhaps the flexibility comes at no cost. Perhaps the flexibility comes at a great cost. The question goes unanswered for now and may be the subject of a future investigation.

The authors of both papers present interesting ideas that offer solutions to the problem of selecting components from an additive model when $$p$$ is large. Both techniques are similar as they both rely on $$l_1$$ penalization methods and RKHS theory.
