# Supervised Machine Learning: Regression and Classification 

## Key Concepts

### Linear Regression Model

**Model**
$$
f_{(w,b)}=wx+b
$$

**Parameters**
$$
w,b
$$

**Cost Function**
* Squared error cost function
$$
J(w,b)=
\frac {1}{2m}
\sum^{m}_{i=1}
(f_{w,b}(x^{(i)})-y^{(i)})^2
$$
where
* <span class="math-inline"><0\>f\_\{w,b\}\(x^\{\(i\)\}\)</span> is our prediction for example <span class="math-inline">i</span> using parameters <span class="math-inline">w,b</span>
* <span class="math-inline">\(f\_\{w,b\}\(x^\{\(i\)\}\)\-y^\{\(i\)\}\)^2</span> is the squared difference between the target value and the prediction.
* These differences are summed over all the <span class="math-inline">m</span> examples and divided by <span class="math-inline">2m</span> to produce the cost, <span class="math-inline">J\(w,b\)</span>.

**Objective**
$$
\underset{w,b}{\text{minimize}} J(w,b) 
$$

### Gradient Descent

Gradient descent can be described as:

repeat until convergence: \{
$$
w=w-{\alpha}
\frac{\partial J(w,b)}{\partial w}
$$
$$
b=b-{\alpha}
\frac{\partial J(w,b)}{\partial b}
$$
\}

where, parameters <span class="math-inline">w</span>, <span class="math-inline">b</span> are updated simultaneously.

### Cost With Multiple Variables

The equation for the cost function with multiple variables <span class="math-inline">J\(w,b\)</span> is:

$$
J(w,b)=
\frac {1}{2m}
\sum^{m-1}_{i=0}
(f_{w,b}(x^{(i)})-y^{(i)})^2
$$

where:
$$
f_{w,b} (x^{(i)}) = w \cdot x^{(i)}+b
$$

In contrast to previous labs, <span class="math-inline">\\mathbf\{w\}</span> and <span class="math-inline">\\mathbf\{x\}^\{\(i\)\}</span> are vectors rather than scalars supporting multiple features.

### Gradient Descent With Multiple Variables

Gradient descent for multiple variables:

repeat until convergence: \{
<span class="math-block">\\text\{for \} j\=0\.\.n\-1</span> 
$$
w_j=w_j-{\alpha}
\frac{\partial J(w,b)}{\partial w_j}
$$
$$
b=b-{\alpha}
\frac{\partial J(w,b)}{\partial b}
$$
\}
where, <span class="math-inline">n</span> is the number of features, parameters <span class="math-inline">w\_j</span>, <span class="math-inline">b</span> are updated simultaneously and where
$$
\frac{\partial J(w,b)}{\partial w_j} = 
\frac {1}{m}
\sum^{m-1}_{i=0}
(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}_{j}
$$
$$
\frac{\partial J(w,b)}{\partial b} = 
\frac {1}{m}
\sum^{m-1}_{i=0}
(f_{w,b}(x^{(i)})-y^{(i)})
$$
* <span class="math-inline">m</span> is the number of training examples in the data set
* <span class="math-inline">f\_\{w,b\}\(x^\{\(i\)\}\)</span> is the model's prediction, with <span class="math-inline">y^\{\(i\)\}</span> is the target value

### Logistic Regression Model & Decision Boundary

For logistic regression, the model is represented as:
$$
f_{w,b} = g(w \cdot x^{(i)} + b)
$$
where <span class="math-inline">g\(z\)</span> is known as the sigmoid function and it maps all input values to values between 0 and 1:
$$
g(z) = \frac {1}{1+e^{-z}}
$$
and <span class="math-inline">w \\cdot x</span> is the vector dot product:
$$
w \cdot x = w_0x_0+w_1x_1
$$
We interpret the output of the model <span class="math-inline">\(f\_\{w,b\}\(x\)\)</span> as the probability that <span class="math-inline">y\=1</span> given <span class="math-inline">x</span> and parameterized by <span class="math-inline">w</span> and <span class="math-inline">b</span>.

* Therefore, to get a final prediction (<span class="math-inline">y \= 0</span> or <span class="math-inline">y \= 1</span>) from the logistic regression model, we can use the following heuristic: 

if <span class="math-inline">f\_\{w,b\}\(x\) \\ge 0\.5</span>, predict <span class="math-inline">y \= 1</span>

if <span class="math-inline">f\_\{w,b\}\(x\) < 0\.5</span>, predict <span class="math-inline">y \= 0</span>

### Logistic Loss

Squared Error Cost
$$
J(w,b)=
\frac {1}{2m}
\sum^{m}_{i=1}
(f_{w,b}(x^{(i)})-y^{(i)})^2
$$

where
$$
f_{w,b}(x^{(i)})= \text{sigmoid}(wx^{(i)} + b)
$$

**Logistic Loss Function**

Logistic Regression uses a loss function more suited to the task of categorization where the target is 0 or 1 rather than any number. 
* Loss is a measure of the difference of a single example to its target value while
* The Cost is a measure of the losses over the training set

This is defined:

* <span class="math-inline">\\text\{loss\}\(f\_\{w,b\}\(x^\{\(i\)\}\),y^\{\(i\)\}\)</span> is the cost for a single data point, which is:

$$
\text{loss}(f_{w,b}(x^{(i)}),y^{(i)}) = 
\begin{cases}
   -\log(f_{w,b}(x^{(i)}))  &\text{if } y^{(i)}=1 \\
   -\log(1 - f_{w,b}(x^{(i)}))  &\text{if } y^{(i)}=0 \\
\end{cases}
$$

* <span class="math-inline">f\_\{w,b\}\(x^\{\(i\)\}\)</span> is the model's prediction, while <span class="math-inline">y^\{\(i\)\}</span> is the target value.
* <span class="math-inline">f\_\{w,b\}\(x^\{\(i\)\}\) \= g\(w \\cdot <4\>x^\{\(i\)\} \+ b\)</span> where function <span class="math-inline">g</span> is the sigmoid function.

The defining feature of this loss function is the fact that it uses two separate curves. One for the case when the target is zero (<span class="math-inline"><4\>y\=0</span>) and another for when the target is one (<span class="math-inline">y\=1</span>). Combined, these curves provide the behavior useful for a loss function, namely, being zero when the prediction matches the target and rapidly increasing in value as the prediction differs from the target. 

Combined, the curves are similar to the quadratic curve of the squared error loss. Note, the x-axis is <span class="math-inline">f\_\{<4\>w,b\}</span> which is the output of a sigmoid. The sigmoid output is strictly between 0 and 1.

The loss function above can be rewritten to be easier to implement.
    <span class="math-block">\\text\{<5\>loss\}\(f\_\{\\mathbf\{w\},b\}\(\\mathbf\{x\}^\{\(i\)\}\), y^\{\(i\)\}\) \= \(\-y^\{\(i\)\} \\log\\<4\>left\(f\_\{\\mathbf\{w\},b\}\\left\( \\mathbf\{x\}^\{\(i\)\} \\right\) \\right\) \- \\left\( 1 \- y^\{\(i\)\}\\right\) \\log \\left\( 1 \- f\_\{\\mathbf\{w\},b\}\\left\( \\mathbf\{x\}^\{\(i\)\} \\right\) \\right\)</span>
  
This is a rather formidable-looking equation. It is less daunting when you consider <span class="math-inline">y^\{\(i\)\}</span> can have only two values, 0 and 1. One can then consider the equation in two pieces:  
when $ y^{(i)} = 0$, the left-hand term is eliminated:
$$
\begin{align}
\text{loss}(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 0\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \\
&= -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$
and when $ y^{(i)} = 1$, the right-hand term is eliminated:
$$
\begin{align}
  \text{loss}(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 1\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\\
  &=  -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

With this new logistic loss function, a cost function can be produced that incorporates the loss from all the examples. 


### Regularization
**Cost Function for Regularized Linear Regression**

The equation for the cost function regularized linear regression is:
$$J