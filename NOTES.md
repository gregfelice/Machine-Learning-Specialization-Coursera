
# Supervised Machine Learning: Regression and Classification 

## Key Concepts

### Linear Regression Model

**Model**
$f_{(w,b)}=wx+b$

**Parameters**
$w,b$

**Cost Function**
* Squared error cost function
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

where
* $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$
* $(f_{w,b}(x^{(i)})-y^{(i)})^2$ is the squared difference between the target value and the prediction.
* These differences are summed over all the $m$ examples and divided by $2m$ to produce the cost, $J(w,b)$.

**Objective**

$$\underset{w,b}{\text{minimize}\space} J(w,b)$$

### Gradient Descent

Gradient descent can be described as:

repeat until convergence: {
$$w=w-{\alpha}\frac{\partial J(w,b)}{\partial w}$$
$$b=b-{\alpha}\frac{\partial J(w,b)}{\partial b}$$
}

where, parameters $𝑤, 𝑏$ are updated simultaneously.

### Cost With Multiple Variables

The equation for the cost function with multiple variables $J(w,b)$ is:

$$J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

where:
$$f_{w,b} (x^{(i)}) = w \cdot x^{(i)}+b$$

In contrast to previous labs, $𝐰$ and $𝐱(𝑖)$ are vectors rather than scalars supporting multiple features.

### Gradient Descent With Multiple Variables

Gradient descent for multiple variables:

repeat until convergence: {
$$for\space j=0..n-1$$

$$w_j = w_j - \alpha \frac{\partial J(w,b)}{\partial w_j}$$
$$b = b - \alpha \frac{\partial J(w,b)}{\partial b}$$
}
where, n is the number of features, parameters $w_j, b$ aer updated simultaneously and where

$$\frac{\partial J(w,b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)}_{j}$$

$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})$$  

* m is the number of training examples in the data set
* $f_{f,w}(x^{(i)})$ is the model's prediction, with $y^{(i)}$ is the target value

### Logistic Regression Model & Decision Boundary

For logistic regression, the model is represented as:

$$f_{w,b} = g(w \cdot x^{(i)} + b)$$

where $g(z)$ is known as the signmoid function and it maps all input values to values between 0 and 1:

$$g(z) = \frac {1}{1+e^{-z}}$$

and $$w \cdot x$$ is the vector dot product:

$$w \cdot x = w_0x_0+w_1x_1$$

We interperet the output of the model $(f_w,b(x))$ as the probability that y=1 given x and parameterized by w and b.

* Therefore, to get a final prediction (y = 0 or y = 1) from the logistic regression model, we can use the following heuristic: 

if $f_{w,b}(x) >= 0.5$, predict y = 1

if $f_{w,b}(x) < 0.5$. predict y = 0

### Logistic Loss

Squared Error Cost

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

where

$$f_{w,b}(x^{(i)}) = \text{sigmoid}(wx^{(i)} + b)$$

**Logistic Loss Function**

Logistic Regression uses a loss function more suited to the task of categorization where the target is 0 or 1 rather than any number. 
* Loss is a measure of the difference of a single example to its target value while
* The Cost is a measure of the losses over the training set

This is defined:

* $loss(f_{w,b}(x^{(i)}),y^{(i)})$ is the coxt for a single data point, which is:

$$
loss(f_{w,b}(x^{(i)}),y^{(i)}) = 
\begin{cases}
   -log(f_{w,b}(x^{(i)}))  &\text{if } y^{(i)}=1 \\
   -log(1 - f_{w,b}(x^{(i)}))  &\text{if } y^{(i)}=0 \\
\end{cases}
$$

* $f_{w,b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value.
* $f_{w,b}(x^{(i)}) = g(w \cdot x^{(i)} + b)$ where function g is the sigmoid function.

The defining feature of this loss function is the fact that it uses two separate curves. One for the case when the target is zero or $(𝑦=0)$ and another for when the target is one $(𝑦=1)$. Combined, these curves provide the behavior useful for a loss function, namely, being zero when the prediction matches the target and rapidly increasing in value as the prediction differs from the target. 

Combined, the curves are similar to the quadratic curve of the squared error loss. Note, the x-axis is 𝑓𝐰,𝑏
which is the output of a sigmoid. The sigmoid output is strictly between 0 and 1.

The loss function above can be rewritten to be easier to implement.
    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$
  
This is a rather formidable-looking equation. It is less daunting when you consider $y^{(i)}$ can have only two values, 0 and 1. One can then consider the equation in two pieces:  
when $ y^{(i)} = 0$, the left-hand term is eliminated:

$$
\begin{aligned}
\text{loss}(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log(f_{\mathbf{w},b}( \mathbf{x}^{(i)} )) - ( 1 - 0) \log ( 1 - f_{\mathbf{w},b}( \mathbf{x}^{(i)} ))) \\
&= -\log ( 1 - f_{\mathbf{w},b}( \mathbf{x}^{(i)} ))
\end{aligned}
$$

and when $ y^{(i)} = 1$, the right-hand term is eliminated:

$$
\begin{aligned}
  \text{loss}(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log(f_{\mathbf{w},b}( \mathbf{x}^{(i)} )) - ( 1 - 1) \log ( 1 - f_{\mathbf{w},b}( \mathbf{x}^{(i)} )))\\
  &=  -\log(f_{\mathbf{w},b}( \mathbf{x}^{(i)} ))
\end{aligned}
$$

With this new logistic loss function, a cost function can be produced that incorporates the loss from all the examples. 


### Regularization
**Cost Function for Regularized Linear Regression**

The equation for the cost function regularized linear regression is:
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2  + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{1}$$ 
where:
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{2} $$ 

Compare this to the cost function without regularization (which you implemented in  a previous lab), which is of the form:

$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 $$ 

The difference is the regularization term,  <span style="color:blue">
    $\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$ </span> 
    
Including this term encourages gradient descent to minimize the size of the parameters. Note, in this example, the parameter $b$ is not regularized. This is standard practice.

**Cost Function for Regularized Logistic Regression**

For regularized **logistic** regression, the cost function is of the form
$$J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{3}$$
where:
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = sigmoid(\mathbf{w} \cdot \mathbf{x}^{(i)} + b)  \tag{4} $$ 

Compare this to the cost function without regularization (which you implemented in  a previous lab):

$$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right] $$

As was the case in linear regression above, the difference is the regularization term, which is    <span style="color:blue">
    $\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$ </span> 

Including this term encourages gradient descent to minimize the size of the parameters. Note, in this example, the parameter $b$ is not regularized. This is standard practice. 

### Gradient Descent with Regularization

**Computing the Gradient with regularization (both linear/logistic)**

The gradient calculation for both linear and logistic regression are nearly identical, differing only in computation of $f_{\mathbf{w}b}$.

$$\begin{aligned}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}  +  \frac{\lambda}{m} w_j \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
\end{aligned}$$

* m is the number of training examples in the data set      
* $f_{\mathbf{w},b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target

      
* For a  <span style="color:blue"> **linear** </span> regression model  
    $f_{\mathbf{w},b}(x) = \mathbf{w} \cdot \mathbf{x} + b$  
* For a <span style="color:blue"> **logistic** </span> regression model  
    $z = \mathbf{w} \cdot \mathbf{x} + b$  
    $f_{\mathbf{w},b}(x) = g(z)$  
    where $g(z)$ is the sigmoid function:  
    $g(z) = \frac{1}{1+e^{-z}}$   
    
The term which adds regularization is the <span style="color:blue">
$\frac{\lambda}{m} w_j $
</span>


