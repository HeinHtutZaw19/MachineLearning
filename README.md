# MACHINE LEARNING

## Supervised Learning

### Linear Regression

- **Linear Model**
    
    $f_w,_b(x^{(i)}) =wx^{(i)} + b$ 
    
- **Cost Function**
    
    $J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_w,_b(x^{(i)}) - y^{(i)})^2$

![Linear Regression Cost](include/imgs/linear_regression_cost.png)
*Figure 1: Mean Squared Error (MSE) loss function provides a convex loss function, which ensures that there is a single global minimum and no local minima.
    
- **Gradient Descent**
    
    repeat until convergence
  
    $$\color{black}\small\begin{align*}
    \; \color{black} {w - \alpha \frac{\partial J(w, b)}{\partial w}}  \\  \color{black} {b - \alpha \frac{\partial J(w, b)}{\partial b}} \\
    \end{align*}$$

  
    
- **Partial Derivatives
    
    $\frac{\partial J(w,b)}{\partial w} & = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$
    $\frac{\partial J(w,b)}{\partial b} & = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})$
    
    - **Simultaneously update w and b parameters**

https://github.com/HeinHtutZaw19/Youtube/blob/main/include/imgs/gradient_descend.mp4
*Figure 2: Gradient Descend In Action* 

- **Z-sore Normalization**

    - In order to make the contour uniform in all axes and improve efficiency of gradient descend, z score normalization is used.
      
  $x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j} \tag{1}$
  $\mu_j &= \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j \tag{2}\\\sigma^2_j &= \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2  \tag{3}$

