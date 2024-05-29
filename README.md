# MACHINE LEARNING

## Supervised Learning

### Linear Regression

- **Linear Model**
    
    $f_w,_b(x^{(i)}) =wx^{(i)} + b$ 
    
- **Cost Function**
    
    $J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_w,_b(x^{(i)}) - y^{(i)})^2$


![Linear Regression Cost Function](include/imgs/LR/linear_regression_cost.png)
*Figure 1: Mean Squared Error (MSE) loss function provides a convex loss function, which ensures that there is a single global minimum and no local minima.
    
![Equations](include/imgs/LR/equations.png)
    

<div style="text-align:center;">
    <img src="include/imgs/LR/gradient_descend_LR.gif" alt="Gradient Descent" />
    <p><em>Figure 2: Gradient Descent In Action</em></p>
</div>

- **Z-sore Normalization**

    - In order to make the contour uniform in all axes and improve efficiency of gradient descend, z score normalization is used.
![Z-score Normalization](include/imgs/LR/normalization.png)
