# practical3_gradient_descent.py
# Requirements: matplotlib, numpy

import numpy as np
import matplotlib.pyplot as plt

# Function and gradient
def f(x): return (x + 3)**2
def grad(x): return 2*(x + 3)

def gradient_descent(start_x=2.0, lr=0.1, max_iter=100, tol=1e-8):
    x = start_x
    history = [x]
    for i in range(max_iter):
        g = grad(x)
        x_new = x - lr * g
        history.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x, history

min_x, hist = gradient_descent(start_x=2.0, lr=0.1, max_iter=200)
print("Local minimum at x=", round(min_x,6), " f(x)=", round(f(min_x),8))

# Plot function and steps
xs = np.linspace(-6, 2, 200)
ys = f(xs)
plt.figure(figsize=(7,4))
plt.plot(xs, ys, label='f(x)=(x+3)^2')
plt.scatter(hist, [f(x) for x in hist], color='red', s=30, label='GD steps')
for i, x in enumerate(hist):
    if i in (0,1,len(hist)-1):
        plt.text(x, f(x)+0.1, f"{round(x,3)}", fontsize=8)
plt.legend()
plt.title("Gradient Descent steps on f(x)=(x+3)^2")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.grid(True)
plt.show()
