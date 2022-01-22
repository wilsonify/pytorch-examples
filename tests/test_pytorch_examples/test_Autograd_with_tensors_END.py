# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="KPXHZpput5Xf" colab_type="text"
# # Autograd with tensors

# +
import torch

# +
w = torch.randn(4, 3, requires_grad=True)

# +
w

# +
w.requires_grad_(False)
w

# +
w.requires_grad_(True)

# +
y = torch.exp(w)
print(y)

# +
print(y.grad_fn)

# +
outp = y.mean()
print(outp)

# +
print(w.grad)

# +
outp.backward()

# +
print(w.grad)

# +
print(w.detach())

# +
print(outp.requires_grad)

with torch.no_grad():
    outp = (w + y).mean()

print(outp.requires_grad)

# +


# +


# +


# +


# +


# +


# +


# +


# +
