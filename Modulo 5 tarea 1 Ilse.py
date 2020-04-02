#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# ## Optimización de los pesos

# In[2]:


data = pd.read_excel("./PreciosPortf.xlsx", index_col=0)
data.head()


# In[3]:


rend = data.pct_change().dropna()
rend_ports = rend.mean() * 252
rend_ports


# In[4]:


rend.cov()


# In[5]:


weights = np.random.random(5)
weights /= sum(weights)
weights


# In[6]:


sum(weights)


# In[7]:


E_p = (rend.mean() * weights).sum() *252 #rendimiento esperado anualizado del portafolio
S_p = np.sqrt(weights.T @ (rend.cov() *252) @ weights)


# In[8]:


E_p, S_p


# In[9]:


def port_rend(weights, r):
    E_p =(r.mean() @ weights) * 252
    return E_p
def port_vol(weights, r):
    S_p = np.sqrt(weights.T @ (r.cov() *252) @ weights)
    return S_p


# In[10]:


port_rend(weights, rend), port_vol(weights, rend)


# In[11]:


weights_sim = np.random.random([5_000,5])
weights_sim = weights_sim / weights_sim.sum(axis=1, keepdims=True)
weights_sim.sum(axis=1)


# In[12]:


rend_sim = np.apply_along_axis(port_rend, 1, weights_sim, r=rend)
col_sim = np.apply_along_axis(port_vol, 1, weights_sim, r=rend)


# In[13]:


sharpe_ratio = rend_sim / col_sim
plt.scatter(col_sim, rend_sim, c = sharpe_ratio)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel(r"$\sigma^2_r$", fontsize=13)
plt.ylabel("E(r)",fontsize=13);


# ## Creando la frontera eficiente

# In[14]:


from scipy.optimize import minimize


# In[15]:


N, M  = rend.shape
w0 = np.random.randn(M)

def sum_weights(weights):
    return weights.sum() -1

constraints = [
    {"type":"eq", "fun":sum_weights}
]
port0 = minimize(port_vol, w0, constraints=constraints, args=rend)
port0


# In[16]:


wp0 = port0.x
port_rend(wp0,rend),port_vol(wp0,rend)


# In[17]:


sharpe_ratio = rend_sim / col_sim
plt.scatter(col_sim, rend_sim, c = sharpe_ratio)
plt.scatter(port_vol(wp0,rend),port_rend(wp0,rend))
plt.colorbar(label="Sharpe Ratio")
plt.xlabel(r"$\sigma^2_r$", fontsize=13)
plt.ylabel("E(r)",fontsize=13);


# # TAREA 

# ### Optimización sujeta a un rendimiento esperado Y a condición adicional

# In[18]:


N, M  = rend.shape
w0 = np.random.randn(M)

def sum_weights(weights):
    return weights.sum() -1 # == 0

def rend_esperado(w, E):
    return port_rend(w, rend) - E # == 0
e0 = .1
constraints = [
    {"type":"eq", "fun":sum_weights},
    {"type":"eq", "fun":lambda w: rend_esperado(w, e0)}
]
port1 = minimize(port_vol, w0, constraints=constraints, args=rend)
port1


# In[19]:


wp1 = port1.x
port_rend(wp1,rend),port_vol(wp1,rend)


# In[20]:


sharpe_ratio = rend_sim / col_sim
plt.scatter(col_sim, rend_sim, c = sharpe_ratio)
plt.scatter(port_vol(wp1,rend),port_rend(wp1,rend))
plt.colorbar(label="Sharpe Ratio")
plt.xlabel(r"$\sigma^2_r$", fontsize=13)
plt.ylabel("E(r)",fontsize=13);


# In[21]:


#rmin = port_rend(wp0,rend)
rmin = rend_ports.min()
rmax = rend_ports.max()


# In[22]:


rend_maxs = np.linspace(rmin,rmax)
rend_maxs


# añado la condicion:
# ninguna accion tiene peso mayor a .1

# In[30]:


def maximo(weights):
    return .1-weights.max()


# In[31]:


pesos_optimos = []
for e in rend_maxs:
    constraints = [
        {"type":"eq", "fun":sum_weights},
        {"type":"eq", "fun":lambda w: rend_esperado(w, e)},
        {"type":"ineq", "fun":lambda w: maximo(w)}
    ]
    port1 = minimize(port_vol, w0, constraints=constraints, args=rend)
    w_opt = port1.x
    pesos_optimos.append(w_opt)


# In[32]:


r_opt = []
v_opt = []
for w in pesos_optimos:
    r_opt.append(port_rend(w,rend))
    v_opt.append(port_vol(w,rend))
    
r_opt = np.array(r_opt)
v_opt = np.array(v_opt)


plt.scatter(v_opt,r_opt, c=r_opt/v_opt)

plt.scatter(col_sim,rend_sim, c=sharpe_ratio, alpha = 0.1, s=5)
plt.colorbar()


# ### Portafolio Tangencial

# In[34]:


rf = 0.06
def min_func_sharpe(weights):
        return - (port_rend(weights, rend) - rf)/ port_vol(weights,rend)

constraints = [
    {"type":"eq", "fun":sum_weights}
]
f_sharpe = minimize(min_func_sharpe,w0, constraints=constraints)


# In[35]:


w_sharpe = f_sharpe.x


# In[36]:


w_sharpe


# In[37]:


e_sharpe = port_rend(w_sharpe, rend)
vol_sharpe = port_vol(w_sharpe, rend)


# In[38]:


sigma_c = np.linspace(0,0.25)
sharpe = rf + sigma_c*(e_sharpe - rf)/vol_sharpe
plt.plot(sigma_c, sharpe, linestyle="--",c="tab:orange", label="Capital Allocation Line")
plt.plot(v_opt, r_opt)
#plt.plot(v_opt_c,r_opt_c)
plt.legend(fontsize=12)


# In[ ]:





# In[ ]:





# In[ ]:




