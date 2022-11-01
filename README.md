# DENN: Dynamic Epidemiological Neural Networks for Regions Diversity in COVID-19 with High-Precision and High-Interpretation
 
Abstract: A novel coronavirus disease 2019 has spread worldwide and made huge influences across the world recently. Modeling the infectious spread situation of COVID-19 is essential to understand the current condition and formulate intervention measurements. Traditional epidemiological equations based on the SEIR model stimulate susceptibles, exposed, infectious, recovered groups. The traditional parameter estimation method to solve SEIR equations could not precisely fit real-world data due to different social distancing policies and intervention strategies. Additionally, learning-based models achieve outstanding fitting performance, but their interpretability is comparably low. Thus, in our paper, we propose a DENN (Dynamic Epidemiological Neural Networks) method that combines epidemiological equations and deep-learning advantages to obtain high-precision and high interpretation. The DENN contains neural networks to fit the effect function to simulate the ever-changing social distance and isolation measurements based on the neural ODE method in solving variants' equations, ensuring the diversity of regions and fitting performance. We introduce four SEIR variants to fit different situations in different countries and regions. We compare our DENN method with traditional parameter estimation methods (Nelder-Mead, BFGS, Powell, Truncated Newton Conjugate-Gradient, Neural ODE) in fitting the real-world data in the cases of countries (the USA, Columbia, South Africa) and regions (Wuhan in China, Piedmont in Italy). Our DENN method achieves the best Mean Square Error and Pearson coefficient in all five areas. Further, compared with the state-of-art learning-based approaches, the DENN outperforms all techniques, including LSTM, RNN, GRU, Random Forest, Extremely Random Trees, and Decision Tree.
 
## Graphic Abstract

 
## SIR\SEIR models
 
The original SEIR models are simple and only use the infectious rate, exposed rate, recovered rate to describe the development trend of infectious diseases. However, the infection situation in the real world is complicated. Some existing studies based on SIR or SEIR models propose modeling improvements to fit the different population trends.
 
## Varients of SIR\SEIR models

### SIRD
When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are 
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

The Cauchy-Schwarz Inequality

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
### SEIRD
### SMCRD
### SEMCRD
