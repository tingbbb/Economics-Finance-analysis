# Economics-Finance-analysis
This repository contains the code used for constructing and backtesting an equity signal in my research. The analysis is based on Logit prediction, portfolio method, Fama MacBeth regression method, and four factor model. The dataset used for this research contains confidential information and cannot be shared. 


# Requirements:
- Python 3.x
- Pandas
- NumPy
- Statsmodels
- Matplotlib 

# Usage
   Each script in this repository corresponds to a specific part of the analysis.
   
- `LogitPrediction.py`: run Logit model to predict an event probability, which will then be used as a signal. 
- `RetCalculation.py`: use the above signal to construct a portfolio and calculate excess return.
- `Alpha.py`: use the above excess return to estimate three-factor, four-factor, five-factor alphas.
- `Plot.py`: make graphs of the alphas over time. 
- `FamaMacBeth reg.py`: run Fama MacBeth to test if the above signal explains cross-sectional stock returns.



