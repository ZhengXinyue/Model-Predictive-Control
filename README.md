# Model Predictive Control for Pendulum-v0

Pure MPC uses the real model for simulation.    

Learning MPC uses a learned system dynamics with neural network for simulation.


## Installation
```
pip install numpy matplotlib gym 
pip install mxnet  (or mxnet-cu90 which corresponds your cuda version)
```

Examples:
  


Pure_MPC with different rollouts and horizon:   
![image](https://github.com/ZhengXinyue/Model-Predictive-Control/blob/master/Pure_MPC/MPC_Pendulum_v0.png) 