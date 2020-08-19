# time-domain-synchrony
Generalised synchrony in time domain


## program objects : definition
#### objects
* state, flow - vector representing dynamics, size=degree of freedom(state space)
* state0 - starting point of dynamics
* t,time - now, timeline beginning from 0
* dynamics - time evolved system, size=(time,size(state))
#### functions
* system : state,t $\rightarrow$ flow
* coupling : stateA,stateB,t $\rightarrow$ flow   
* stabilizer : stateA,stateB,t $\rightarrow$ flow   
