# DDP_General
Vanila DDP with regularization and line search. Work built up from Hassan Almubarak's codebase.

Arthur Scaquetti do Nascimento (nascimento@gatech.edu) - CORE Lab @ Georgia Tech

### Please DO NOT use this codebase without contacting the author

### TBD - update instructions
Follow those steps to run any example
1. Call the system's dynamics
2. Generate the obstacle course and define the safe set function (h)
3. Call DBaS_dyn to generate the DBaS dynamics
4. Call Safety_Embedding_dynamics to augment the DBaS to the system's dynamics
5. Define DDP and optimization paramters and run vanilla DDP
