- Components of NN:
  - Linear layer (matmul)
  - Nonlinear components (activation functions, dropout)
  - Initialize parameters
  - Batch size
  - Hyperparameters (embedding size, number of neurons...)
  - Train, validation, test dataset
  - Validation dataset is for hyperparameters tuning
  - Only run your network with test data a few times, normally at the end of your training + validation
- Your network is a compressed version of your data, the better the data, the higher quality of the compression
- The goal of training is your network has the capability to generalize over unseen data
- View/plot your input, output in 2d, 3d to get a better grasp of what the components are doing
- Start with a tiny, small NN first (ideally one layer of MLP)
- Careful about dead neural (all inputs were zero -> matmul x*w is zero, no gradient)
- We want the input to be Gaussian distributed as we as the output (mean ~= 0 and std ~= 1) -> use kaiming init