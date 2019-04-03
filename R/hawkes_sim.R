a  = simulateHawkes(0.5, 0.9, 2, 10000)
write.csv(a[[1]], file = "../practice/hsim.csv", row.names=FALSE)
likelihoodHawkes(0.5, 0.9, 2, a[[1]])