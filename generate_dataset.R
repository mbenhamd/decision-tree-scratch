# A small training set from Bouzy's course :
# http://www.math-info.univ-paris5.fr/~bouzy/Doc/AA1/InductionDecisionTree.pdf
# Slide 6
# Script generate data randomly

outlook_possible_values = c("sunny","overcast","rain")
temperature_possible_values = c("hot","mid","cold")
humidity_possible_values = c("high","normal","low")
windy_possible_values = c("true","false")
C_possible_values = c("N","P")






s = as.matrix.data.frame(data)
varCpy = s

varCpy

# Get index from all possible values 

for (i in seq(1:nrow(varCpy))) {
  for (j in seq(1:ncol(varCpy))) {
    print(varCpy[i,j])
  }
}
