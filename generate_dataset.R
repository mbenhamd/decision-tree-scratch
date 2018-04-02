# A small training set from Bouzy's course :
# http://www.math-info.univ-paris5.fr/~bouzy/Doc/AA1/InductionDecisionTree.pdf
# Slide 6
# Script generate data randomly

file = read.csv("test_data.csv")
s = as.matrix.data.frame(file)

# Names of columns
namescol = c("outlook","temperature","humidity","windy","C")



# All possible values for each columns
outlook_possible_values = c("sunny","overcast","rain")
temperature_possible_values = c("hot","mid","cold")
humidity_possible_values = c("high","normal","low")
windy_possible_values = c("true","false")
C_possible_values = c("N","P")

#   Matrix representing all possible values
matrix_all_values = cbind(outlook_possible_values,temperature_possible_values,humidity_possible_values,windy_possible_values,C_possible_values)


# Vector of possible indexes for generating a line
rand_values_for_all = c(floor(runif(1, 1,4)),floor(runif(1, 1,4)),floor(runif(1, 1,4)),floor(runif(1, 1,3)),floor(runif(1, 1,3)))

# Line computed
ligne = cbind(outlook_possible_values[rand_values_for_all[1]],temperature_possible_values[rand_values_for_all[2]],humidity_possible_values[rand_values_for_all[3]],windy_possible_values[rand_values_for_all[4]],C_possible_values[rand_values_for_all[5]])


# Function that will generate a data dataframe of n random lines
generate_n_line = function(nline = 100){
    final = vector()
    for( i in seq(1:nline)){
    rand_values_for_all = c(floor(runif(1, 1,4)),floor(runif(1, 1,4)),floor(runif(1, 1,4)),floor(runif(1, 1,3)),floor(runif(1, 1,3)))    
    ligne = cbind(outlook_possible_values[rand_values_for_all[1]],temperature_possible_values[rand_values_for_all[2]],humidity_possible_values[rand_values_for_all[3]],windy_possible_values[rand_values_for_all[4]],C_possible_values[rand_values_for_all[5]])  
    final=rbind(final,ligne)
    }
    colnames(final) = namescol
    return(final)
}

# Test
test = generate_n_line(10000)
test

# Write into a csv
write.csv(test,"random_data.csv",row.names=FALSE)

