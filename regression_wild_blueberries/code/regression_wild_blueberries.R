#------------------------------------------------------------------------------#
# Regression with a wild blueberry yield dataset
# based on kaggle competition:
# https://www.kaggle.com/competitions/playground-series-s3e14/overview
#
# The goal of this exercise is to predict the yield of blueberry bushes based
# on the other features.
# First, we will have a look at the data. Then, we will try to create a model
# that can predict yield.
#------------------------------------------------------------------------------#
library(data.table)
library(ggplot2)
library(car)

setwd("D:/projects/regression_wild_blueberries/")

#------------------------------------------------------------------------------#
# Import data

train <- fread("inputs/train.csv")
test <- fread("inputs/test.csv")

head(train)
head(test)

#------------------------------------------------------------------------------#
# Exploring the data

# check if there is any missing values
print(train[, lapply(.SD, function(x) sum(is.na(x)))])
print(test[, lapply(.SD, function(x) sum(is.na(x)))])
# no missing values in any of the variables

# plots
colnames <- as.list(names(train))
for(col in colnames){
  p <- ggplot(train, aes(x = .data[[col]], y = yield)) +
    geom_point(color = "blue", size = 2, alpha = 0.6) +
    labs(title = paste0("Scatter Plot of ", col, " by Yield"),
         x = col,
         y = "Yield") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
    
  print(p)
}
# We can focus on variables fruitset, fruitmass and seeds for the moment

#------------------------------------------------------------------------------#
# Linear regression

linear_regression <- lm(yield ~ fruitset, train)
summary(linear_regression)

# Residuals:
#   Min      1Q  Median      3Q     Max 
# -5546.2  -248.1   -23.7   226.5  5334.4 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -1980.51      34.26  -57.81   <2e-16 ***
#   fruitset    15924.11      67.42  236.21   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 620.1 on 15287 degrees of freedom
# Multiple R-squared:  0.7849,	Adjusted R-squared:  0.7849 
# F-statistic: 5.579e+04 on 1 and 15287 DF,  p-value: < 2.2e-16

# R-squared is 0.78, which indicates a strong fit.
# Variable fruitset is statistically significant

# checks
# multicollinearity - we removed fruitmass and seeds in order
# to take into account multicollinearity
# vif_values <- vif(linear_regression)
# print(vif_values)

# heteroscedasticity
plot(linear_regression$fitted.values, linear_regression$residuals)
abline(h = 0, col = "red")

# normality of residuals
qqnorm(linear_regression$residuals)
qqline(linear_regression$residuals, col = "red")

#------------------------------------------------------------------------------#
# Predicting results
test$yield <- predict(linear_regression, test)
head(test)

#------------------------------------------------------------------------------#
# Exporting results
submission <- test[, c("id", "yield")]
fwrite(submission, "outputs/submission.csv", row.names = FALSE)




