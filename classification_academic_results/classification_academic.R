#--------------------------------------------------------------------------------------------------------
# Classification of academic outputs
#
# based on kaggle competition here:
# https://www.kaggle.com/competitions/playground-series-s4e6/overview
#
# The goal is to predict academic risk of students in higher education
# The variable of interest here is a categorical variable called 'Target', which can take 3 values:
# - Graduate
# - Enrolled
# - Dropout
# Using the rest of the dataset, we want to find a way to predict this variable
# We will explore the data in a first step, to understand it and see what variables can help us
# After that, we will try to implement a model that we can use on the test dataset
#--------------------------------------------------------------------------------------------------------
library(data.table)
library(dplyr)
library(ggplot2)

library(randomForest)
library(xgboost)
library(caret)

inpath <- "D:/projects/classification_academic_results/inputs/"
outpath <- "D:/projects/classification_academic_results/outputs/"

# Import data
train <- fread(paste0(inpath,"train.csv"))
test <- fread(paste0(inpath,"test.csv"))

#--------------------------------------------------------------------------------------------------------
# Exploring the data with visualizations

# Distribution of "Target" variable
ggplot(train, aes(x = Target)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of the Target Variable",
       x = "Target",
       y = "Count")

# The most represented category is 'Graduate' (around 47% of the observations),
# then 'Dropout' (33% of observations), and finally 'Enrolled' (19.5% of observations)

# Density plot of previous qualification grade by 'Target' category
ggplot(train, aes(x = `Previous qualification (grade)`, fill = Target)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Previous Qualification Grade by Target",
       x = "Previous Qualification Grade",
       y = "Density")

# Here, we see that dropouts tend to have average previous grades,
# while graduates and enrolled students have more varied distributions.

# Boxplot of age by target category
ggplot(train, aes(x = Target, y = `Age at enrollment`, fill = Target)) +
  geom_boxplot() +
  labs(title = "Boxplot of Age by Target Category",
       x = "Target Category",
       y = "Age") +
  theme_minimal()

# For graduates, the median age is slightly lower compared to the other categories,
# with most students graduating around the age of 18-20.
# Enrolled students show a similar distribution as graduates, though the distribution is a bit more wide
# The median age of dropouts is slightly higher, with a larger interquartile range,
# This suggests a more varied age distribution among students who drop out.


# Distribution of gender by Target category
ggplot(train, aes(x = as.factor(Gender), fill = Target)) +
  geom_bar(position = "dodge") +
  labs(title = "Gender Distribution by Target Category",
       x = "Gender",
       y = "Count") +
  theme_minimal()

targetcategory <- unique(train[, Target])
gendercategory <- unique(train[, Gender])

for(gender in gendercategory){
  for(target in targetcategory){
    percentage <- nrow(train[Gender==gender&Target==target,])/nrow(train[Gender==gender,])*100
    label <- ifelse(gender==1, 'males', 'females')
    print(paste0("Around ", round(percentage, 2), "% of ", label, " are ", target))
  }
}

# Around 68% of our observations are females, among which:
# "Around 57.75% of females are Graduate"
# "Around 23.53% of females are Dropout"
# "Around 18.73% of females are Enrolled"

# Around 32% of our observations are males, among which:
# "Around 25.04% of males are Graduate"
# "Around 53.71% of males are Dropout"
# "Around 21.25% of males are Enrolled"

# Most observations in our train dataset are females. They tend to do better than males,
# with a much higher percentage of graduates (58% vs 25% for men) and lower percentage of dropouts (23.5% vs 54% for men)

# Scatter plot of admission grade vs. previous qualification grade
ggplot(train, aes(x = `Admission grade`, y = `Previous qualification (grade)`, color = Target)) +
  geom_point(alpha = 0.7) +
  labs(title = "Scatter Plot of Admission Grade vs. Previous Qualification Grade",
       x = "Admission Grade",
       y = "Previous Qualification Grade") +
  theme_minimal()

# We observe a positive correlation between admission grades and previous qualification grades.
# Graduates tend to cluster in the higher grade ranges, which suggests a strong academic performance is linked to a better outcome
# Dropouts appear more scattered, particularly in the lower grade ranges,
# which indicates that students with lower grades are more likely to drop out.

#--------------------------------------------------------------------------------------------------------
# Anova: we are going to check whether there are significant differences between the three groups of the
# variable "Target" in terms of the variable "Curricular units 1st sem (grade)"
# Ensure the 'Target' variable is a factor
train$Target <- as.factor(train$Target)

# Quick check for missing values
print(paste0("There are ", nrow(train[is.na(Target),]), " missing values in variable Target"))
print(paste0("There are ", nrow(train[is.na(`Curricular units 1st sem (grade)`),]), " missing values in variable Curricular units 1st sem (grade)"))

# Run the anova
anova_result <- aov(`Curricular units 1st sem (grade)` ~ Target, data = train)

# Summary of ANOVA
summary(anova_result)

# F-statistic: The F-value is extremely large (34,578.97), which suggests that the variance
# between the groups ("Graduate", "Dropout", and "Enrolled") is much larger than the variance within the groups.

# p-value: The p-value is effectively 0. Since the p-value is much smaller than the common significance level of 0.05,
# we can confidently reject the null hypothesis.

# Post-hoc analysis: we want to see which groups differ
# Run Tukey HSD test
tukey_result <- TukeyHSD(anova_result)

# Display the results
print(tukey_result)

# All pair of groups seem to show a significant difference between each other

#--------------------------------------------------------------------------------------------------------
# Modelling: now we will try to create a predictive model that we can use on our test dataset


# 1. Determine important features
train$Target <- as.factor(train$Target)
# Handle missing values
train <- na.omit(train)
# Remove the 'id' column
train <- train %>% select(-id)
# Clean column names to make them syntactically valid
colnames(train) <- make.names(colnames(train))

set.seed(123)
# Train a Random Forest model
rf_model <- randomForest(Target ~ ., data = train, importance = TRUE)
# Extract feature importance
importance(rf_model)
# Plot feature importance
importance_df <- data.frame(Feature = row.names(importance(rf_model)), 
                            Importance = importance(rf_model)[, "MeanDecreaseGini"])
# Order by importance
importance_df <- importance_df %>% arrange(desc(Importance))

# Plot
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Importance")

# It seems that the most important variables that explain the Target are the variables
# related to the curricular units taken in the previous semesters and their grades


# 2. Run an XGBoost model
# Convert data to matrix format
target <- as.numeric(train$Target) - 1  # Convert factors to numeric starting from 0
data_matrix <- model.matrix(Target ~ . - 1, data = train)  # Create a matrix and remove the intercept

# Split the data into training and test sets
train_index <- createDataPartition(target, p = 0.8, list = FALSE)
train_data <- data_matrix[train_index,]
train_target <- target[train_index]
test_data <- data_matrix[-train_index,]
test_target <- target[-train_index]

# Train the XGBoost model
xgb_train <- xgb.DMatrix(data = train_data, label = train_target)
xgb_test <- xgb.DMatrix(data = test_data, label = test_target)

params <- list(
  booster = "gbtree",
  objective = "multi:softprob",  # Multiclass classification
  eval_metric = "mlogloss",
  num_class = length(unique(target))  # Number of classes
)

xgb_model <- xgb.train(
  params = params,
  data = xgb_train,
  nrounds = 100,
  watchlist = list(train = xgb_train, eval = xgb_test),
  print_every_n = 10,
  early_stopping_rounds = 10
)

# Make predictions
preds <- predict(xgb_model, xgb_test)
pred_labels <- matrix(preds, ncol = length(unique(target)), byrow = TRUE)
pred_labels <- max.col(pred_labels) - 1  # Convert to class labels

# Confusion matrix
confusion_matrix <- table(Predicted = pred_labels, Actual = test_target)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(pred_labels == test_target) / length(test_target)
print(paste("Accuracy: ", round(accuracy * 100, 2), "%", sep = ""))
# "Accuracy: 83.41%"

# Save the model
xgb.save(xgb_model, paste0(outpath, "xgboost_model.bin"))


# 3. Apply XGBoost model to test dataset
# Handle missing values
test <- na.omit(test)
# Remove the 'id' column if it exists
test <- test %>% select(-id)
# Clean column names to make them syntactically valid
colnames(test) <- make.names(colnames(test))

# Convert test data to matrix format
test_matrix <- model.matrix(~ . - 1, data = test)

# Create DMatrix for the test data
xgb_test_data <- xgb.DMatrix(data = test_matrix)

# Make predictions
preds <- predict(xgb_model, xgb_test_data)
pred_labels <- matrix(preds, ncol = length(unique(target)), byrow = TRUE)
pred_labels <- max.col(pred_labels) - 1  # Convert to class labels

# Get the 'id' variable back
test_data_with_id <- fread(paste0(inpath,"test.csv"))

# Create a lookup table for mapping numeric predictions to categories
lookup_table <- c("Dropout", "Enrolled", "Graduate")

# Map the numeric predictions back to original target categories
predicted_categories <- lookup_table[pred_labels + 1]

# Add the predictions to the test data
test_data_with_id$Target <- predicted_categories
test_data_with_id <- test_data_with_id[, .(id, Target)]

# Save the predictions to a CSV file
write.csv(test_data_with_id, paste0(outpath, "submission.csv"), row.names = FALSE)


