# Package loading
library(car)
library(DescTools)
library(corrplot)

getwd()

# Set Working Directory
setwd("C:/Users/aaa/Downloads")
getwd() 

# Read the data file
TrainRaw = read.csv("PropertyPrice_Train.csv", stringsAsFactors = TRUE)
PredictionRaw = read.csv("PropertyPrice_Prediction.csv", stringsAsFactors = TRUE)

View(TrainRaw)
View(PredictionRaw)

# Check total columns
ncol(TrainRaw)
ncol(PredictionRaw)


# Create Source Column in both Train and Test
TrainRaw$Source = "Train"
PredictionRaw$Source = "Test"

# Combine Train and Test
FullRaw = rbind(TrainRaw, PredictionRaw)

# View the data
View(FullRaw) # "Sale_Price" is our "dependent variable"

# Lets drop "Id" column
FullRaw = subset(FullRaw, select = -Id)

# Validate the deletion
dim(FullRaw) # Should be 1 column less
colnames(FullRaw) # Should not have the "Id" column

# Check the summary of the file
summary(FullRaw)

# Check for NAs
colSums(is.na(FullRaw))


############################
# Missing value imputation (manually)
############################

# Variables having missing value
# Garage_Built_Year, Garage

# Garage_Built_Year
# Step 1: Find median
median(TrainRaw$Garage_Built_Year)
tempMedian = median(TrainRaw$Garage_Built_Year, na.rm = TRUE)
tempMedian

# Step 2: Find missing value rows
missingValueRows = is.na(FullRaw[, "Garage_Built_Year"]) 
sum(missingValueRows)

# Step 3: Impute (or Fill) missing values in data
FullRaw[missingValueRows, "Garage_Built_Year"] = tempMedian
colSums(is.na(FullRaw))
summary(FullRaw)

# Garage
# Step 1: Find Mode
Mode(TrainRaw$Garage, na.rm = TRUE)[1]
tempMode = Mode(TrainRaw$Garage, na.rm = TRUE)[1]
tempMode

# Step 2: Find missing value rows
missingValueRows = is.na(FullRaw[, "Garage"]) 
sum(missingValueRows)

# Step 3: Impute (or Fill) missing values in data
FullRaw[missingValueRows, "Garage"] = tempMode


# Check for NAs
colSums(is.na(FullRaw)) # Should have NO NAs except for Sale_Price

# summary(FullRaw)
# colSums(is.na(FullRaw))
# sum(is.na(FullRaw))
# dim(FullRaw)

############################
# Correlation check
############################

library(corrplot)

continuous_variable_check = function(x)
{
  return(is.numeric(x) | is.integer(x))
}

continuousVars = sapply(TrainRaw, continuous_variable_check)
continuousVars
corrDf = cor(TrainRaw[TrainRaw$Source == "Train", continuousVars])
View(corrDf)

windows()
corrplot(corrDf)


############################
# Dummy variable creation
############################

factorVars = sapply(FullRaw, is.factor)
factorVars
dummyDf = model.matrix(~ ., data = FullRaw[,factorVars])
View(dummyDf)

dim(dummyDf)
FullRaw2 = cbind(FullRaw[,!factorVars], dummyDf[,-1])

# Check the dimensions of FullRaw2
dim(FullRaw2)

# Check if all variables are now numeric/integer
str(FullRaw2) 


############################
# Sampling
############################

# Step 1: Divide Train into Train and Test
Train = subset(FullRaw2, subset = FullRaw2$Source == "Train", select = -Source)
PredictionDf = subset(FullRaw2, subset = FullRaw2$Source == "Test", select = -Source)


# Step 2: Divide Train further into Train and Test by random sampling
#set.seed(123) # This is used to reproduce the SAME composition of the sample EVERYTIME
set.seed(100)
RowNumbers = sample(x = 1:nrow(Train), size = 0.80*nrow(Train))
head(RowNumbers)
Test = Train[-RowNumbers, ] # Testset
Train = Train[RowNumbers, ] # Trainset

dim(Train)
dim(Test)


############################
# Multicollinearity check
############################

# Remove variables with VIF > 5

M1 = lm(Sale_Price ~ ., data = Train)
# M1 = lm(Dependent ~ x1 + x2 + x3 + x4, data = Train)

library(car)
sort(vif(M1), decreasing = TRUE)[1:3]



# Remove GarageAttchd
M2 = lm(Sale_Price ~ . - GarageAttchd, data = Train)
sort(vif(M2), decreasing = TRUE)[1:3]


# Remove Kitchen_QualityTA
M3 = lm(Sale_Price ~ . - GarageAttchd - Kitchen_QualityTA, data = Train)
sort(vif(M3), decreasing = TRUE)[1:3]

# Remove First_Floor_Area
M3 = lm(Sale_Price ~ . - GarageAttchd - Kitchen_QualityTA - First_Floor_Area, data = Train)
sort(vif(M3), decreasing = TRUE)[1:3]

summary(M3)

############################
# Model optimization (by selecting ONLY significant variables through step() function)
############################

# Use step() function to remove insignificant variables from the model iteratively
M4 = step(M3) # Step function works on the concept of reducing AIC. Lower the AIC, better the model

summary(M4) # Lets finalize this model


############################
# Model diagnostics
############################

# Few checks
# Homoskedasticity check
plot(M4$fitted.values, M4$residuals) 

# Normality of errors check
summary(M4$residuals) # To check the range. Will be used in histogram is next step
hist(M4$residuals, breaks = seq(-490000, 340000, 10000)) # Should be somewhat close to normal distribution



############################
# Model Evaluation
############################

M4_Pred = predict(M4, Test)
head(M4_Pred)
head(Test$Sale_Price)


############################
Actual = Test$Sale_Price
Prediction = M4_Pred

# RMSE (Root Mean Square Error)

sqrt(mean((Actual - Prediction)^2)) # 38705

# MAPE (Mean Absolute Percentage Error)
mean(abs((Actual - Prediction)/Actual))*100 # 16%

############################

