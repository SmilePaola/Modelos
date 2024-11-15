
library(randomForest)
library(e1071)     
library(xgboost)    
library(caret)      
library(pROC)       
library(rpart)      

data <- read.csv("clientes.csv")

head(data)

data$purchased <- as.factor(data$purchased)

data$gender <- as.numeric(as.factor(data$gender))  

set.seed(42)
trainIndex <- createDataPartition(data$purchased, p = 0.7, list = FALSE, times = 1)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

train_matrix <- xgb.DMatrix(data = as.matrix(trainData[, -which(names(trainData) == "purchased")]), 
                            label = as.numeric(trainData$purchased) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(testData[, -which(names(testData) == "purchased")]))


tree_model <- rpart(purchased ~ ., data = trainData, method = "class")

rf_model <- randomForest(purchased ~ ., data = trainData)

log_model <- glm(purchased ~ ., data = trainData, family = binomial)


svm_model <- svm(purchased ~ ., data = trainData, probability = TRUE)

xgb_model <- xgboost(data = train_matrix, max.depth = 3, eta = 0.1, nrounds = 100, 
                     objective = "binary:logistic", eval_metric = "logloss", verbose = 0)

calculate_metrics <- function(pred, true_labels) {
  cm <- confusionMatrix(as.factor(pred), true_labels)
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensibilidad"]
  specificity <- cm$byClass["Especifidad"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  list(precision = precision, recall = recall, specificity = specificity, f1 = f1)
}


tree_pred <- predict(tree_model, testData, type = "class")
tree_metrics <- calculate_metrics(tree_pred, testData$purchased)


rf_pred <- predict(rf_model, testData)
rf_metrics <- calculate_metrics(rf_pred, testData$purchased)


log_pred <- predict(log_model, testData, type = "response")
log_class <- ifelse(log_pred > 0.5, "1", "0")
log_metrics <- calculate_metrics(log_class, testData$purchased)


svm_pred <- predict(svm_model, testData, probability = TRUE)
svm_class <- ifelse(svm_pred == "1", "1", "0")
svm_metrics <- calculate_metrics(svm_class, testData$purchased)


xgb_pred <- predict(xgb_model, test_matrix)
xgb_class <- ifelse(xgb_pred > 0.5, "1", "0")
xgb_metrics <- calculate_metrics(xgb_class, testData$purchased)


par(mfrow = c(1, 1)) 
roc_tree <- roc(as.numeric(testData$purchased), as.numeric(tree_pred))
roc_rf <- roc(as.numeric(testData$purchased), as.numeric(rf_pred))
roc_log <- roc(as.numeric(testData$purchased), as.numeric(as.factor(log_class)))
roc_svm <- roc(as.numeric(testData$purchased), as.numeric(as.factor(svm_class)))
roc_xgb <- roc(as.numeric(testData$purchased), xgb_pred)

plot(roc_tree, col = "blue", main = "Curvas ROC de Modelos")
lines(roc_rf, col = "green")
lines(roc_log, col = "red")
lines(roc_svm, col = "purple")
lines(roc_xgb, col = "orange")
legend("bottomright", legend = c("Árbol de Decisión", "Random Forest", 
                                 "Regresión Logística", "SVM", "XGBoost"), 
       col = c("blue", "green", "red", "purple", "orange"), lwd = 2)


auc_tree <- auc(roc_tree)
auc_rf <- auc(roc_rf)
auc_log <- auc(roc_log)
auc_svm <- auc(roc_svm)
auc_xgb <- auc(roc_xgb)


resultados <- data.frame(
  Modelo = c("Árbol de Decisión", "Random Forest", "Regresión Logística", "SVM", "XGBoost"),
  Precisión = round(c(tree_metrics$precision, rf_metrics$precision, log_metrics$precision, svm_metrics$precision, xgb_metrics$precision) * 100, 1),
  Sensibilidad = round(c(tree_metrics$recall, rf_metrics$recall, log_metrics$recall, svm_metrics$recall, xgb_metrics$recall) * 100, 1),
  Especificidad = round(c(tree_metrics$specificity, rf_metrics$specificity, log_metrics$specificity, svm_metrics$specificity, xgb_metrics$specificity) * 100, 1),
  F1_Score = round(c(tree_metrics$f1, rf_metrics$f1, log_metrics$f1, svm_metrics$f1, xgb_metrics$f1) * 100, 1),
  AUC = round(c(auc_tree, auc_rf, auc_log, auc_svm, auc_xgb), 3)
)

for (i in 1:nrow(resultados)) {
  cat("Comparación de Resultados\n")
  cat(resultados$Modelo[i], ":\n")
  cat("• Precisión:", resultados$Precisión[i], "%\n")
  cat("• Sensibilidad:", resultados$Sensibilidad[i], "%\n")
  cat("• Especificidad:", resultados$Especificidad[i], "%\n")
  cat("• F1 Score:", resultados$F1_Score[i], "%\n")
  cat("• AUC:", resultados$AUC[i], "\n\n")
}