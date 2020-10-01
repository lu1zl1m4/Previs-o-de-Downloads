# Projeto - Prever se um usuário fará o download de um app depois de clicar no anúncio para
# dispositvo móvel.

# Data set - https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

Sys.getenv("R_MAX_VSIZE")

# Carregando as bibliotecas
library(data.table)
library(ggplot2)
library(tidyr)
library(randomForest)
library(ROSE)
library(dplyr)
library(readr)
library(mlbench)
library(caret)

# ip - endereço IP do clique
# app - identificador para o aplicativo
# device - um identificador para o tipo de dispositivo
# os - um identificador para o sistema operacional
# channel - um identificador para o canal publicador do anúncio
# click_time - data e hora do clique
# attributed_time - se o usuário fez o download do app, a hora e data do download
# is_attirbuted - se foi feito o download, 1, caso contrário, 0.

# Carregando dados que serão usados para criar o código devido ao tamanho 
# do arquivo com os dados completos.
TrainData <- fread("train.csv", sep = ",", 
                   colClasses = list(factor = c("is_attributed")))
head(TrainData)

# Vamos pegar uma amostra
set.seed(1)
linhas <- sample(1:nrow(TrainData), size = 200000, replace = FALSE)
Data <- TrainData[linhas,]
head(Data)

# Etapa 1 - Explorando dados (data munging)

# Verificando se existem valores NA
sapply(Data, function(x) sum(is.na(x)))

# Verificando número de caracteres espaços
sapply(Data, function(x) sum(ifelse(x == "", TRUE, FALSE)))

# Contando a quantidade de números 0 e 1 em is_attributed
table(Data$is_attributed)

# Verificando os tipos dos dados
str(Data)

# Feature Selection
# A coluna attributed_time é totalmente dependente se foi feito o download, então
# a decisão é não usá-la para treinar o modelo até porque não deve ter nos dados de 
# teste
ind <- grep("attributed_time", names(Data))
Data <- Data[,-c(7)]

# Plotando o gráfico do is_attributed
ggplot(data = Data) +
 geom_bar(mapping = aes(x = is_attributed, fill = is_attributed))

# Etapa 2 - Engenharia de dados

# Remodelando o data frame
Downloads <- separate(data = Data, col = click_time, 
                   into = c("data", "hora"), sep = " ")
head(Downloads)
sapply(Downloads, function(x) sum(is.na(x)))

Downloads$data <- as.factor(Downloads$data)

unique(Downloads$data)

Downloads <- separate(data = Downloads, col = hora, into = c("hora","minuto","segundo"), 
                   sep = ":")
head(Downloads)

Downloads$hora <- as.integer(Downloads$hora)
Downloads$minuto <- as.integer(Downloads$minuto)
Downloads$segundo <- as.integer(Downloads$segundo)

head(Downloads)

# Etapa 3 - Criando primeiro modelo

set.seed(1)
linhasTreino <- sample(1:nrow(Downloads), 0.7*nrow(Downloads), replace = FALSE)
treinoData <- Downloads[linhasTreino,]
testeData <- Downloads[-linhasTreino,]
modelo <- randomForest(is_attributed ~ ., data = treinoData, 
                       mtry = 3, ntree = 500)
pred <- predict(modelo, testeData)

tabelaClass <- table(testeData$is_attributed, pred)
tabelaClass

# Avalia o quão frequentemente o classificador executa a predição correta
# Relação entre o número de acertos e o total de predições
acuracia <- (tabelaClass[1,1] + tabelaClass[2,2])/
  (tabelaClass[1,1] + tabelaClass[2,2] + tabelaClass[1,2] + tabelaClass[2,1])
acuracia

# Avalia, dentro do que o modelo aponta como positivo, o que de fato é positivo
precisao <- tabelaClass[2,2]/(tabelaClass[2,2] + tabelaClass[1,2])
precisao

# Avalia, dentro do que de fato é positivo, o que o modelo classifica como positivo
revocacao <- tabelaClass[2,2]/(tabelaClass[2,2] + tabelaClass[2,1])
revocacao

# F1-Score - Nos dá a qualidade geral do modelo
F1 <- 2*precisao*revocacao/(precisao + revocacao)
F1

# Fazendo o balanceamento dos dados
DataBalanc <- ROSE(formula = is_attributed ~ ., data = Downloads, seed = 1)$data
head(DataBalanc)
table(DataBalanc$is_attributed)

TrainData2 <- DataBalanc[linhasTreino,]
TestData2 <- DataBalanc[-linhasTreino,]
modelo2 <- randomForest(is_attributed ~ ., data = TrainData2, 
                        mtry = 3, ntree = 1000)
pred2 <- predict(modelo2, TestData2)

tabelaClass2 <- table(TestData2$is_attributed, pred2)
tabelaClass2

# Avalia o quão frequentemente o classificador executa a predição correta
# Relação entre o número de acertos e o total de predições
acuracia2 <- (tabelaClass2[1,1] + tabelaClass2[2,2])/
  (tabelaClass2[1,1] + tabelaClass2[2,2] + tabelaClass2[1,2] + tabelaClass2[2,1])
acuracia2

# Avalia, dentro do que o modelo aponta como positivo, o que de fato é positivo
precisao2 <- tabelaClass2[2,2]/(tabelaClass2[2,2] + tabelaClass2[1,2])
precisao2

# Avalia, dentro do que de fato é positivo, o que o modelo classifica como positivo
revocacao2 <- tabelaClass2[2,2]/(tabelaClass2[2,2] + tabelaClass2[2,1])
revocacao2

# F1-Score - Nos dá a qualidade geral do modelo
F12 <- 2*precisao2*revocacao2/(precisao2 + revocacao2)
F12

# As datas vão ser diferentes, então ficaria difícil ter a data como factor, vamos
# dia, mês e ano e transformar em dias
Downloads$data <- as.factor(Downloads$data)
DownloadsNew <- separate(data = Downloads, col = data, into = c("ano","mês","dia"), 
                         sep = "-")

DownloadsNew$ano <- as.integer(DownloadsNew$ano)
DownloadsNew$mês <- as.integer(DownloadsNew$mês)
DownloadsNew$dia <- as.integer(DownloadsNew$dia)

# Novo modelo com mudança de dados
TrainData3 <- DownloadsNew[linhasTreino,]
TestData3 <- DownloadsNew[-linhasTreino,]
modelo3 <- randomForest(is_attributed ~ ., data = TrainData3)
pred3 <- predict(modelo3, TestData3)

tabelaClass3 <- table(TestData3$is_attributed, pred3)
tabelaClass3

# Avalia o quão frequentemente o classificador executa a predição correta
# Relação entre o número de acertos e o total de predições
acuracia3 <- (tabelaClass3[1,1] + tabelaClass3[2,2])/
  (tabelaClass3[1,1] + tabelaClass3[2,2] + tabelaClass3[1,2] + tabelaClass3[2,1])
acuracia3

# Avalia, dentro do que o modelo aponta como positivo, o que de fato é positivo
precisao3 <- tabelaClass3[2,2]/(tabelaClass3[2,2] + tabelaClass3[1,2])
precisao3

# Avalia, dentro do que de fato é positivo, o que o modelo classifica como positivo
revocacao3 <- tabelaClass3[2,2]/(tabelaClass3[2,2] + tabelaClass3[2,1])
revocacao3

# F1-Score - Nos dá a qualidade geral do modelo
F3 <- 2*precisao3*revocacao3/(precisao3 + revocacao3)
F3

# Fazendo o balanceamento dos dados
DataBalanc2 <- ROSE(formula = is_attributed ~ ., 
                    data = DownloadsNew, seed = 1)$data

table(DataBalanc2$is_attributed)

TrainData4 <- DataBalanc2[linhasTreino,]
TestData4 <- DataBalanc2[-linhasTreino,]
modelo4 <- randomForest(is_attributed ~ ., data = TrainData4, 
                        mtry = 3, ntree = 2000)
pred4 <- predict(modelo4, TestData4)

tabelaClass4 <- table(TestData4$is_attributed, pred4)
tabelaClass4

# Avalia o quão frequentemente o classificador executa a predição correta
# Relação entre o número de acertos e o total de predições
acuracia4 <- (tabelaClass4[1,1] + tabelaClass4[2,2])/
  (tabelaClass4[1,1] + tabelaClass4[2,2] + tabelaClass4[1,2] + tabelaClass4[2,1])
acuracia4

# Avalia, dentro do que o modelo aponta como positivo, o que de fato é positivo
precisao4 <- tabelaClass4[2,2]/(tabelaClass4[2,2] + tabelaClass4[1,2])
precisao4

# Avalia, dentro do que de fato é positivo, o que o modelo classifica como positivo
revocacao4 <- tabelaClass4[2,2]/(tabelaClass4[2,2] + tabelaClass4[2,1])
revocacao4

# F1-Score - Nos dá a qualidade geral do modelo
F4 <- 2*precisao4*revocacao4/(precisao4 + revocacao4)
F4

# Tentando melhorar parâmetros do algoritmo
x <- DataBalanc2[,-12]
y <- DataBalanc2[,12]

# mtry = 3
bestmtry <- tuneRF(x, y, stepFactor = 1.5, 
                   improve = 1e-5, ntree = 2000)

