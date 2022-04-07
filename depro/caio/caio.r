### Pra rodar este programa, você deve primeiramente instalar as bibliotecas
### a seguir, exemplo pra readODS: install.packages("readODS")

### Importando as bibliotecas
library(readODS)
library(corrplot)
#library(igraph)

### Importando os dados para o data.frame
df = read_ods("tab.ods", sheet=1) ### Tabela 1

### Salvando os dados em vetores
N = length(df[1,])  ### Qtdade de dados
br = as.numeric(as.vector(df[1,2:N]))
mg = as.numeric(as.vector(df[23,2:N]))
es = as.numeric(as.vector(df[24,2:N]))
rj = as.numeric(as.vector(df[25,2:N]))
sp = as.numeric(as.vector(df[26,2:N]))
pr = as.numeric(as.vector(df[27,2:N]))
sc = as.numeric(as.vector(df[28,2:N]))
rs = as.numeric(as.vector(df[29,2:N]))
ba = as.numeric(as.vector(df[22,2:N]))
pe = as.numeric(as.vector(df[19,2:N]))
ce = as.numeric(as.vector(df[16,2:N]))
go = as.numeric(as.vector(df[32,2:N]))
am = as.numeric(as.vector(df[9,2:N]))

### Construido as respectivas séries temporais
sbr = ts(br, start=c(2017,1), frequency=12)
smg = ts(mg, start=c(2017,1), frequency=12)
ses = ts(es, start=c(2017,1), frequency=12)
srj = ts(rj, start=c(2017,1), frequency=12)
ssp = ts(sp, start=c(2017,1), frequency=12)
spr = ts(pr, start=c(2017,1), frequency=12)
ssc = ts(sc, start=c(2017,1), frequency=12)
srs = ts(rs, start=c(2017,1), frequency=12)
sba = ts(ba, start=c(2017,1), frequency=12)
spe = ts(pe, start=c(2017,1), frequency=12)
sce = ts(ce, start=c(2017,1), frequency=12)
sgo = ts(go, start=c(2017,1), frequency=12)
sam = ts(am, start=c(2017,1), frequency=12)

### Novo data.frame
DF = data.frame(br, mg, es, rj, sp, pr, sc, rs, ba, pe, ce, go, am)
### Matriz de correlação
MC = cor(DF, method="pearson")

### Plotando a matriz de correlação
png("corr_matrix.png")
corrplot(MC, method="number")
dev.off()
