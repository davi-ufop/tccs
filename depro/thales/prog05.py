### Programa para realizar o treinamento de um agente por reforço
### usando um modelo intuitivo, prévio ao de redes neurais profundas,
### usando apenas a biblioteca de ambientes GYM e outras pertinentes
### 
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Bibliotecas necessárias
import gym
from tqdm import tqdm     ### Barra de progresso
from time import sleep    ### Tempo pra observações

### Parâmetros dop processo
reward = 0

### Criando o ambiente de simulação - Carro com Pêndulo Invertido
### O objetivo é movimentar o carro pra manter o pêndulo ereto, 
### vencendo a gravidade -> 0 move pra esquerda e 1 pra direita
carpo = gym.make("CartPole-v0") 
carpo.seed(111)  ### Semente randômica pra estabelecer 
carpo.reset()    ### a mesma condição inicial

### Lista da sequência de ações pra tentar equilibrar o pêndulo
#lista = [1, 1, 1, 1, 1, 1, 1]  ### Recompensa = 0.26
#lista = [0, 0, 0, 0, 0, 0, 0]  ### Recompensa = 0.23 
#lista = [1, 0, 1, 0, 1, 0, 1]  ### Recompensa = 0.66 - 2º lugar
#lista = [0, 1, 0, 1, 0, 1, 0]  ### Recompensa = 0.51
#lista = [0, 0, 0, 1, 1, 1, 0]  ### Recompensa = 0.40 
#lista = [1, 1, 1, 0, 0, 0, 1]  ### Recompensa = 0.49 
#lista = [1, 0, 0, 1, 0, 1, 0]  ### Recompensa = 0.57 - 3º lugar
lista = [1, 0, 0, 1, 0, 1, 1]   ### Recompensa = 0.80 - Melhor!
#lista = [1, 0, 0, 1, 0, 1, 1, 0]  ### Recompensa = 0.93
acoes = lista + lista + lista + lista + lista
total = len(acoes)  ### 35 ações

### Aplicando as ações constituidas pela lista acima
for acao in tqdm(acoes):
  carpo.render()  ### Renderização do ambiente
  ### Próximo passo do sistema
  state, obs, done, info = carpo.step(acao)  
  sleep(0.25)     ### Tempo pra gente observar
  ### Condição de fracasso -> True = Caiu!
  if done:
    reward += 1;

### Resultado do processo -> objetivo = 1.0, vide listas acima.
print("Recompensa da lista selecionada: %.2f" %(1 - (reward/total)))
### Fechando o ambiente de simulação
carpo.close()

### FIM
