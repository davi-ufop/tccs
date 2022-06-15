### Programa para elucidar a técnica de aprendizado por reforço usando
### a técnica denominada por 'escalada' (hill-climbing).
### 
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Módulos e métodos necessários
import gym
import pylab as pl
import torch as tk

### Parâmetros essenciais pra simulação
num_epi = 1000       ### Número de episódios de treinamento
num_test = 100       ### Número de episódios pra testes
best_tt_rew = 0      ### Melhor recompensa
best_weight = None   ### Melhor matriz de pesos
list_tt_rew = []     ### Lista de recompensas  
noise_search = 0.01  ### Ruído pra busca - evita ótimos locais

### Ambiente de simulação
carpo = gym.make("CartPole-v0")
carpo.seed(123)   ### Fixando a semente randômica

### Dimensões da matriz de pesos
n_state = carpo.observation_space.shape[0]
n_action = carpo.action_space.n

### Processo de busca pela melhor matriz de pesos

