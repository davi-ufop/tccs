### Programa para realizar o treinamento de um agente por reforço
### usando um modelo com redes neurais profundas: DQN, implementado
### usando a biblioteca pytorch
### 
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Bibliotecas
import gym
import torch as tk
import pylab as pl
from time import sleep


### Criando o ambiente de simulação - Pêndulo Invertido
carpo = gym.make("Taxi-v3")


### FIM
