### Programa para tratar o problema do cartPole com Q-Learning
### 
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Importando os módeulos e métodos necessários
import gym                   ### Ambiente de simulação
import numpy as np           ### Cálculos numéricos
from time import sleep       ### Pausa na execução

### Ambiente de simulação
carpo = gym.make("CartPole-v0")

### Seguindo a lógica do programa: prog02
q_tab = np.zeros([carpo.observation_space.shape[0], carpo.action_space.n])
print("Q Table: \n", q_tab)

### Como podemos observar, neste caso os 4 estados do sistema não 
### possibilitam a construção de uma q-table, devido seu carater
### infinitesimal 

### No entanto, de acordo com [1], temos que o vetor que representa
### cada estado do sistema contém as seguintes informações:
### [ Posição [-5, 5], 
###   Velocidade, 
###   Ângulo da aste (pole) [-24°, 24°], 
###   Velocidade angular ]

### Como pretendemos manter a aste ereta, de acordo com [1],
### sempre que o ângulo for negativo, devemos mecher pra 
### esquerda (ação-reação de Newton), pra ângulos positivos
### mecheremos pra direita. [1]: 0 <- esquerda e 1 -> direita   

### Função que realiza a tarefa descrita acima
def melhor_acao(angulo):
  if angulo < 0:
    return 0
  else:
    return 1

### Começando a brincadeira
state = carpo.reset()
action = melhor_acao(state[2])
carpo.render()

### Continuando a brincadeira
erro = 0   ### Avaliação de erros
N = 100    ### Número total de episódios
for i in range(N):
  ### Novo estado
  state, reward, done, info = carpo.step(action)   
  ### Nova ação
  action = melhor_acao(state[2])
  ### Mostrando o agente
  carpo.render()
  sleep(0.05)
  ### Avaliando se deu certo:
  if done:
    erro += 1

### Resultado
print("Precisão do método: ", 100*(1-(erro/N)), "%")

### FIM

### Referência:
# [1] https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
