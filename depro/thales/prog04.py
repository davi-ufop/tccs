### Programa para realizar o treinamento de um agente por reforço
### usando um modelo randômico, prévio ao de redes neurais profundas,
### usando apenas a biblioteca de ambientes GYM 
### 
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Bibliotecas necessárias
import gym

### Parâmetros do processo
reward = 0

### Criando o ambiente de simulação - Carro com Pêndulo Invertido
### O objetivo é movimentar o carro pra manter o pêndulo ereto, 
### vencendo a gravidade -> 0 move pra esquerda e 1 pra direita
carpo = gym.make("CartPole-v0") 
semente = input("Digite a semente randômica: ")
carpo.seed(int(semente))   ### Semente randômica pra estabelecer 
state = carpo.reset()      ### a mesma condição inicial

### Analisando o ambiente de simulação -> help(carpo)
print("\n\t\tANÁLISE DO AMBIENTE DE SIMULAÇÃO: ")
n_state = carpo.observation_space.shape[0]
print("Número de estados acessíveis: ", n_state)
n_action = carpo.action_space.n
print("Número de ações possíveis: ", n_action)
r_rewards = carpo.reward_range
print("Escala das recompensas: ", r_rewards)

### Primeira ação e primeiro estado:
state0 = state                              ### 1º estado                        
print("\nPrimeiro estado: ", state0)
action0 = carpo.action_space.sample()       ### 1ª ação
print("Primeira ação: ", action0)
state1, reward1, done1, info1 = carpo.step(action0)  ### 2º estado
print("Segundo estado: ", state0)

### Até a gravidade vencer: done = True
done = done1      ### Parâmetro de controle
tot_rews = 0      ### Parâmetro de contagem 
while not done:
  carpo.render()  ### Renderização do ambiente
  ### Selecionando a ação aleatoriamente
  action = carpo.action_space.sample()
  ### Próximo passo do sistema
  state, reward, done, info = carpo.step(action)  
  ### Somando as recompensas (1 pra cada passo)
  tot_rews += reward

### Resultado do processo -> objetivo = 1.0, vide listas acima.
print("\nRecompensa total (número de movimentos): ", tot_rews)
### Fechando o ambiente de simulação
carpo.close()

### FIM 
