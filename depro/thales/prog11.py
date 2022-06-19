### Programa para resolver o problemma do CartPole com uma rede neural
### usando os métodos elaborrados na biblioteca stable-baselines3
###
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Biblioteca GYM e métodos de aprendizado por reforço
import gym
from stable_baselines3 import DQN, A2C  ### Vide [1]

### Parâmetros da simulação
NE = 5000
NT = 100

### Criando o ambiente de simulação
carpo = gym.make('CartPole-v0')

### Modelo baseado no método A2C
carpo.seed(123)
carpo.reset()
model_a2c = A2C('MlpPolicy', carpo, verbose=1)
model_a2c.learn(total_timesteps=NE)       ### Treino com N episódios

### Modelo baseado no método DQN
carpo.seed(123)
carpo.reset()
model_dqn = DQN("MlpPolicy", carpo, verbose=1)
model_dqn.learn(total_timesteps=NE)       ### Treino com N episódios

### Realizando os testes de A2C
carpo.seed(123)
state = carpo.reset()
ks = 0
for i in range(NT):
    action, obs = model_a2c.predict(state, deterministic=True) ### Previsão
    state, reward, done, info = carpo.step(action)             ### Próximo passo
    carpo.render()     ### Mostra o ambiente
    ks += reward;      ### Conta os sucessos 
    if done:           ### Condição de parada - CAIU!
      break
print("Método A2C teve ", (100*(ks/NT)), "% de sucessos!")

### Realizando os testes de DQN
carpo.seed(123)
state = carpo.reset()
ks = 0
for i in range(NT):
    action, obs = model_dqn.predict(state, deterministic=True) ### Previsão
    state, reward, done, info = carpo.step(action)             ### Próximo passo
    carpo.render()     ### Mostra o ambiente
    ks += reward;      ### Conta os sucessos 
    if done:           ### Condição de parada - CAIU!
      break
print("Método DQN teve ", (100*(ks/NT)), "% de sucessos!")

### FIM

### Referências
# [1] https://www.hoshd.com/project/cartpole/

################################################################################
### DICA ÚTIL ->  Pra salvar e carrega modelos
"""
model.save("dqn_cartpole")
model = DQN.load("dqn_cartpole")
"""
################################################################################
