### Programa para realizar o aprendizado por reforço, usando
### o método de busca randômica pela política do agente
###
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Bibliotecas necessárias
import gym
import pylab as pl
import torch as tk

### Parâmetros da simulação
num_epi = 1000             ### Número de episódios
num_test = 100             ### Número de testes
best_tt_rew = 0            ### Melhor recompensa dentre episódios
best_weight = None         ### Melhores pesos para o agente
list_rews = []             ### Lista pra salvar as recompensas
list_best = []             ### ... as melhores recomepnsas

### Função pra simular os episódios
def run_epi(env, weight):  ### Matriz weight [4x2]
  state = env.reset()      ### Estado inicial do episódio
  total_rew = 0            ### Recompensa total
  done = False             ### Quando True, termina o episódio
  ### Varredura até que o módulo do ângulo exceda 22º
  while not done:
    state = tk.from_numpy(state).float()                 ### Estado a cada passo -> vetor [1x4]
    action = tk.argmax(tk.matmul(state, weight))         ### Ação a cada passo -> max de [1x2]
    state, reward, done, info = env.step(action.item())  ### Novo passo, de acordo com item acima
    total_rew += reward                                  ### Contagem de recompensas 
    #env.render()                                        ### Renderização do ambiente
  return total_rew         ### Retorna a recompensa total

### Criando o ambiente de simulação
carpo = gym.make("CartPole-v0")
#carpo.seed(123)    ### Fixando a semente randômica

### Dimensões da matriz W (pesos): [estados x ações]
n_states = carpo.observation_space.shape[0]
n_actions = carpo.action_space.n

### Realizando a simulação
for epi in range(num_epi):   ### Pra 1000 episódios
  wg = tk.rand(n_states, n_actions)                ### Matriz W - Randômica   
  tt_rew = run_epi(carpo, wg)                      ### Cálculo da recompensa pra W  
  print("Episódio ", epi, " teve recompensa ", tt_rew)
  if tt_rew > best_tt_rew:                         ### Avaliação se a matriz W é melhor
    best_weight = wg                               ### Salvando a melhor W 
    best_tt_rew = tt_rew                           ### Melhor recompensa
  ### Registro de convergência
  list_rews.append(tt_rew)                ### Lista  de todas as recompensas
  list_best.append(best_tt_rew)           ### Lista das melhores recompensas

### Maior valor de recompensas:
max_rew = max(list_rews)

### Plotando a convergência
pl.plot(list_rews, color='red')
pl.plot(list_best, color='blue')
pl.title('Rewards convergence')
pl.xlabel('episodes')
pl.ylabel('rewards')
pl.show()

### Resultado para a matriz W
print("Melhor matriz de pesos:\n", best_weight)

### Teste da melhor matriz peso
ks = 0   ### Contagem de sucessos
for test in range(num_test):               ### 100 testes
  rew_test = run_epi(carpo, best_weight)   ### Recompensa pra cada teste
  if rew_test > (0.95*max_rew):            ### Acima de 95% de sucessos
    ks += 1                                ### ... a gente conta

### Precisão:
print("Precisão deste método = ", 100*(ks/num_test), "%")

### FIM
