### Programa para elucidar a técnica de aprendizado por reforço usando
### a técnica denominada por 'escalada' (hill-climbing).
### 
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Módulos e métodos necessários
import gym
import pylab as pl
import torch as tk
from tqdm import tqdm

### Parâmetros essenciais pra simulação
num_epi = 1000          ### Número de episódios de treinamento
num_test = 100          ### Número de episódios pra testes
best_tt_rew = 0         ### Melhor recompensa
list_tt_rew = []        ### Lista de recompensas  
noise_search = 0.1      ### Ruído pra busca - taxa de aprendizagem - VARIE!!!

### Função pra simular os episódios
def run_epi(env, weight):  
  state = env.reset()      ### Estado inicial do episódio
  total_rew = 0            ### Recompensa total
  done = False             ### Quando True, termina o episódio
  ### Varredura até que o módulo do ângulo exceda 22º
  while not done:
    state = tk.from_numpy(state).float()                 ### Estado a cada passo -> vetor [1x4]
    action = tk.argmax(tk.matmul(state, weight))         ### Ação a cada passo -> max de [1x2]
    state, reward, done, info = env.step(action.item())  ### Novo passo, de acordo com item acima
    total_rew += reward                                  ### Contagem de recompensas 
  return total_rew         ### Retorna a recompensa total

### Ambiente de simulação
carpo = gym.make("CartPole-v0")
#carpo.seed(123)   ### Fixando a semente randômica

### Dimensões da matriz de pesos
n_state = carpo.observation_space.shape[0]
n_action = carpo.action_space.n

### Processo de busca pela melhor matriz de pesos
best_weight = tk.rand(n_state, n_action)            ### Modelo para a melhor matriz W
for epi in tqdm(range(num_epi)):                    ### Varredura pra buscas
  noise_w = noise_search*tk.rand(n_state, n_action) ### Ruido pra W
  mat_w = best_weight + noise_w                     ### Matriz W
  tt_rew = run_epi(carpo, mat_w)                    ### Recompensa total do episódio
  if tt_rew >= best_tt_rew:                         ### Condição da busca - por que >=? - Pra evitar máximos locais
    best_tt_rew = tt_rew                            ### Atualização da melhor recompensa
    best_weight = mat_w                             ### Atualização da mehlor matriz W
  list_tt_rew.append(tt_rew)                        ### Lista de recompensas

### Plotando o resultado da varredura
pl.plot(list_tt_rew, color='red')
pl.title("Rewards in each episode")
pl.xlabel("episodes")
pl.ylabel("rewards")
pl.show()

### Resultado para a matriz W
print("Melhor matriz de pesos:\n", best_weight)

### Teste da melhor matriz peso
ks = 0                                     ### Contagem de sucessos
max_rew = max(list_tt_rew)                 ### Maior recompensa 
for test in range(num_test):               ### 100 testes
  rew_test = run_epi(carpo, best_weight)   ### Recompensa pra cada teste
  if rew_test > (0.95*max_rew):            ### Acima de 95% de sucessos
    ks += 1                                ### ... a gente conta

### Precisão:
print("Precisão deste método = ", 100*(ks/num_test), "%")

### FIM
