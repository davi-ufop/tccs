### Programa para elucidar o algoritmo gradiente de política
### para otimizar o agente no ambiente CartPole-v0 de GYM
###
### Davi Neves - Ouro Preto, MG, Brasil - 06/2022

### Bibliotecas necessárias
import gym                 ### Ambiente de simulação
import torch as tk         ### Aprendizado de máquina
import pylab as pl         ### Plotar gráficos
from tqdm import tqdm      ### Barra de evolução

### Parâmetros do problema
num_epi = 1000            ### Número de episódios pra treino
num_test = 100            ### Número de episódios pra teste
list_tt_rew = []          ### Lista pra salvar as recompensas
alfa = 0.01               ### Taxa de aprendizagem

### Função pra determinar a recompensa em um episódio -> vide [1]
def run_epi(env, weight):              ### Entradas: ambiente e matriz de pesos [4x2]
  state = env.reset()                  ### Inicia o ambiente de simulação
  grads = []                           ### Lista para os gradientes
  tt_rew = 0                           ### Recompensa total do episódio
  is_done = False                      ### Condição de equilíbrio do pêndulo (True = caiu)
  while not is_done:                   ### Realizando o episódio -> Até cair!
    state = tk.from_numpy(state).float()   ### Estado do agente, convertido para um vetor [1x4]
    z = tk.matmul(state, weight)           ### Estado x Pesos -> Saída do somatório ponderado [1x2]
    prob = tk.nn.Softmax()(z)              ### Função de ativação Softmax() -> Como num neurônio [1x2]
    ### A ação é calculada com a distribuição de probabilidade (Bernoulli) baseada na saída (prob) acima
    ### Por exemplo, se as probabilidades para esquerda e direita forem: [0.6, 0.4], então a esquerda 
    ### deve ser escolhida 60% das vezes e a direita 40% das vezes. Nos métodos anteriores (escalada)
    ### isto indicaria a escolha da esquerda, a direita nunca seria escolhida ...  
    ### No comando abaixo, usamos prob[1], probabilidade de mover pra direita (->), então, se este 
    ### valor for menor que 0.2, a dist. Bernoulli resultará na ação 0, o que inverte a direção (<-)  
    ### Se o valor for maior que 20% o resultado será 1, e a ação continuará movendo pra direita (->)
    action = int(tk.bernoulli(prob[1]).item())    ### Ação definida pela dist. Bernoulli -> Monte Carlo
    ### Agora vamos calcular os gradientes da Saída com relação aos pesos, que neste caso remete a
    ### derivada do tensor de probabilidades:
    ###
    ###         Derivada =  | prob[0]    0     |  -   |     prob[0]²     prob[0]*prob[1] |      
    ###                     |   0      prob[1] |      | prob[1]*prob[0]      prob[1]²    |
    ###
    ### O valor desta derivada deve ser normalizado para evitar erros de escala (explosão ou implosão) -> LOG 
    d_smx = tk.diag(prob) - prob.view(-1, 1)*prob ### Derivada da função de ativação - pra calcular o gradiente
    d_log = d_smx[action] / prob[action]          ### Logaritmo da derivada - pra melhorar a escala dos cálculos
    grad = state.view(-1, 1)*d_log                ### Gradiente calculado - vide referência [1]
    grads.append(grad)                            ### Registrando os gradientes
    ### Pronto, agora devemos ir para o próximo passo
    state, reward, is_done, info = env.step(action)   ### Próximo passo do episódio
    tt_rew += reward                                  ### Calculando a recompensa total deste episódio
    if is_done:                                       ### Parar se o agente cair
      break
  return tt_rew, grads     ### Retorno da função -> recompensa e lista de gradientes

### Ambiente de simulação
carpo = gym.make("CartPole-v0")
carpo.seed(123)    ### Pra fixar os eventos randômicos
carpo.reset()      ### Iniciando o ambiente

### Dimensões da matriz W
n_state = carpo.observation_space.shape[0]  ### Estados
n_action = carpo.action_space.n             ### Ações
mat_w = tk.rand(n_state, n_action)          ### matriz de pesos - aleatória

### Realizando o treinamento - 1000 episódios
for epi in tqdm(range(num_epi)):
  tt_reward, gradients = run_epi(carpo, mat_w)    ### Rodando a simulação
  for i, gradient in enumerate(gradients):        ### Ajustando a matriz pesos
    mat_w += alfa*gradient*(tt_reward - i)        ### ...conforme a taxa de aprendizado
  list_tt_rew.append(tt_reward)                   ### Registrando as recompensas 

### Plotando a convergência das recompensas
pl.plot(list_tt_rew)
pl.title("Rewards for each episode")
pl.xlabel("episodes")
pl.ylabel("rewards")
pl.show()

### Resultado para a matriz W
print("Melhor matriz de pesos:\n", mat_w)

### Teste da melhor matriz peso
ks = 0                                         ### Contagem de sucessos
max_rew = max(list_tt_rew)                     ### Maior recompensa 
for test in range(num_test):                   ### 100 testes
  rew_test, grad_test = run_epi(carpo, mat_w)  ### Recompensa pra cada teste
  if rew_test > (0.95*max_rew):                ### Acima de 95% de sucessos
    ks += 1                                    ### ... a gente conta

### Precisão:
print("Precisão deste método = ", 100*(ks/num_test), "%")

### FIM

### Referências:
# [1] https://towardsdatascience.com/policy-gradient-methods-104c783251e0
