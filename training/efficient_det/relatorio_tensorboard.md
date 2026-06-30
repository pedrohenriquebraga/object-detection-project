# Relatório das Métricas do TensorBoard

Este documento detalha o comportamento das métricas observadas no TensorBoard, como cada uma age nas camadas do modelo, os sinais interpretados a partir dos gráficos reais do experimento e as ações recomendadas para mitigar o comportamento de sobreajuste (*overfitting*) identificado.

---

## 1. Métricas de Desempenho (Treino / Validação)

### `epoch_accuracy`
* **O que é:** A acurácia mede a proporção de previsões corretas feitas pelo modelo em relação ao total de amostras processadas durante uma época completa (uma passagem inteira pelo *dataset*). É uma métrica de avaliação intuitiva, geralmente variando de 0 a 1 (ou 0% a 100%), que indica o quão bem o modelo está a classificar ou acertar os dados naquele momento específico do treino (dados já vistos) ou da validação (dados novos).
* **Como age nas camadas:** Reflete o efeito agregado de todas as camadas na decisão final; aumentos geralmente correspondem aos pesos das camadas classificadoras sendo alinhados com a superfície de decisão.
* **Efeito no treinamento:** Subida consistente sugere aprendizado; subida muito rápida com validação estática indica *overfitting* localizado nas camadas finais.
* **Sinais comuns:** Acurácia de treino sobe continuamente enquanto a acurácia de validação fica estável (platô).
* **Análise do modelo atual:** No gráfico de `epoch_accuracy`, a curva de treino (azul) sobe de forma acentuada aproximando-se de 100% (1.0), enquanto a curva de validação (verde) estabiliza-se precocemente em um patamar significativamente inferior. Isso indica que as camadas aprenderam a mapear os dados de treino quase perfeitamente, mas falham em generalizar para dados não vistos.

### `epoch_loss`
* **O que é:** A *loss* (função de perda ou custo) quantifica a divergência entre as previsões do modelo e os rótulos reais. Diferente da acurácia, que é binária (acertou ou errou), a *loss* penaliza o modelo de forma contínua pelo grau de incerteza da previsão (usando, por exemplo, *Cross-Entropy* para classificação). É o valor matemático fundamental que o otimizador tenta minimizar ativamente calculando os gradientes após cada processamento.
* **Como age nas camadas:** A *loss* guia o sinal de gradiente que ajusta os pesos em todas as camadas do modelo.
* **Efeito no treinamento:** Queda consistente mostra que os gradientes estão reduzindo o erro. Variações bruscas ou divergências indicam instabilidade ou inadequação da taxa de aprendizado.
* **Sinais comuns:** *Loss* de treino caindo enquanto a de validação sobe de forma contínua após um ponto de mínimo.
* **Análise do modelo atual:** O gráfico de `epoch_loss` demonstra o comportamento clássico de *overfitting*. A curva de treino decresce agressivamente em direção ao zero. Em contrapartida, a curva de validação atinge um ponto mínimo nas primeiras épocas e, a partir daí, passa a subir de forma constante, indicando que o modelo começou a memorizar ruídos do dataset de treino.

### `epoch_learning_rate`
* **O que é:** A taxa de aprendizado (*learning rate*) é o hiperparâmetro que define o tamanho do "passo" matemático que o modelo dá ao atualizar os seus pesos na direção calculada para minimizar a *loss*. Pode ser um valor fixo estático ou um valor dinâmico que decai e se ajusta ao longo do tempo através de um *scheduler*, determinando a velocidade e a estabilidade com que o modelo assimila os padrões.
* **Como age nas camadas:** Controla a magnitude das atualizações dos pesos em todas as camadas.
* **Efeito no treinamento:** LR alto acelera a convergência inicial, mas pode causar oscilações e impedir o modelo de atingir o mínimo global; LR baixo torna o aprendizado excessivamente lento.
* **Análise do modelo atual:** O comportamento do LR permitiu uma convergência suave e rápida nas primeiras épocas, mas devido à falta de regularização, facilitou a rápida divergência da perda de validação.

### `evaluation_loss_vs_iterations` e `evaluation_accuracy_vs_iterations`
* **O que são:** Representam a evolução contínua das métricas de perda e acurácia computadas no conjunto de validação (dados isolados não utilizados para treinar), mas plotadas iterativamente ao longo do processo (frequentemente após um determinado número de *batches*, sem esperar a época inteira acabar). Elas fornecem uma visão altamente granular de como a capacidade de generalização do modelo flutua microscopicamente a cada rodada de atualização de pesos.
* **Como agem nas camadas:** Expõem diretamente se as representações intermediárias e os filtros aprendidos são genéricos o suficiente para dados nunca vistos.
* **Análise do modelo atual:** O gráfico de `evaluation_loss_vs_iterations` atinge o seu vale (ponto ideal de generalização) logo no primeiro terço do treinamento, subindo de forma linear e contínua nas iterações seguintes. Esse comportamento valida a necessidade urgente de um mecanismo de interrupção antecipada (*Early Stopping*).

---

## 2. Estatísticas Internas do Modelo (Camadas)

Essas métricas aparecem na forma de histogramas e distribuições temporais, fornecendo pistas sobre estabilidade numérica, saturação e dinâmica dos gradientes.

### `kernel` (Pesos da Camada)
* **O que é:** Os *kernels* (ou pesos) são as matrizes numéricas aprendíveis centrais do modelo. Nas camadas convolucionais, atuam como filtros que detectam características visuais (bordas, texturas). Nas camadas densas (*fully-connected*), representam a força da conexão entre neurônios, ditando a importância de cada característica extraída para a classificação final. O TensorBoard mapeia a distribuição estatística de todos esses milhões de valores.
* **Como age nas camadas:** Mudanças nos *kernels* indicam o aprendizado e refinamento de filtros visuais ou combinações lineares.
* **Análise do modelo atual:** Os histogramas tridimensionais (gráficos roxos) de `kernel` mostram uma distribuição bem comportada, centrada em torno de zero, que se alarga de forma gradual e suave com o passar das épocas. Isso demonstra excelente estabilidade numérica: não há indícios de explosão de gradiente (alargamento excessivo) nem de desvanecimento (linhas extremamente finas e estáticas).

### `bias`
* **O que é:** O *bias* (viés) é um parâmetro linear aditivo extra associado a cada neurônio ou filtro, independente da entrada. Ele atua deslocando o limiar da função de ativação (para cima ou para baixo, para a esquerda ou para a direita). Isso garante que, mesmo quando os valores de entrada são nulos, a rede tenha uma linha de base operacional e maior flexibilidade geométrica para ajustar sua curva aos dados.
* **Como age nas camadas:** Ajustam os limiares operacionais das funções de ativação.
* **Análise do modelo atual:** A distribuição de `bias` exibe um formato de sino estreito e simétrico que evolui de forma estável. Isso comprova que a rede está ajustando seus limiares de ativação corretamente, sem criar vieses severos ou desproporcionais para classes específicas.

### `gamma` e `beta` (BatchNorm / LayerNorm)
* **O que são:** São os dois parâmetros ajustáveis exclusivos das camadas de normalização. Depois que a camada padroniza as saídas (subtraindo a média e dividindo pelo desvio padrão), ela aplica o `gamma` como um fator multiplicador de escala e o `beta` como um fator de deslocamento da média. Isso permite que a rede "desfaça" a normalização apenas o necessário para restaurar algum poder expressivo que o nivelamento estrito possa ter apagado.
* **Como agem nas camadas:** Controlam a amplitude e a média das saídas normalizadas antes da próxima ativação não-linear.
* **Análise do modelo atual:** As distribuições de `gamma` e `beta` mantêm-se densas, simétricas e estáveis ao longo do tempo. O fato de `gamma` não colapsar em direção a zero indica que as camadas de normalização continuam ativas e desempenhando seu papel de estabilização do fluxo de ativações entre as camadas.

### `mean` e `variance` (Ativação por Batch)
* **O que são:** Refletem as estatísticas matemáticas (média e variância absolutas) computadas localmente sobre as ativações (sinais que saem dos neurônios) de um único *mini-batch* durante o processo de *forward pass* no treino. Eles mostram se, no calor do momento, os valores numéricos estão concentrados e equilibrados, ou se a rede sofre de picos de magnitude que causariam instabilidade.
* **Efeito no treinamento:** Valores controlados evitam a saturação de funções como ReLU (neurônios mortos) ou Sigmoid/Tanh (gradiente nulo).
* **Análise do modelo atual:** Os gráficos mostram oscilações controladas dentro de uma faixa saudável, confirmando que a dinâmica interna dos dados durante o *forward pass* está equilibrada.

### `moving_mean` e `moving_variance`
* **O que são:** São estimativas globais e suavizadas do comportamento estatístico da rede, mantidas através de uma média móvel exponencial durante o treinamento. A função vital delas é atuar como substitutas fixas para a média e a variância originais do *batch* assim que o modelo é colocado em produção ou modo de avaliação. Isso garante que a inferência em imagens avulsas seja determinística e confiável, não dependendo do tamanho ou do conteúdo de um *batch* local.
* **Análise do modelo atual:** Apresentam curvas suavizadas e bem calibradas nos histogramas, garantindo consistência no comportamento do modelo quando alternado para o modo de avaliação (`model.eval()`).

---

## 3. Leitura Conjunta e Diagnóstico Técnico

A análise integrada de todas as curvas permite formular um diagnóstico preciso:

* **Diagnóstico Geral:** O modelo apresenta um quadro nítido de **Overfitting Severo (Sobreajuste)**. 
* **Justificativa:** A saúde interna da rede é excelente — demonstrada pela perfeita estabilidade e evolução dos histogramas de `kernel`, `bias`, `gamma` e `beta`. Os gradientes estão fluindo perfeitamente e a rede tem alta capacidade de aprendizado, o que causa a queda contínua da `epoch_loss` de treino. Contudo, essa capacidade matemática é excessiva frente à complexidade ou volume do dataset atual, fazendo com que as camadas profundas decorem os dados de treino em vez de aprender características genéricas.
