# Relatório das métricas do TensorBoard

Este texto explica, de forma mais simples, o que cada métrica do TensorBoard quer dizer e como ela ajuda a entender se o modelo está aprendendo bem.

As métricas deste projeto podem ser vistas em dois grupos:

1. As que mostram o desempenho do treino e da validação.
2. As que mostram como os pesos e camadas internas do modelo estão se comportando.

## 1. Métricas de desempenho

### epoch_accuracy

Mostra a porcentagem de acertos do modelo no treino ao final de cada época. Em outras palavras, indica quantas previsões estavam certas naquele ciclo.

Quando essa métrica sobe, é um bom sinal: o modelo está aprendendo. Se ela sobe muito rápido, mas a validação não acompanha, pode ser que o modelo esteja decorando o treino em vez de aprender de verdade.

Exemplo: se a acurácia sobe de 45% para 90%, o modelo está melhorando bastante. Mas se a validação fica parada em 60%, isso sugere que ele ainda não generaliza bem para dados novos.

### epoch_loss

Mostra o erro médio do modelo no treino. Quanto menor esse valor, melhor.

Na prática, a loss ajuda a ver se o modelo está ficando mais seguro nas respostas. Às vezes a acurácia quase não muda, mas a loss cai, o que significa que o modelo está acertando com mais confiança.

Exemplo: um modelo pode continuar com 82% de acerto, mas a loss cair de 0,9 para 0,4. Isso mostra que ele ainda está refinando o aprendizado.

### epoch_learning_rate

Mostra o tamanho do passo usado para atualizar os pesos em cada época.

Se esse valor estiver alto demais, o treino pode ficar instável e oscilar. Se estiver baixo demais, o aprendizado fica muito lento. Em geral, essa curva ajuda a entender por que o modelo acelerou ou desacelerou durante o treinamento.

Exemplo: começar com uma taxa maior e depois diminuir ajuda o modelo a aprender rápido no início e fazer ajustes mais finos no final.

## 2. Métricas de validação

### evaluation_accuracy_vs_iterations

Mostra a acurácia na validação ao longo das iterações.

Ela ajuda a ver se o modelo está realmente funcionando bem em dados que ele não viu no treino. Se essa curva sobe e depois cai, pode ser um sinal de que o melhor ponto do treinamento já passou.

Exemplo: a validação sobe até 78% e depois começa a cair. Nesse caso, o melhor modelo talvez seja aquele salvo no pico, e não o último do treino.

### evaluation_loss_vs_iterations

Mostra o erro na validação ao longo das iterações.

Essa é uma das curvas mais importantes para saber se o modelo está generalizando bem. Se o erro de treino cai, mas o da validação sobe, normalmente é sinal de overfitting.

Exemplo: o erro no treino cai de 0,6 para 0,2, mas o da validação sobe de 0,7 para 1,1. Isso mostra que o modelo está indo muito bem no treino, mas piorando em dados novos.

## 3. Estatísticas internas do modelo

Essas métricas aparecem como histogramas ou distribuições. Elas não medem diretamente se o modelo está bom ou ruim, mas ajudam a enxergar se as camadas internas estão estáveis.

### kernel

Representa os pesos principais das camadas.

Se esses pesos ficam muito próximos de zero por muito tempo, a camada pode estar aprendendo pouco. Se eles se espalham demais, pode haver instabilidade.

Exemplo: em uma camada convolucional, um kernel bem distribuído indica que o modelo está aprendendo filtros úteis, como bordas e texturas.

### bias

São valores de ajuste que ajudam a camada a fazer previsões melhores.

Biases muito grandes ou muito estranhos podem indicar que o modelo está tentando compensar algum problema nos pesos.

Exemplo: se um bias cresce demais na camada final, o modelo pode começar a favorecer uma classe mais do que deveria.

### gamma

É um valor que ajusta a força da saída de uma camada de normalização.

Se ele crescer demais ou cair demais, a camada pode perder sua utilidade. Quando está estável, ajuda o treino a ficar mais consistente.

Exemplo: um gamma equilibrado ajuda a manter os sinais da rede em uma faixa boa para aprendizado.

### beta

É outro valor usado em camadas de normalização, funcionando como um deslocamento.

Quando esse valor muda de forma muito brusca, pode ser um sinal de que as ativações estão variando demais.

Exemplo: quando o beta começa perto de zero e vai se ajustando aos poucos, a rede está encontrando uma distribuição melhor para os dados.

### mean

Mostra a média dos valores acompanhados pela camada.

Se a média sobe ou desce demais, isso pode indicar que a camada está saindo do padrão esperado.

Exemplo: uma média que cresce sem parar pode mostrar que a camada está ficando menos estável ao longo do treino.

### variance
 
Mostra o quanto os valores estão espalhados em torno da média.

Se a variância é muito baixa, a camada pode estar produzindo respostas parecidas demais. Se é muito alta, o modelo pode ficar instável.

Exemplo: uma variância muito pequena pode fazer o modelo tratar imagens diferentes como se fossem parecidas demais.

### moving_mean

É uma média “acumulada” que o modelo usa para funcionar melhor na validação e na produção.

Se essa curva é suave e estável, é um bom sinal. Se ela oscila demais, o treinamento pode estar irregular.

Exemplo: com dados muito variados, um moving_mean estável ajuda o modelo a se comportar de forma mais previsível depois de treinado.

### moving_variance

É a versão acumulada da variância usada na validação e na inferência.

Ela ajuda o modelo a continuar funcionando com o mesmo comportamento depois que o treino termina.

Exemplo: se essa métrica fica estável, o modelo tende a manter um desempenho mais parecido quando for exportado para uso real.

### count

Mostra quantas vezes uma métrica foi registrada.

Ela não diz se o modelo está bom, mas ajuda a saber se existe informação suficiente para analisar a curva com confiança.

Exemplo: uma métrica com poucos pontos ainda não dá uma leitura tão segura quanto uma curva com muitos registros.

## 4. Como ler tudo junto

O ideal é observar as curvas em conjunto.

Se a acurácia sobe e o erro cai ao mesmo tempo, o modelo está aprendendo. Se isso também acontece na validação, a tendência é que ele esteja generalizando bem. Se o treino melhora, mas a validação piora, geralmente o modelo está exagerando no aprendizado do treino e precisa parar antes, usar mais dados ou ter alguma forma de regularização.

As métricas internas, como kernel, bias, gamma, beta, mean, variance, moving_mean e moving_variance, servem como sinais de alerta. Elas ajudam a perceber quando alguma camada está instável ou quando os pesos estão saindo do esperado.
