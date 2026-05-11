[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ARkoM8Jo)
# Diagnóstico de retomada - Aprendizado de Máquina

Esta atividade serve para mapear o que você já domina em Aprendizado de Máquina depois das atividades anteriores da disciplina.

Responda individualmente. Use suas palavras. Rode o código quando possível. Se usar IA depois da primeira tentativa, registre o uso na seção 8.

Prazo: 11/05/2026 às 23:59, horário de Fortaleza.

## 1. Mapa do que eu lembro

Marque cada tópico como: lembro bem, lembro parcialmente, não lembro, nunca vi ou não tenho certeza.

- vetores, matrizes e produto escalar: **lembro parcialmente**
- média, desvio padrão e correlação: **lembro bem**
- probabilidade condicional e Teorema de Bayes: **lembro parcialmente**
- regressão linear: **lembro parcialmente**
- classificação supervisionada: **lembro bem**
- treino, teste e validação: **lembro bem**
- normalização ou padronização de dados: **lembro bem**
- KNN: **lembro parcialmente**
- árvore de decisão: **lembro bem**
- matriz de confusão: **nunca vi**
- acurácia, precisão, recall e F1-score: **lembro bem**
- overfitting e underfitting: **lembro bem**
- validação cruzada: **lembro bem**
- Random Forest: **lembro parcialmente**
- XGBoost ou boosting: **nunca vi**
- `predict_proba()`: **lembro bem**
- SQL/ETL aplicado a dados: **lembro parcialmente**
- simulação de Monte Carlo: **lembro bem**

## 2. O que foi trabalhado antes

Explique, em 8 a 12 linhas:

1. quais desses tópicos você lembra de ter trabalhado na disciplina;
- regressão linear
- treino, teste e validação
- normalização ou padronização de dados
- KNN
- árvore de decisão
- acurácia, precisão, recall e F1-score
- overfitting e underfitting
- validação cruzada
- Random Forest
- XGBoost ou boosting: foi trabalhado, mas não cheguei a estudar e usar
- `predict_proba()`
- SQL/ETL aplicado a dados
- simulação de Monte Carlo

2. quais atividades ou exemplos você lembra; <br>
Teve varias atividades, cada uma era um progresso que ajudava na próxima. Teve algumas para lembrar os fundamentos da algebra linear. Outros conceitos para ajudar na construção do modelo. O exemplos que lembro, usar a regressão linear para encontrar o valor do aluguel de um imóvel e um mais recente que foi de usar o random forest e o monte carlo para ver a probabilidade de cada time ser ganhador em um campeonato.

3. o que você conseguiu fazer com autonomia; <br>
Acredito que separar os dados para o treino e o teste, aplicar a normalização, construir os modelos de regressão linear e calcular as métricas. Mas mesmo assim ainda consultei alguns exemplos antigos como base.

4. o que você só conseguiu fazer seguindo roteiro; <br>
Lidar com as partes mais matématicas, construir e treinar os modelos mais complexos de árvore de decisão e random forest

5. qual assunto precisa ser retomado com mais urgência. <br>
Sinto que a manipulação avançada de dados, e evitar o vazamento de dados.

## 3. Conceitos essenciais

Responda com suas palavras e dê um exemplo simples.

1. O que é aprendizado supervisionado?<br>
Basicamente você coloca o modelo para treinar com dados que já sabemos as respostas, sendo tipo um professor. Por exemplo, dou duas imagens para ele e digo que uma é um carro e outra é uma moto, depois dou uma terceira imagem, ele tem que me dizer se aquilo é uma moto ou um carro, como base no que ele aprendeu, depois da resposta ele vê se acertou ou errou, e assim ele vai aprendendo com o conjunto de dados que foram passados.

2. O que é uma tarefa de classificação?<br>
É classificar, usando o mesmo exemplo de cima, ele vai classicificar se aquilo é um carro ou uma moto, é tipo as prateleiras do mercado que são divididas, cada uma com um tipo diferente de mercadoria.

3. O que são features e target?<br>
As features são os dados de características, o target é o alvo onde queremos chegar. Por exemplo, eu pego uma tabela de dados e nela tem diversas informações, usando o mesmo exemplo do carro e da moto, o carro possui diversas características que fazem ele ser um carro (target), assim como a moto possui diversas características que fazem ela ser uma moto (target). É através das features que o modelo aprende a identificar.

4. Para que serve separar treino e teste?<br>
Pelo que lembro, essa separação serve para treinar e testar o modelo, pegamos um conjunto de dados e dividimos esses dados, uma parte vai para o treino, onde o modelo aprende atraves das features a identificar e classificar o target. Depois nós testamos o modelo para ver se ele realmente aprendeu, usando os dados que ficaram de fora do treino, ou seja, ele nunca viu aqueles dados, então com base no que ele aprendeu ele vai fazer uma predição e vamos ver o quão bem ele é capaz de ir.

5. O que é overfitting?<br>
O overfitting é quando o modelo aprende até demais, só que ele não aprendeu de verdade, ele apenas decorou as respostas, então ele não consegue generalizar quando aparece dados novos. Ainda usando o mesmo exemplo, suponhamos que eu treine o modelo apenas com carros vermelhos, ele decora essa característica e todo carro vermelho que aparecer ele vai acertar, mas quando aparecer um carro branco ele vai errar.

6. Por que acurácia pode ser uma métrica enganosa?<br>
A acuráncia é a taxa de acerto, se ela tiver alta significa que está acertando muito, mas isso não significa que essa taxa é confiavel porque ela pode estar errando no mais importante, principalmente quando está com overfitting. Usando o exemplo do carro, se eu tiver 95 carros vermelhos e 5 brancos e ela sempre chutar vermelho ela vai ter uma taxa de acerto de 95%, ela chuta, mas não aprendeu.

## 4. Diagnóstico prático com Scikit-Learn

No arquivo `diagnostico_ml.py`, use o dataset `load_breast_cancer` do Scikit-Learn e faça:

1. carregue os dados;
2. separe `X` e `y`;
3. divida em treino e teste;
4. treine uma regressão logística;
5. treine uma árvore de decisão;
6. mostre matriz de confusão, acurácia, precisão, recall e F1-score para cada modelo;
7. compare o desempenho em treino e teste;
8. escreva aqui qual modelo generalizou melhor e por quê.

Se não conseguir terminar tudo, registre até onde chegou e qual erro apareceu.

### Resultados

Cole aqui os principais resultados do seu código.

```text
=== Regressão logística ===
Acurácia treino: 0.958
Acurácia teste: 0.958
Precisão teste: 0.947
Recall teste: 0.989
F1-score teste: 0.967
Matriz de confusão:
[[48  5]
 [ 1 89]]
Probabilidades das 5 primeiras amostras de teste:
[[0.01854698 0.98145302]
 [0.99816861 0.00183139]
 [0.17716357 0.82283643]
 [0.2344741  0.7655259 ]
 [0.19796601 0.80203399]]

=== Árvore de decisão ===
Acurácia treino: 1.000
Acurácia teste: 0.923
Precisão teste: 0.954
Recall teste: 0.922
F1-score teste: 0.938
Matriz de confusão:
[[49  4]
 [ 7 83]]
Probabilidades das 5 primeiras amostras de teste:
[[0. 1.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [0. 1.]]
```

### Interpretação

Qual modelo generalizou melhor? Explique usando as métricas e a comparação entre treino e teste.

Resposta: A regressão logística generalizou melhor. A taxa de acuráncia do teste e de treino são praticamente a mesma, indicando que o modelo aprendeu os padrões, e as métricas de recall e F1-score mostram que se sairam bem em novos dados.

## 5. Probabilidade e interpretação

Escolha um dos modelos treinados e responda: **Regressão logística**

1. O modelo produz probabilidade com `predict_proba()`? <br>
Sim.
2. O que significa uma probabilidade alta para uma classe? <br>
Significa que o modelo acha que aquela classe tem mais chances de ser a resposta. Por exemplo, onde [0.01854698 0.98145302], ele está confiante que a classe 1 tem 98.14% de ser verdadeira.
3. Probabilidade alta garante que a previsão está correta? Explique.<br>
Não, um exemplo que posso citar foi o trabalho anterior que fiquemos, nosso modelo de random forest através do predict_proba() deu uma alta probabilidade de um time ganhar, mas quem realmente ganhou foi o time que tinha apenas 1% de ganhar. Isso acontece porque o modelo apenas usa os dados anteriores para tentar adivinhar o futuro usando padrões e as vezes não representa o mundo real.
4. Em um problema real, qual seria o risco de confiar cegamente nessa previsão?<br>
O risco seria muito alto, principalmente na área da saúde. O danos poderia ser graves, por isso é sempre bom ter alguém humano acompanhando e avaliando esses resultados.

Resposta:

## 6. Generalização

Compare treino e teste:

1. Há sinal de overfitting?<br>
Sim, na árvore de decisão. Acurácia treino: 1.000 x Acurácia teste: 0.923. A árvore aprendeu detalhes demais do treino e não conseguiu generalizar quando apareceu dados novos
2. Há sinal de underfitting?<br>
Não, na árvore teve overfitting e na regressão tivemos o valor de treino e teste iguais a 0.958.
3. O que você tentaria mudar para melhorar o resultado?<br>
Eu tentaria limitar um pouco o treinamento da árvore de decisão, configurando as folhas e a profundidade da árvore.
4. O que você precisaria estudar melhor para responder com mais segurança?<br>
Acredito que preciso entender melhor como prever essas consequências, entender por que determinados comportamentos acontecem no modelo, como evitar e onde devo arrumar para corrigir.

Resposta:

## 7. Ponto de dificuldade

Escolha um tópico da lista inicial e escreva:

1. o que você entende dele;
2. onde você se confunde;
3. que tipo de explicação ajudaria: exemplo no quadro, notebook guiado, exercício curto, revisão matemática, visualização ou projeto pequeno.

Resposta: Escolhi o tópico de Random Forest. Entendo que o Random Forest junta várias árvores de decisão para melhorar o resultado final, mas me confundo na parte de como as árvores são criadas com subconjuntos diferentes dos dados. Também tenho dificuldade em entender os hiperparâmetros, como quantidade de árvores, profundidade e critérios de divisão. Acho que exemplos visuais ajudaria bastante, mostrando passo a passo como cada árvore aprende e alguns exercícios curtos alterando hiperparâmetros também ajudariam a entender o impacto de cada configuração.

## 8. Uso de IA, se houver

Se você usou IA depois da primeira tentativa, registre:

```text
Pergunta feita: Usei para entender melhor o que era regressão logística e o parâmetro max_iter
Resumo da resposta:  Explicou que a regressão logística é um modelo de classificação supervisionada que calcula probabilidades para cada classe. Também explicou que o max_iter define o número máximo de iterações que o algoritmo pode executar para conseguir convergir durante o treinamento.
Como eu verifiquei: Testei diferentes valores
O que eu alterei na minha resposta: 
O que ainda não entendi: Ainda tenho dificuldade em entender mais profundamente como acontece a convergência do modelo
```

## Submissão no Moodle

Depois de finalizar, copie no Moodle:

```text
Repositório:
Commit final:
Autoavaliação: nível atual, maior dificuldade e tópico que precisa ser retomado.
```
