## 🧠 5. Feature Engineering

Foi criada uma nova variável denominada `comprometimento_renda`.

A fórmula utilizada foi:

```python
comprometimento_renda = (loan_amnt / person_income) * 100
```

Essa variável representa o percentual do valor do empréstimo em relação à renda do cliente.

Antes do cálculo, foi verificada a existência de valores iguais a zero na variável `person_income`, evitando possíveis divisões por zero.

O resultado encontrado foi:

```text
Rendas iguais a zero: 0
```

Dessa forma, não houve risco de divisão por zero na criação da nova variável.

---

## 🔧 6. Preparação dos Dados para Modelagem

Antes do treinamento dos modelos, foram realizadas etapas para transformar a base em um formato adequado aos algoritmos de Machine Learning.

### 6.1 Encoding das Variáveis Categóricas

As variáveis categóricas foram convertidas para formato numérico utilizando **One-Hot Encoding**.

Após o processo de encoding:

- Base tratada: **13 colunas**
- Base para modelagem: **24 colunas**

O número de registros permaneceu inalterado.

### 6.2 Separação entre Variáveis Preditoras e Variável Alvo

A variável alvo foi separada das demais variáveis:

```python
X = df_modelo.drop(columns=["loan_status"])
y = df_modelo["loan_status"]
```

Após a separação:

- `X`: 32.411 registros e 23 variáveis preditoras
- `y`: 32.411 valores

### 6.3 Divisão entre Treino e Teste

A base foi dividida da seguinte forma:

- **80%** para treino
- **20%** para teste

Foi utilizado:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

O parâmetro `stratify=y` foi utilizado para preservar a proporção das classes nos conjuntos de treino e teste.

As distribuições permaneceram próximas de:

| Classe | Treino | Teste |
|---|---:|---:|
| `0` - Adimplente | 78,13% | 78,13% |
| `1` - Inadimplente | 21,87% | 21,87% |

---

## ⚖️ 7. Balanceamento com SMOTE

Como a variável alvo apresentava desbalanceamento, foi utilizado **SMOTE** somente no conjunto de treino.

O conjunto de teste não foi balanceado, preservando sua distribuição original e evitando vazamento de dados.

### Antes do SMOTE

| Classe | Registros |
|---|---:|
| `0` | 20.257 |
| `1` | 5.671 |

### Após o SMOTE

| Classe | Registros |
|---|---:|
| `0` | 20.257 |
| `1` | 20.257 |

O conjunto de treino passou de:

```text
25.928 registros
```

para:

```text
40.514 registros
```

O balanceamento foi realizado exclusivamente no treino para evitar **Data Leakage**.

---

## 📏 8. Escalonamento

O modelo KNN utiliza medidas de distância entre os registros e, por isso, é sensível à escala das variáveis.

Foi utilizado o `StandardScaler`.

O scaler foi ajustado somente no conjunto de treino balanceado:

```python
X_train_knn = scaler.fit_transform(X_train_bal)
```

E posteriormente aplicado ao conjunto de teste:

```python
X_test_knn = scaler.transform(X_test)
```

Dessa forma, nenhuma informação do conjunto de teste foi utilizada durante o ajuste do escalonador.

A Árvore de Decisão utilizou os dados sem escalonamento, pois esse algoritmo não depende de medidas de distância entre os registros.

---

## 🤖 9. Modelagem

Foram treinados e avaliados dois algoritmos:

- **K-Nearest Neighbors (KNN)**
- **Árvore de Decisão**

O objetivo foi comparar diferentes hiperparâmetros e analisar possíveis sinais de overfitting.

---

### 9.1 K-Nearest Neighbors

Foram avaliados quatro valores de `K`:

- K = 3
- K = 5
- K = 7
- K = 9

#### Resultados

| K | Acurácia Treino | Acurácia Teste |
|---:|---:|---:|
| 3 | 95,12% | 87,38% |
| 5 | 93,89% | 88,26% |
| 7 | 93,25% | 88,85% |
| 9 | 92,79% | **89,36%** |

Os resultados mostram que valores menores de K apresentaram maior desempenho no conjunto de treino, porém uma diferença maior em relação ao conjunto de teste.

O modelo com `K = 3` apresentou maior tendência ao overfitting.

Conforme o valor de K aumentou:

- a acurácia de treino diminuiu;
- a acurácia de teste aumentou;
- a diferença entre treino e teste diminuiu.

Entre os valores avaliados, `K = 9` apresentou o melhor equilíbrio entre desempenho e capacidade de generalização.

**Configuração selecionada:**

```text
K = 9
```

**Acurácia no teste:**

```text
89,36%
```

---

### 🌳 9.2 Árvore de Decisão

Foram avaliados os seguintes valores de `max_depth`:

- `max_depth = 3`
- `max_depth = 5`
- `max_depth = 7`
- sem limite de profundidade

#### Resultados

| max_depth | Acurácia Treino | Acurácia Teste |
|---|---:|---:|
| 3 | 84,88% | 86,60% |
| 5 | 88,68% | 89,08% |
| 7 | 90,51% | **90,53%** |
| Sem limite | 100,00% | 87,97% |

A configuração sem limite de profundidade alcançou:

```text
100% de acurácia no treino
```

porém apresentou apenas:

```text
87,97% de acurácia no teste
```

Esse comportamento demonstra um caso claro de **overfitting**.

A árvore passou a se ajustar excessivamente aos dados de treinamento e perdeu capacidade de generalização.

Já `max_depth = 7` apresentou praticamente o mesmo desempenho entre treino e teste.

**Configuração selecionada:**

```text
max_depth = 7
```

**Acurácia no teste:**

```text
90,53%
```

---

### 🌲 9.3 Interpretação Visual da Árvore

Também foi gerada uma representação visual de uma Árvore de Decisão com `max_depth = 3`.

Entre as variáveis utilizadas nos primeiros níveis da árvore apareceram:

- `loan_grade_D`
- `comprometimento_renda`
- `person_home_ownership_RENT`
- `loan_intent_MEDICAL`
- `person_emp_length`

A visualização permite compreender como o algoritmo realiza sucessivas divisões dos dados até chegar à classificação final entre clientes adimplentes e inadimplentes.

---

## 📈 10. Avaliação dos Modelos

As melhores configurações dos dois modelos foram avaliadas utilizando:

- Precision
- Recall
- F1-score
- Acurácia
- Matriz de Confusão

---

### 10.1 KNN

**Configuração:**

```text
K = 9
```

#### Classification Report — Classe Inadimplente

| Métrica | Resultado |
|---|---:|
| Precision | 0,81 |
| Recall | 0,68 |
| F1-score | 0,74 |

**Acurácia geral:**

```text
89%
```

#### Matriz de Confusão

| Resultado | Quantidade |
|---|---:|
| Verdadeiros Negativos | 4.835 |
| Falsos Positivos | 230 |
| Falsos Negativos | 460 |
| Verdadeiros Positivos | 958 |

---

### 10.2 Árvore de Decisão

**Configuração:**

```text
max_depth = 7
```

#### Classification Report — Classe Inadimplente

| Métrica | Resultado |
|---|---:|
| Precision | 0,86 |
| Recall | 0,68 |
| F1-score | 0,76 |

**Acurácia geral:**

```text
91%
```

#### Matriz de Confusão

| Resultado | Quantidade |
|---|---:|
| Verdadeiros Negativos | 4.902 |
| Falsos Positivos | 163 |
| Falsos Negativos | 451 |
| Verdadeiros Positivos | 967 |

---

## 💼 11. Análise de Negócio

A avaliação dos modelos não deve considerar apenas a acurácia.

Em um problema de risco de crédito, diferentes tipos de erro possuem impactos distintos.

### Falso Positivo

Um cliente adimplente é classificado como inadimplente.

Possíveis consequências:

- recusa de crédito para um bom cliente;
- perda de receita;
- perda de relacionamento com o consumidor;
- oportunidade comercial perdida para um concorrente.

### Falso Negativo

Um cliente inadimplente é classificado como adimplente.

Possíveis consequências:

- concessão de crédito para um cliente de maior risco;
- atraso no pagamento;
- inadimplência;
- perda financeira direta.

Por esse motivo, os **Falsos Negativos** merecem atenção especial nesse cenário.

---

## 🏆 12. Comparação Final

| Métrica | KNN | Árvore de Decisão |
|---|---:|---:|
| Acurácia de Teste | 89,36% | **90,53%** |
| Precision - Inadimplente | 0,81 | **0,86** |
| Recall - Inadimplente | 0,68 | 0,68 |
| F1-score - Inadimplente | 0,74 | **0,76** |
| Falsos Positivos | 230 | **163** |
| Falsos Negativos | 460 | **451** |

A Árvore de Decisão apresentou:

- maior acurácia;
- maior precision para inadimplentes;
- maior F1-score;
- menor quantidade de Falsos Positivos;
- menor quantidade de Falsos Negativos.

Os dois modelos apresentaram recall aproximado de 68% para a classe inadimplente.

---

## 📝 13. Resumo Executivo

A análise exploratória revelou uma base com:

- desbalanceamento entre as classes;
- valores ausentes;
- registros duplicados;
- valores inconsistentes;
- outliers relevantes.

Após o tratamento dos dados, foi criada a variável `comprometimento_renda`, representando a relação percentual entre o valor do empréstimo e a renda do cliente.

As variáveis categóricas foram transformadas utilizando One-Hot Encoding.

A base foi dividida em treino e teste utilizando estratificação, e o conjunto de treino foi balanceado por meio do SMOTE.

Para o KNN, foi utilizado `StandardScaler`.

Foram avaliadas diferentes configurações dos modelos KNN e Árvore de Decisão.

No KNN, o melhor resultado entre os valores testados foi:

```text
K = 9
```

com aproximadamente:

```text
89,36% de acurácia no teste
```

Na Árvore de Decisão, o melhor equilíbrio foi obtido com:

```text
max_depth = 7
```

com aproximadamente:

```text
90,53% de acurácia no teste
```

A árvore sem limite apresentou 100% de acurácia no treino e 87,97% no teste, evidenciando um caso de overfitting.

---

## ✅ 14. Veredito do Projeto

Entre os modelos avaliados neste projeto, a **Árvore de Decisão com `max_depth = 7`** apresentou o melhor equilíbrio entre desempenho e capacidade de generalização.

O modelo apresentou:

- acurácia de teste de aproximadamente **90,53%**;
- precision de **86%** para clientes inadimplentes;
- recall de **68%**;
- F1-score de **76%**;
- **163 Falsos Positivos**;
- **451 Falsos Negativos**.

Apesar do desempenho superior ao KNN nas configurações avaliadas, o recall da classe inadimplente mostra que ainda existem clientes de risco que não são identificados pelo modelo.

Em um cenário real, seria recomendável continuar o processo de otimização, testar outros algoritmos e ajustar métricas e hiperparâmetros com foco principalmente na redução dos Falsos Negativos.

---

## 🛠️ 15. Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## 📁 16. Estrutura do Projeto

```text
projeto-risco-credito/
├── data/
│   └── credit_risk_dataset.csv
├── projeto_risco_credito.ipynb
└── README.md
```

---

## 🌿 17. Versionamento

O desenvolvimento foi realizado utilizando Git e GitHub, com branches separadas para diferentes etapas do projeto.

Branches utilizadas:

```text
main
fase/eda
fase/data-prep
```

Também foram realizados commits incrementais ao longo do desenvolvimento, evitando concentrar todo o projeto em um único commit.

---

## 👨‍💻 Autor

**Alisson Rafael Silva**

Projeto desenvolvido como parte do módulo de **Machine Learning e Visão Computacional**.
