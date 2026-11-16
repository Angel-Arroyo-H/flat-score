# flat-score
Transformação numérica para converter probabilidades preditas em scores com uma Distribuição Uniforme contínua.

## 🎯 O Problema e a Solução

Pelo **Teorema da Transformação Integral da Probabilidade**, se $X$ tem uma função de distribuição acumulada contínua $F_X$, então $F_X(X)$ tem distribuição $\mathcal{U}(0, 1)$. Na prática, como a distribuição é estimada, o resultado do pacote é uma aproximação que depende da qualidade do ajuste.

Em modelos preditivos de eventos raros (fraude, *churn* extremo, falhas mecânicas), as probabilidades estimadas (`predict_proba`) podem se concentrar em valores muito próximos de zero. Modelar diretamente nessa escala pode ser sensível à representação numérica e ao comportamento das caudas.

Este pacote resolve o problema em três etapas:
1. **Esticamento Logit:** Mudamos o espaço de probabilidade (0, 1) → (-∞, +∞) usando a transformação logit `z = ln(p/(1-p))` com proteção contra extremos (`clip`), espalhando os dados aglomerados.
2. **Mistura de Gaussianas (GMM) com Seleção via BIC:** Ajusta automaticamente o número ideal de componentes normais no espaço ***logit*** minimizando o *Bayesian Information Criterion*.
3. **FDA da Mistura:** Calcula a soma ponderada das FDAs normais de cada componente, produzindo scores entre 0 e 1 que tendem a ser uniformes quando a mistura representa bem os dados.

---

## 📦 Instalação

Clone o repositório e instale o pacote localmente:

```bash
git clone https://github.com/Angel-Arroyo-H/flat-score.git
cd flat-score

# Instalação padrão
pip install .

# Ou em modo editável com dependências de desenvolvimento (pytest)
pip install -e ".[dev]"
```

---

## 🚀 Uso Rápido

A classe `AutoLogitGMMUniformizer` segue a API padrão do `scikit-learn` (`fit` / `transform`), evitando vazamento de dados (*data leakage*) entre treino e teste.

Forneça um vetor de probabilidades de uma única classe. Para a saída de `predict_proba`, selecione a coluna da classe de interesse (por exemplo, `predict_proba(X)[:, 1]`).

```python
import numpy as np
from sklearn.model_selection import train_test_split
from uniformizer import AutoLogitGMMUniformizer

# 1. Simulando scores de um fenômeno raro (concentrados perto de 0)
np.random.seed(42)
scores = np.concatenate([
    np.random.beta(0.5, 500, size=8000),
    np.random.beta(2.0, 50, size=2000)
])

p_train, p_test = train_test_split(scores, test_size=0.3, random_state=42)

# 2. Treina no conjunto de treino escolhendo até 8 componentes via BIC
uniformizer = AutoLogitGMMUniformizer(max_components=8)
uniformizer.fit(p_train)

# 3. Transforma os dados de teste para Uniforme(0, 1)
u_test = uniformizer.transform(p_test)

print(f"Componentes selecionados: {uniformizer.best_gmm_.n_components}")
print(f"Quantis (10%, 50%, 90%): {np.quantile(u_test, [0.1, 0.5, 0.9]).round(3)}")
```

---

## 🧪 Validação e Testes

Para rodar a suíte de testes unitários:

```bash
pytest
```

Os testes cobrem o ajuste, a transformação e a validação dos valores de entrada.

## 📄 Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
