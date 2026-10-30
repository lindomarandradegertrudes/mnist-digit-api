# API de Classificação de Dígitos Manuscritos — MNIST

## 1. Sobre o projeto

Este projeto consiste no desenvolvimento de uma API REST para classificação de dígitos manuscritos utilizando Machine Learning. A aplicação foi desenvolvida com Python e FastAPI e utiliza um modelo de rede neural treinado com o conjunto de dados MNIST.

A API recebe uma imagem representada por um vetor de pixels, realiza a previsão com o modelo treinado e retorna o dígito classificado, o nível de confiança da previsão e as probabilidades associadas às classes.

O projeto também inclui validações de entrada, tratamento de erros HTTP e testes automatizados para verificar o comportamento dos endpoints.

## 2. Objetivos

- Desenvolver uma API para disponibilizar um modelo de classificação de dígitos manuscritos.
- Receber imagens representadas por vetores de pixels no formato JSON.
- Retornar a classe prevista, a confiança e as probabilidades de classificação.
- Validar os dados recebidos e tratar situações de erro.
- Testar os endpoints utilizando `pytest`.

## 3. Tecnologias utilizadas

As tecnologias, versões e ferramentas utilizadas no projeto estão descritas na tabela abaixo. As dependências Python também estão registradas no arquivo `requirements.txt`.

| Tecnologia | Versão | Finalidade |
|---|---|---|
| Python | 3.14 | Linguagem utilizada no desenvolvimento. |
| FastAPI | 0.143.3 | Criação dos endpoints da API REST. |
| Uvicorn | 0.54.0 | Servidor ASGI utilizado para executar a aplicação. |
| Pydantic | 2.14.1 | Validação e estruturação dos dados de entrada e saída. |
| scikit-learn | 1.9.1 | Ferramentas de Machine Learning utilizadas no projeto. |
| NumPy | 2.5.4 | Manipulação de vetores e dados numéricos. |
| joblib | 1.6.0 | Serialização e carregamento do modelo treinado. |
| pytest | 9.1.1 | Execução dos testes automatizados. |
| HTTPX | 0.28.1 | Suporte a testes de requisições HTTP. |
| MNIST | — | Conjunto de dados de dígitos manuscritos utilizado no treinamento. |

## 4. Origem e treinamento do modelo

O modelo utilizado pela API é uma rede neural do tipo **Multilayer Perceptron (MLP)**, treinada com o conjunto de dados MNIST.

O processo de treinamento e serialização está documentado no notebook `notebooks/01_treino_serializacao.ipynb`.

O modelo alcançou **97,8% de acurácia no conjunto de teste**, conforme a avaliação registrada no projeto. Após o treinamento, foi serializado com `joblib`, permitindo seu carregamento pela aplicação sem a necessidade de realizar o treinamento novamente a cada inicialização.

O arquivo serializado está localizado em:

```text
models/mnist_mlp.joblib
```

A aplicação utiliza esse arquivo para realizar as previsões solicitadas pelo endpoint `/predict`.

A acurácia informada corresponde ao resultado obtido na avaliação do modelo no conjunto de teste e não representa uma garantia de acerto para todas as imagens enviadas à API.

## 5. Estrutura do projeto

```text
mnist-digit-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routers.py
│   ├── schemas.py
│   └── services.py
├── models/
│   └── mnist_mlp.joblib
├── notebooks/
│   └── 01_treino_serializacao.ipynb
├── reports/
│   └── metrics.json
├── examples/
│   ├── digito_0.json
│   ├── digito_3.json
│   └── digito_7.json
├── tests/
│   └── test_api.py
├── requirements.txt
├── README.md
└── .gitignore
```

Descrição dos principais diretórios e arquivos:

- `app/main.py`: cria a aplicação FastAPI e carrega o modelo na inicialização (`lifespan`).
- `app/routers.py`: rotas `GET /health` e `POST /predict`.
- `app/schemas.py`: schemas Pydantic de entrada e saída.
- `app/services.py`: carga do modelo e função de previsão.
- `models/mnist_mlp.joblib`: modelo MLP treinado e serializado.
- `notebooks/`: notebook utilizado no treinamento e na serialização do modelo.
- `reports/metrics.json`: métricas e hiperparâmetros do modelo.
- `examples/`: três imagens de exemplo (dígitos 0, 3 e 7) para testar a API.
- `tests/test_api.py`: testes automatizados da API.
- `requirements.txt`: dependências e versões necessárias para executar o projeto.
- `README.md`: documentação principal do projeto.

## 6. Instalação e configuração

### 6.1. Clonar o repositório

```bash
git clone https://github.com/lindomarandradegertrudes/mnist-digit-api.git
cd mnist-digit-api
```

### 6.2. Criar um ambiente virtual

```bash
python -m venv .venv
```

Ative o ambiente virtual de acordo com seu sistema operacional.

**Windows — PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux ou macOS**

```bash
source .venv/bin/activate
```

### 6.3. Instalar as dependências

Com o ambiente virtual ativado, execute:

```bash
python -m pip install -r requirements.txt
```

Esse comando instala as dependências declaradas no `requirements.txt`, respeitando as versões especificadas no arquivo.

## 7. Executar a aplicação

Na raiz do projeto, execute:

```bash
python -m uvicorn app.main:app
```

Para habilitar o recarregamento automático durante o desenvolvimento, utilize:

```bash
python -m uvicorn app.main:app --reload
```

Depois que o servidor iniciar, a API estará disponível localmente nos seguintes endereços:

- **API:** http://127.0.0.1:8000
- **Documentação interativa (Swagger):** http://127.0.0.1:8000/docs
- **Documentação alternativa (ReDoc):** http://127.0.0.1:8000/redoc

A documentação interativa permite consultar os endpoints, visualizar os modelos de dados e enviar requisições para testar a classificação.

## 8. Endpoints da API

### 8.1. Verificar a saúde da aplicação

**Método:** `GET`

**Rota:** `/health`

Verifica se a API está respondendo e permite consultar o estado da aplicação e do carregamento do modelo na memória (RAM).

Exemplo de requisição:

```bash
curl http://127.0.0.1:8000/health
```

Exemplo de resposta:

```json
{
  "status": "ok",
  "modelo_carregado": true
}
```

### 8.2. Classificar um dígito manuscrito

**Método:** `POST`

**Rota:** `/predict`

Recebe um vetor de pixels no formato JSON e utiliza o modelo MLP para classificar a imagem.

A entrada esperada contém **784 valores numéricos**, correspondentes a uma imagem de 28 × 28 pixels. Cada valor representa a intensidade de um pixel entre 0 e 255.

A aplicação normaliza os valores dos pixels antes de realizar a previsão, dividindo-os por 255.

#### Exemplo de requisição

O exemplo abaixo utiliza o arquivo `examples/digito_3.json`.

**Windows — PowerShell ou terminal compatível com `curl.exe`:**

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@examples/digito_3.json"
```

A requisição deve enviar o conteúdo do arquivo JSON no corpo da solicitação.

Em Linux ou macOS, use `curl` no lugar de `curl.exe`.

#### Exemplo de resposta

O exemplo abaixo ilustra uma previsão para o dígito 3, com as probabilidades arredondadas.

```json
{
  "digito": 3,
  "confianca": 0.9963,
  "probabilidades": [
    0.0,
    0.0,
    0.0,
    0.9963,
    0.0,
    0.0037,
    0.0,
    0.0,
    0.0,
    0.0
  ]
}
```

Os índices do vetor `probabilidades` correspondem aos dígitos de 0 a 9. Por exemplo, o índice 3 representa a probabilidade atribuída ao dígito 3.

A confiança e as probabilidades variam de acordo com a imagem enviada.

## 9. Tratamento de erros HTTP

A API utiliza códigos de status HTTP para indicar o resultado do processamento das requisições.

| Código HTTP | Significado | Situação |
|---|---|---|
| `200 OK` | Requisição processada com sucesso. | A previsão foi realizada ou a consulta de saúde foi atendida. |
| `400 Bad Request` | Requisição inválida para a regra de negócio. | Imagem vazia, representada por um vetor composto apenas por zeros. |
| `422 Unprocessable Entity` | Dados de entrada inválidos. | Vetor com quantidade incorreta de valores ou dados incompatíveis com o esquema esperado. |
| `500 Internal Server Error` | Erro interno da aplicação. | Falha inesperada durante o processamento da previsão. |
| `503 Service Unavailable` | Serviço temporariamente indisponível. | Modelo não carregado ou indisponível para realizar previsões. |

## 10. Testes automatizados

O projeto possui testes automatizados para verificar o comportamento da API, incluindo suas requisições e respostas.

Os testes utilizam `pytest` e o `TestClient` do FastAPI (baseado em HTTPX).

### 10.1. Executar os testes

Na raiz do projeto, com o ambiente virtual ativado, execute:

```bash
python -m pytest
```

Para visualizar os nomes dos testes executados e seus resultados, utilize:

```bash
python -m pytest -v
```

O resultado permite verificar quais testes passaram e identificar eventuais falhas durante a execução.

## 11. Exemplos de entrada

O diretório `examples/` contém arquivos JSON que podem ser utilizados como entrada para testar o endpoint `/predict`.

Arquivos disponíveis:

- `examples/digito_0.json`: imagem de um dígito 0.
- `examples/digito_3.json`: imagem de um dígito 3.
- `examples/digito_7.json`: imagem de um dígito 7.

As três imagens vêm do conjunto de teste do MNIST, que o modelo não viu durante o treino.

Cada arquivo de entrada deve respeitar o formato esperado pela API, com 784 valores numéricos correspondentes aos pixels de uma imagem de 28 × 28.

## 12. Vídeo de apresentação do projeto

Veja abaixo o vídeo de apresentação do projeto, demonstrando a execução da API, o acesso à documentação interativa e um exemplo de classificação de dígito manuscrito.

**Link do vídeo:** https://drive.google.com/file/d/1iU9NKHT_jrvgKx96y-xXAULWPkDTihR2/view?usp=drive_link

## 13. Considerações finais

Este projeto demonstra a integração entre Machine Learning e desenvolvimento de APIs REST, disponibilizando um modelo de classificação de dígitos manuscritos por meio de uma interface HTTP.

Além da utilização de um modelo MLP treinado com o conjunto de dados MNIST, a aplicação contempla validação de dados, tratamento de erros, documentação interativa e testes automatizados.

A estrutura permite que o modelo treinado seja utilizado por outras aplicações sem a necessidade de executar novamente o processo de treinamento a cada requisição.

## 14. Repositório

O código-fonte e os arquivos do projeto estão disponíveis no GitHub:

https://github.com/lindomarandradegertrudes/mnist-digit-api
