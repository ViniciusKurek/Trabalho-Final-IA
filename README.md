# Trabalho Final
Este trabalho propõe a utilização de modelos de inteligência artificial para classificação de imagens.

# Dataset
O dataset escolhido foi o dos `Simpsons`. A ideia é treinar um modelo, e depois o aplicar em algumas imagens de teste para analisar a taxa de acerto do modelo.

# Como executar
- Primeiro descompacte o arquivo `simpsons.tar.gz` e o deixe na pasta raiz do projeto;
- Inicie um ambiente virtual python com `python -m venv env`;
- Ative o ambiente virtual do python com `source env/bin/activate`;
- Instale as bibliotecas necessárias com `pip install -r requirements.txt`;
- Execute os scripts na ordem numérica deles.

O script 1 gerará as features no arquivo `features.joblib`;

O script 2 gerará os melhores modelos encontrados com as features na pasta `ensemble`. Esse processo é meio demorado, se já houver arquivos salvos, poderão ser reaproveitados e essa etapa pode ser pulada;

O script 3 fará a combinação dos modelos.

O script 4 imprime a matriz de confusão do melhor MLP encontrado;

# Pastas

`ensemble`: Modelos que serão usados para combinação;
`resultados`: Resultados obtidos do tuning e ensemble;
`docs`: Relatório final e slides;

# Alunos
Eduardo Kurek && Vinicius Kurek - UTFPR-cm, Curso de Ciência da Computação, disciplina de Inteligência Computacional 2025/2
