# Classificação de Pneumonia em Radiografias com InceptionNet

##  Visão Geral

Este projeto tem como objetivo desenvolver uma rede neural convolucional baseada na arquitetura **InceptionNet (GoogLeNet)** para classificar radiografias de tórax como **normais** ou com **pneumonia**. O modelo foi treinado com o dataset **Chest X-Ray Pneumonia**, com foco em aplicações práticas na área médica.

---

## Dataset: Chest X-Ray Pneumonia

* Imagens de radiografias de tórax de pacientes pediátricos, divididas em:

  * **NORMAL**: sem sinais de pneumonia
  * **PNEUMONIA**: com sinais visíveis da doença
* Estrutura:

/chest\_xray/
├── train/
├── val/
└── test/

- Distribuição dos dados:
  - **Treino**: 1.341 (NORMAL), 3.875 (PNEUMONIA)
  - **Validação**: 8 (NORMAL), 8 (PNEUMONIA)
  - **Teste**: 234 (NORMAL), 390 (PNEUMONIA)
- As imagens estão em **escala de cinza**, mas foram convertidas para 3 canais RGB para compatibilidade com o modelo.

---

## Arquitetura Utilizada: InceptionNet (GoogLeNet)
A InceptionNet utiliza módulos Inception, que aplicam vários filtros convolucionais em paralelo (1x1, 3x3, 5x5 e max pooling), permitindo a extração de padrões em diferentes escalas.

### Benefícios:
- Captura padrões complexos com menos parâmetros
- Boa generalização mesmo com datasets menores
- Ideal para imagens médicas com variação local e global

---

## Etapas do Projeto

### 1. **Pré-processamento**
- Redimensionamento das imagens para **299x299** pixels
- Normalização dos valores dos pixels (0 a 1)
- Data augmentation: rotação, zoom, flips

### 2. **Criação do Modelo**
- Carregamento da InceptionNet com pesos do ImageNet
- Congelamento das camadas convolucionais
- Adição de:
  - GlobalAveragePooling
  - Dropout (0.5)
  - Camada final densa com ativação sigmoid

### 3. **Treinamento**
- Otimizador Adam (lr = 0.0001)
- 10 épocas com validação em cada ciclo

### 4. **Avaliação e Métricas**
- **Acurácia**: proporção total de acertos
- **Sensibilidade (Recall)**: capacidade de detectar pneumonia
- **Especificidade**: capacidade de identificar casos normais corretamente
- **Curva ROC e AUC**: avalia performance geral do classificador

---

## Código

### Importações
Importa bibliotecas para manipulação de dados, imagens, redes neurais e métricas.

### Carregamento dos dados
Utiliza `ImageDataGenerator` para criar os conjuntos de treino, validação e teste, aplicando data augmentation no treino.

### Modelo InceptionNet
1. `InceptionV3(weights='imagenet', include_top=False)`: usa a parte convolucional do modelo pré-treinado.
2. Congela as camadas para transfer learning.
3. Adiciona novas camadas para classificação binária:
   - GlobalAveragePooling2D
   - Dropout
   - Dense com sigmoid

### Compilação
- Função de perda: `binary_crossentropy`
- Métrica: `accuracy`

### Treinamento
Executado por 10 épocas usando os dados de treino e validação.

### Avaliação
1. Predição das classes no teste
2. Geração da matriz de confusão
3. Cálculo de sensibilidade, especificidade e acurácia
4. Geração da curva ROC e cálculo da AUC

### Visualizações
- Curvas de acurácia e perda por época
- Curva ROC
