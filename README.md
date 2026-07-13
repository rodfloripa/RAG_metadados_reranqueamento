# RAG_metadados_reranqueamento
RAG com metadados nos documentos e reranqueamento


# 1. RAG com FAISS, Sentence Transformers e OpenAI

<p align="justify">
Este projeto implementa um pipeline completo de <b>Retrieval-Augmented Generation (RAG)</b> utilizando <b>Sentence Transformers</b> para geração de embeddings, <b>FAISS</b> para indexação vetorial, filtros por metadados, reranking baseado em similaridade do cosseno e uma LLM da OpenAI para geração da resposta final. O objetivo é permitir consultas em documentos PDF de maneira semântica, recuperando apenas os trechos mais relevantes antes da etapa de geração da resposta.
Para rodar:

crie o .env

OPENAI_API_KEY=sk...
</p>

---

# 2. Arquitetura

```text
                  PDFs
                    │
                    ▼
          Extração de Texto
                    │
                    ▼
          Quebra em Sentenças
                    │
                    ▼
      Sentence Transformers
      (Embeddings 384 dimensões)
                    │
                    ▼
               FAISS Index
                    │
        Consulta do Usuário
                    │
                    ▼
         Embedding da Consulta
                    │
                    ▼
           Busca Vetorial
                    │
                    ▼
        Filtro por Metadados
                    │
                    ▼
              Reranking
                    │
                    ▼
          Contexto Consolidado
                    │
                    ▼
             OpenAI GPT
                    │
                    ▼
           Resposta Final
```

---

# 3. Objetivos

<p align="justify">
O sistema foi desenvolvido para demonstrar uma arquitetura moderna de RAG baseada em recuperação semântica. Em vez de enviar documentos completos para a LLM, apenas os trechos semanticamente mais relevantes são recuperados através do FAISS, refinados por um processo de reranking e utilizados como contexto para geração da resposta. Essa estratégia reduz custo computacional, melhora a precisão das respostas e diminui significativamente a ocorrência de alucinações.
</p>

---

# 4. Bibliotecas Utilizadas

```python
torch
transformers
sentence-transformers
faiss
numpy
scikit-learn
PyPDF2
nltk
openai
```

---

# 5. Etapas da Solução

## 5.1 Download do Tokenizador

<p align="justify">
O NLTK é utilizado para segmentar automaticamente o texto dos documentos em sentenças. Essa abordagem permite granularidade maior durante a indexação, melhorando a recuperação semântica.
</p>

```python
nltk.download("punkt")
```

---

## 5.2 Carregamento do Modelo

<p align="justify">
O modelo escolhido foi o <b>all-MiniLM-L6-v2</b>, amplamente utilizado em aplicações de busca semântica devido ao excelente equilíbrio entre qualidade dos embeddings e velocidade de inferência. Cada sentença é representada por um vetor de 384 dimensões.
</p>

```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)

dimension = 384

index = IndexFlatL2(dimension)
```

---

## 5.3 Leitura dos PDFs

<p align="justify">
Cada documento PDF é percorrido página por página utilizando o PyPDF2. Todo o conteúdo textual é concatenado em uma única string para posterior processamento.
</p>

```python
def read_pdf(file_path):
```

<p align="justify">
Caso algum arquivo não exista, o sistema apenas informa o problema e continua a execução, evitando interrupções desnecessárias do pipeline.
</p>

---

## 5.4 Geração dos Embeddings

<p align="justify">
Cada sentença é transformada em um vetor numérico através do Transformer. Neste projeto foi utilizado <b>Mean Pooling</b>, que calcula a média das representações produzidas pelo último estado oculto da rede neural.
</p>

```python
outputs.last_hidden_state.mean(dim=1)
```

<p align="justify">
O resultado é um vetor que representa semanticamente cada sentença, permitindo comparar significados utilizando distância vetorial.
</p>

---

## 5.5 Segmentação em Sentenças

<p align="justify">
Após extrair o texto, o documento é dividido automaticamente em sentenças utilizando o NLTK.
</p>

```python
sentences = sent_tokenize(text)
```

<p align="justify">
Cada sentença passa a ser uma unidade independente dentro do índice vetorial, permitindo recuperar apenas pequenos trechos relevantes em vez de documentos completos.
</p>

---

## 5.6 Metadados

<p align="justify">
Durante a indexação são adicionados metadados para cada sentença. Esses metadados permitem realizar filtros posteriores sem necessidade de criar índices separados.
</p>

Exemplo:

```python
metadata = {
    "fonte": file_path,
    "autor": "...",
    "categoria": "..."
}
```

<p align="justify">
Os metadados podem representar autor, categoria, data, idioma, empresa, departamento ou qualquer informação útil para restringir a busca.
</p>

---

## 5.7 Construção do Índice Vetorial

<p align="justify">
Após gerar todos os embeddings, eles são convertidos para float32 e adicionados ao índice FAISS.
</p>

```python
embeddings_array = np.vstack(all_embeddings).astype("float32")

index.add(embeddings_array)
```

<p align="justify">
A partir desse momento torna-se possível executar buscas aproximadas extremamente rápidas mesmo em grandes volumes de documentos.
</p>

---

# 6. Busca Semântica

<p align="justify">
Quando uma pergunta é enviada pelo usuário, ela também é convertida em embedding utilizando exatamente o mesmo modelo empregado durante a indexação.
</p>

```python
query_embedding = get_embedding(query)
```

<p align="justify">
O FAISS procura os vetores mais próximos utilizando distância L2.
</p>

```python
scores, indices = index.search(query_embedding, top_k)
```

<p align="justify">
Os índices retornados apontam diretamente para as sentenças mais relevantes armazenadas no sistema.
</p>

---

# 7. Filtro por Metadados

<p align="justify">
Após recuperar os candidatos, o sistema realiza um filtro utilizando os metadados associados a cada sentença.
</p>

```python
if categoria and item["metadata"]["categoria"] != categoria:
    continue
```

<p align="justify">
Esse mecanismo permite restringir consultas para documentos de uma categoria específica, reduzindo ruídos e aumentando a precisão da recuperação.
</p>

---

# 8. Reranking

<p align="justify">
Embora o FAISS recupere rapidamente os documentos mais próximos, os resultados podem ser refinados por um segundo estágio. Neste projeto é utilizada similaridade do cosseno para reordenar os candidatos.
</p>

```python
cosine_similarity(query_emb, doc_emb)
```

<p align="justify">
Esse reranking melhora significativamente a qualidade do contexto enviado à LLM, aumentando a probabilidade de que os primeiros trechos realmente contenham a resposta correta.
</p>

---

# 9. Construção do Contexto

<p align="justify">
Após o reranking, os melhores trechos são concatenados formando um único contexto textual.
</p>

```python
contexto_unificado
```

<p align="justify">
Somente essas informações são enviadas ao modelo generativo.
</p>

---

# 10. Prompt Engineering

<p align="justify">
O sistema utiliza um prompt cuidadosamente elaborado para impedir que a LLM invente respostas. Ela deve responder exclusivamente utilizando as informações presentes no contexto recuperado.
</p>

```text
Responda utilizando APENAS o contexto fornecido.

Caso a informação não esteja presente,
informe que ela não consta nos documentos.
```

---

# 11. Geração da Resposta

<p align="justify">
A geração final utiliza a API Chat Completion da OpenAI.
</p>

```python
client.chat.completions.create(...)
```

<p align="justify">
Como apenas poucos trechos relevantes são enviados, o consumo de tokens é reduzido, a latência diminui e a qualidade das respostas tende a aumentar.
</p>

---

# 12. Fluxo Completo

```text
PDF
 │
 ▼
Extração de Texto
 │
 ▼
Tokenização em Sentenças
 │
 ▼
Embeddings
 │
 ▼
FAISS
 │
 ▼
Pergunta
 │
 ▼
Embedding da Pergunta
 │
 ▼
Busca Vetorial
 │
 ▼
Filtro por Categoria
 │
 ▼
Reranking
 │
 ▼
Contexto
 │
 ▼
OpenAI GPT
 │
 ▼
Resposta
```

---

# 13. Vantagens da Arquitetura

<p align="justify">
A utilização de Sentence Transformers juntamente com o FAISS permite buscas extremamente rápidas mesmo em bases documentais muito grandes. O filtro por metadados reduz significativamente o espaço de busca, enquanto o reranking melhora a qualidade dos resultados recuperados. Por fim, a integração com uma LLM possibilita produzir respostas em linguagem natural utilizando exclusivamente informações presentes nos documentos indexados, tornando o sistema mais confiável, interpretável e eficiente.
</p>

---

# 14. Aplicações

<p align="justify">
Essa arquitetura pode ser empregada em assistentes corporativos, sistemas jurídicos, pesquisa científica, mineração de documentos técnicos, consulta a normas regulatórias, bases acadêmicas, prontuários eletrônicos, centrais de conhecimento empresarial, documentação de software e mecanismos inteligentes de busca. A combinação entre recuperação vetorial e modelos generativos permite transformar grandes coleções de documentos em sistemas capazes de responder perguntas complexas de forma rápida, contextualizada e baseada em evidências.
</p>

---

# 15. Conclusão

<p align="justify">
Este projeto demonstra uma implementação completa de um pipeline moderno de Retrieval-Augmented Generation (RAG), integrando extração de texto, geração de embeddings, indexação vetorial com FAISS, recuperação semântica, filtragem por metadados, reranking e geração de respostas por meio de uma LLM. A arquitetura reduz a quantidade de informações enviadas ao modelo generativo, melhora a precisão das respostas e diminui o risco de alucinações. Essa abordagem representa uma solução escalável para sistemas inteligentes de consulta documental, podendo ser adaptada para diferentes domínios e grandes volumes de informação com elevado desempenho e baixo custo computacional.
</p>


<p align="center">
  <img src="https://github.com/rodfloripa/RAG_metadados_reranqueamento/blob/main/fig1.png">
</p>
