# RAG 코드 실습 정답·해설지

이 문서는 생성된 시험 대비 실습 노트북의 `## 정답 입력` 셀에 대응한다.
정답을 바로 복사하기보다 먼저 입력·실행하고, 실패 원인을 기록한 뒤 비교한다.

## 출제 포인트 기준

- RAG 구성요소와 `retrieve → augment → generate` 흐름
- LlamaIndex, MCP 등 관련 라이브러리의 역할과 연결
- Retriever/Reader/KG/MCP tool 호출의 입력·출력 구조

## RAG Practice 01. LlamaIndex Query Engine 코드 학습

원본: `rag/1일차/실습 자료/Code/1. Llama_index.ipynb`
실습본: `practice_notebooks/rag/day1/01-llama-index.ipynb`

### Drill 1 — Answer: **15**

원본 Cell `041`. 이 셀은 **Answer: **15**** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

def generate_answer(question):
    messages = [
        {
            "role": "user",
            "content": question,
        },
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=messages,
    )

    return response.choices[0].message.content
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 1. Defining the query engine in detail

원본 Cell `075`. 이 셀은 **1. Defining the query engine in detail** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class StandardQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 1. Defining the query engine in detail

원본 Cell `077`. 이 셀은 **1. Defining the query engine in detail** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

retriever = index.as_retriever()
synthesizer = get_response_synthesizer(response_mode="compact")
query_engine = StandardQueryEngine(
    retriever=retriever, response_synthesizer=synthesizer
)

response = query_engine.query("What did the author do growing up?")
print(str(response))
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 2. Creating a custom query engine

원본 Cell `080`. 이 셀은 **2. Creating a custom query engine** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

from llama_index.llms.openai import OpenAI

class OurCustomQueryEngine(CustomQueryEngine):

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate = simple_qa_prompt

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — 2. Creating a custom query engine

원본 Cell `082`. 이 셀은 **2. Creating a custom query engine** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

llm = OpenAI(model="gpt-3.5-turbo")

query_engine = OurCustomQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=simple_qa_prompt,
)

response = query_engine.query("What did the author do growing up?")
print(str(response))
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## RAG Practice 02. RAG App 구성 코드 학습

원본: `rag/1일차/실습 자료/Code/2. RAG.ipynb`
실습본: `practice_notebooks/rag/day1/02-rag-app.ipynb`

### Drill 1 — 3. Connect Retriever and Generator

원본 Cell `038`. 이 셀은 **3. Connect Retriever and Generator** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class RAG_from_scratch:
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = query_engine.query(query)
        return results

    def generate_response(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = oai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=
        [
            {"role": "user",
            "content":
            f"We have provided context information below. \n"
            f"---------------------\n"
            f"{context_str}"
            f"\n---------------------\n"
            f"Given this information, please answer the question: {query}"
            }
        ]
        ).choices[0].message.content
        return completion

    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_response(query, context_str)
        return completion

rag = RAG_from_scratch()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki

원본 Cell `055`. 이 셀은 **3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

simple_qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

short_sum_prompt = PromptTemplate(
"""Write a summary of the following. Try to use only the information provided.
Try to include as many key details as possible.
---------------------\n
{context_str}
---------------------\n
SUMMARY:"""
)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki

원본 Cell `056`. 이 셀은 **3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class OurCustomQueryEngine(CustomQueryEngine):

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAI
    qa_prompt: PromptTemplate = simple_qa_prompt

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)

llm = OpenAI(model="gpt-3.5-turbo")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki

원본 Cell `058`. 이 셀은 **3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

retriever = index.as_retriever()
synthesizer = get_response_synthesizer(response_mode="compact")

query_engine_answer = OurCustomQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=simple_qa_prompt,
)

res_answer = query_engine_answer.query("What's the arts and culture scene in Berlin?")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — 3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki

원본 Cell `064`. 이 셀은 **3. Prompt Design for RAG: Add clear instructions to the task so that the LLM generates the answer that the user is looki** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class Refine_RAG:
    def retrieve(self, query: str) -> list:
        ret = retriever.retrieve(query)
        results = query_engine.query(query)
        return ret, results

    def generate_response(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Answer as concisely as possible.",
            },
            {
                "role": "user",
                "content":
                    f"""
                    ###Instruction
Please answer the following question based on the provided context. Your answer should be short and concise.
Basically, you have to answer the question based on the provided context. But you can use your parametrized knowledge when the provided context was wrong or unrelated to the quesion.
When you generate the answer, you should explain the reason why you deduced such a answer from the context.

### Question
{query}

### Provided Context
{context_str}

### Answer


### Reason
                    """
            }
        ]

        response = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=messages,
        )

        return response.choices[0].message.content

    def query(self, query: str) -> str:
        ret, context_str = self.retrieve(query)
        # for i in range(len(ret)):
        #   print("Retrieved Context: \n", ret[i].text) # use only when you want to see intermeidate result
        # print("\n\nIntermediate Summary: \n",  context_str) # use only when you want to see intermeidate result
        completion = self.generate_response(query, context_str)
        return completion

refine_rag = Refine_RAG()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## RAG Day 2 Practice 01. MCP 기반 평가와 업그레이드

원본: `rag/2일차/실습 자료/Code/4_RAG_framework_evaluation_with_MCP.ipynb`
실습본: `practice_notebooks/rag/day2/01_mcp_evaluation.ipynb`

### Drill 1 — 2. How to evalute RAG following new evaluation metrics?

원본 Cell `017`. 이 셀은 **2. How to evalute RAG following new evaluation metrics?** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

def parse_response(response):
    try:
        response = response.lower()
        model_resp = json.loads(response)
        answer = -1
        if "accuracy" in model_resp and (
            (
              model_resp["accuracy"] is True
            )
            or
            (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1

test_text = """{
    "Accuracy": "True"
}"""

print(parse_response(test_text))
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 2. How to evalute RAG following new evaluation metrics?

원본 Cell `019`. 이 셀은 **2. How to evalute RAG following new evaluation metrics?** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

def CRAG_evaluation(question, ground_truth, prediction):
  context_template = f"""Question: {question}
  List of Ground Truth answers: {ground_truth}
  Model Prediction: {prediction}
  """

  evaluation_result = generate_answer(user_prompt=context_template, system_prompt=INSTRUCTIONS)

  eval_res = parse_response(evaluation_result)

  return eval_res
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 1. Define RAG

원본 Cell `022`. 이 셀은 **1. Define RAG** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

external_kg_server = "http://10.2.0.165:8000"   #need to change

class RAG:
    def __init__(self, server=None):
        self.retriever = LlamaIndexRetriever()
        self.kg_query_engine = KGQueryEngine(server=server)
        self.reader = Reader()

    def retrieve(self, query, search_results, topk):
        retrieved_results = self.retriever.retrieve(query, search_results, topk)

        kg_results = self.kg_query_engine.query(query)

        combined_results = [kg_results]
        combined_results.extend(retrieved_results)

        return combined_results

    def generate_response(self, query, retrieved_results):
        answer = self.reader.generate_response(query, retrieved_results)
        return answer

    def inference(self, query, search_results, topk):
        retrieved_results = self.retrieve(query, search_results, topk)
        answer = self.generate_response(query, retrieved_results)
        return {
            "retrieved_results": retrieved_results,
            "answer": answer
        }

rag = RAG(server=external_kg_server)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 1. KG Query Engine Initialization with MCP Client

원본 Cell `046`. 이 셀은 **1. KG Query Engine Initialization with MCP Client** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

from llama_index.llms.openai import OpenAI

class KGQueryEngineWithMCP:
    def __init__(self, mcp_tool_spec: McpToolSpec, model: str, llm = None):
        self.llm = llm or OpenAI(model=model, temperature=0)
        self.mcp_tool_spec = mcp_tool_spec
        self.agent: Optional[FunctionAgent] = None
        self.agent_context: Optional[Context] = None

    async def init_agent(self):
        tools = await self.mcp_tool_spec.to_tool_list_async()
        for tool in tools:
            if len(tool.metadata.description) > 1000:
                tool.metadata.description = tool.metadata.description[:1000] + "..."
        self.agent = FunctionAgent(
            name="Agent",
            description="An agent that can work with Our Knowledge Graph api.",
            tools=tools,
            llm=self.llm,
            system_prompt=SYSTEM_PROMPT,
        )
        self.agent_context = Context(self.agent)

    async def query(self, question: str, verbose: bool = False) -> str:
        if self.agent is None or self.agent_context is None:
            raise RuntimeError("Agent not initialized. Call `await init_agent()` first.")

        handler = self.agent.run(question, ctx=self.agent_context)
        async for event in handler.stream_events():
            if verbose and type(event) == ToolCall:
                print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
            elif verbose and type(event) == ToolCallResult:
                print(f"Tool {event.tool_name} returned {event.tool_output}")

        response = await handler
        return str(response)

kg_engine = KGQueryEngineWithMCP(mcp_tool, model='gpt-3.5-turbo')
await kg_engine.init_agent()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — Key Components

원본 Cell `055`. 이 셀은 **Key Components** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

from llama_index.llms.openai import OpenAI

class RAGwithMCP:
    def __init__(self, mcp_tool):
        self.mcp_application = KGQueryEngineWithMCP(mcp_tool, model='gpt-4o-mini')
        self.reader = Reader()

    async def retrieve(self, query: str, query_time: str, search_results: list, topk: int):
        await self.mcp_application.init_agent()

        full_query = f"""Query: {query} When inferring time information, rely solely on the query time. Do not infer the time scope from the query itself.
Query time: {query_time}"""

        mcp_result = await self.mcp_application.query(full_query, verbose=False)
        return mcp_result

    def generate_response(self, query: str, query_time: str, retrieved_results: str):
        answer = self.reader.generate_response(query, query_time, retrieved_results)
        return answer

    async def inference(self, query: str, search_results: list, query_time: str, topk: int):
        retrieved_results = await self.retrieve(query, query_time, search_results, topk)
        answer = self.generate_response(query, query_time, retrieved_results)
        return {
            "retrieved_results": retrieved_results,
            "answer": answer
        }

rag_mcp = RAGwithMCP(mcp_tool)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## RAG Day 2 Practice 02. CRAG 데이터 전처리

원본: `rag/2일차/실습 자료/Code/1. Data_preprocessing.ipynb`
실습본: `practice_notebooks/rag/day2/02_data_preprocessing.ipynb`

### Drill 1 — II. Check Dataset

원본 Cell `016`. 이 셀은 **II. Check Dataset** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

dataset = []

with bz2.open(file_path, 'rt') as file:
    for line in file:
        try:
            data = json.loads(line.strip())
            dataset.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — II. Check Dataset

원본 Cell `019`. 이 셀은 **II. Check Dataset** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

def generate_answer(question):
    messages = [
        {
            "role": "user",
            "content": question,
        },
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    return response.choices[0].message.content
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 1. Domain

원본 Cell `021`. 이 셀은 **1. Domain** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

unique_domains = {}
for item in dataset:
    if 'domain' in item:
        domain_value = item['domain']
        if domain_value not in unique_domains:
            unique_domains[domain_value] = item

print("Unique domains and their example items:\n")
for domain, example_item in unique_domains.items():
    print(f"Domain: {domain}")
    question = example_item['query']
    answer = example_item['answer']
    print(f"Example question: {question}")
    print(f"Example answer: {answer}\n")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 2. Understanding the reason why we need to parsing search result

원본 Cell `044`. 이 셀은 **2. Understanding the reason why we need to parsing search result** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

from bs4 import BeautifulSoup
from blingfire import text_to_sentences_and_offsets

all_chunks = []

for html_text in example_data['search_results']:
    soup = BeautifulSoup(html_text["page_result"], features="lxml")
    text = soup.get_text(" ", strip=True)
    if not text:
        all_chunks.append("")
    else:
        _, offsets = text_to_sentences_and_offsets(text)

        chunks = []

        for start, end in offsets:
            chunk = text[start:end][:4000]
            all_chunks.append(chunk)

print(all_chunks[:1])
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## RAG Day 2 Practice 03. Web Retriever와 Reader

원본: `rag/2일차/실습 자료/Code/2. Task_1.ipynb`
실습본: `practice_notebooks/rag/day2/03_web_rag.ipynb`

### Drill 1 — 2. Implementing a Chunk Extractor

원본 Cell `007`. 이 셀은 **2. Implementing a Chunk Extractor** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

def parse_htmls(search_results):
    all_documents = []

    # Process each HTML text from the search results to extract text content.
    for html_text in search_results:

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_text["page_result"], features="lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces
        all_documents.append(text)

    return all_documents

def extract_chunks(all_documents):
    # Initialize a list to hold all extracted sentences from the search results.
    all_chunks = []

    for document in all_documents:

        if not document:
            # If no document is extracted, add an empty string as a placeholder.
            all_chunks.append("")
        else:

            # Extract offsets of sentences from the document
            _, offsets = text_to_sentences_and_offsets(document)

            # Initialize a list to store sentences
            chunks = []

            # Iterate through the list of offsets and extract sentences
            for start, end in offsets:
                # Extract the sentence and limit its length
                chunk = document[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
                all_chunks.append(chunk)

    return all_chunks
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 3. Implementing a Retriever

원본 Cell `012`. 이 셀은 **3. Implementing a Retriever** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class BaseRetriever:
    def __init__(self,):
        self.client = openai.OpenAI(api_key = os.environ["OPENAI_API_KEY"])

    def embed_text(self, texts):
        """Generate embeddings using OpenAI's embedding model."""
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        # Extract embeddings correctly from the response object
        embeddings = [np.array(item.embedding) for item in response.data]  # Adjust based on actual attributes
        return np.array(embeddings)

    def retrieve(self, query, search_results, topk):
        # Get documents
        all_documents = parse_htmls(search_results)

        # Get chunks
        all_chunks = extract_chunks(all_documents)

        # Generate embeddings for all chunks and the query.
        all_embeddings = self.embed_text(all_chunks)
        query_embedding = self.embed_text(query)[0]  # Single query embedding

        # Calculate cosine similarity between query and sentence embeddings, and select the top sentences.
        cosine_scores = np.dot(all_embeddings, query_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = (-cosine_scores).argsort()[:topk]
        top_k_chunks = np.array(all_chunks)[top_k_indices]

        return top_k_chunks
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 3. Implementing a Retriever with Llama Index

원본 Cell `016`. 이 셀은 **3. Implementing a Retriever with Llama Index** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

class LlamaIndexRetriever:
  def __init__(self):
      self.parser = SentenceSplitter(chunk_size=512, chunk_overlap=0)

  def retrieve(self, query, search_results, topk):
      documents = []

      for document in parse_htmls(search_results):
        if not document:
            # If no text is extracted, add an empty string as a placeholder.
            documents.append(Document(text=""))
        else:
            documents.append(Document(text=document))

      # Split documents into chunks & Create vector index
      base_index = VectorStoreIndex.from_documents(documents = documents, transformations=[self.parser])

      # Execute query
      base_retriever = base_index.as_retriever(similarity_top_k=topk)

      retrieved_nodes = base_retriever.retrieve(query)

      retrieved_results = [retrieved_node.node.get_content().strip() for retrieved_node in retrieved_nodes]

      return retrieved_results
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 2. Implement a Prompt Generator

원본 Cell `023`. 이 셀은 **2. Implement a Prompt Generator** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

def prompt_generator(query, top_k_chunks, system_prompt):
    user_message = ""
    references = ""

    if len(top_k_chunks) > 0:
        references += "# References \n"
        # Format the top sentences as references in the model's prompt template.
        for chunk_id, chunk in enumerate(top_k_chunks):
            references += f"- {chunk.strip()}\n"

    references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
    # Limit the length of references to fit the model's input size.

    user_message += f"{references}\n------\n\n"
    user_message += f"Using only the references listed above, answer the following question: \n"
    user_message += f"Question: {query}\n"

    llm_input = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_message},
    ]

    return llm_input
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — 3. Implement a Reader

원본 Cell `025`. 이 셀은 **3. Implement a Reader** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

from openai import OpenAI

oai_client = OpenAI()

class Reader:
  def __init__(self):

    self.system_prompt = """
    You are provided with a question and various references.
    Your task is to answer the question succinctly, using the fewest words possible.
    If the references do not contain the necessary information to answer the question, respond with 'I don't know'.
    There is no need to explain the reasoning behind your answers.
    """

  def generate_response(self, query: str, top_k_chunks: list) -> str:
      """
      Generate answer from context.
      """
      llm_input = self.prompt_generator(query, top_k_chunks)
      completion = oai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      temperature=0,
      messages=
      llm_input
      ).choices[0].message.content
      return completion

  def prompt_generator(self, query, top_k_chunks):
      user_message = ""
      references = ""

      if len(top_k_chunks) > 0:
          references += "# References \n"
          # Format the top sentences as references in the model's prompt template.
          for chunk_id, chunk in enumerate(top_k_chunks):
              references += f"- {chunk.strip()}\n"

      references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
      # Limit the length of references to fit the model's input size.

      user_message += f"{references}\n------\n\n"
      user_message
      user_message += f"Using only the references listed above, answer the following question: \n"
      user_message += f"Question: {query}\n"

      llm_input = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_message},
      ]

      return llm_input
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 6 — III. Implementing a RAG

원본 Cell `029`. 이 셀은 **III. Implementing a RAG** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class RAG:
    def __init__(self):
        self.retriever = LlamaIndexRetriever()
        self.reader = Reader()

    def inference(self, query, search_results, topk):
        # 1. retrieve relevant chunks
        retrieved_results = self.retriever.retrieve(query, search_results, topk)

        # 2. answer the question based on the retrieved chunks
        answer = self.reader.generate_response(query, retrieved_results)

        return answer, retrieved_results
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## RAG Day 2 Practice 04. Knowledge Graph RAG

원본: `rag/2일차/실습 자료/Code/3. Task_2.ipynb`
실습본: `practice_notebooks/rag/day2/04_kg_rag.ipynb`

### Drill 1 — Designing prompts

원본 Cell `013`. 이 셀은 **Designing prompts** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

def prompt_generator(query):
    user_message = ""
    user_message += f"Query: {query}\n"

    llm_input = [
      {"role": "system", "content": entity_extract_template},
      {"role": "user", "content": user_message},
    ]

    return llm_input
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — Generating Queries

원본 Cell `015`. 이 셀은 **Generating Queries** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

import json
from openai import OpenAI
from json import JSONDecoder

oai_client = OpenAI()

def generate_query(query):
    llm_input = prompt_generator(query)
    completion = oai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=
    llm_input
    ).choices[0].message.content

    try:
        completion = json.loads(completion)
    except:
        completion = extract_json_objects(completion)

    if "domain" in completion.keys():
        domain = completion["domain"]
        is_finance = domain == "finance"
    else:
        is_finance = False

    return completion, is_finance

def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data
    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 4. Implementing a Query Executor

원본 Cell `021`. 이 셀은 **4. Implementing a Query Executor** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

import copy

def get_finance_kg_results(generated_query):
    formatted_time_list = []
    if 'datetime' in generated_query:
        datetime_list = generated_query['datetime'].split(' - ')
        for datetime in datetime_list:
            formatted_time_list.append(convert_to_standard_format(datetime.strip()))


    kg_results = []
    res = ""
    if "market_identifier" in generated_query.keys() and generated_query["market_identifier"] is not None:
        if isinstance(generated_query["market_identifier"], str):
            company_names = generated_query["market_identifier"].split(",")
        else:
            company_names = generated_query["market_identifier"]

        for company_name in company_names:
            try:
                res = api.finance_get_company_name(company_name)["result"]

                if res == []:
                    ticker_name = company_name.upper()
                else:
                    ticker_name = api.finance_get_ticker_by_name(res[0])["result"]

                if generated_query['metric'].lower().strip() == 'price':
                    response = api.finance_get_price_history(ticker_name)['result']
                elif generated_query['metric'].lower().strip() == 'dividend':
                    response = api.finance_get_dividends_history(ticker_name)['result']
                elif generated_query['metric'].lower().strip() == 'p/e ratio':
                    response = api.finance_get_pe_ratio(ticker_name)['result']
                elif generated_query['metric'].lower().strip() == 'eps':
                    response = api.finance_get_eps(ticker_name)["result"]
                elif generated_query['metric'].lower().strip() == 'marketcap' :
                    response = api.finance_get_market_capitalization(ticker_name)['result']
                else:
                    response = api.finance_get_info(ticker_name)['result']
                    metric_value = get_metric_from_response(response, generated_query['metric'])
                    if metric_value is not None:
                        response = metric_value

                try:
                    for formatted_time in formatted_time_list:
                        if formatted_time in response:
                            filtered_response = copy.deepcopy(response[formatted_time])
                        elif add_one_day(formatted_time) in response:
                            filtered_response = copy.deepcopy(response[add_one_day(formatted_time)])
                        elif subtract_one_day(formatted_time) in response:
                            filtered_response = copy.deepcopy(response[subtract_one_day(formatted_time)])
                        else:
                            filtered_response = copy.deepcopy(response)
                        kg_results.append({company_name + " " + generated_query["metric"]: filtered_response, 'time': formatted_time})
                except:
                    kg_results.append({company_name + " " + generated_query["metric"]: response})

            except Exception as e:
                print("Fail to parse the generated query")
                pass

    kg_results = "<DOC>\n".join([str(res) for res in kg_results]) if len(kg_results) > 0 else ""
    return  kg_results
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 5. Implementing a Mock KG Query Engine

원본 Cell `025`. 이 셀은 **5. Implementing a Mock KG Query Engine** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class KGQueryEngine:
    def query(self, query):
        generated_query, is_finance = self.generate_query(query)

        if is_finance:
            kg_results = self.get_finance_kg_results(generated_query)
        else:
            kg_results = ""

        return kg_results, is_finance

    def generate_query(self, query):
        llm_input = prompt_generator(query)
        completion = oai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=
        llm_input
        ).choices[0].message.content

        try:
            completion = json.loads(completion)
        except:
            completion = extract_json_objects(completion)

        if "domain" in completion.keys():
            domain = completion["domain"]
            is_finance = domain == "finance"
        else:
            is_finance = False

        return completion, is_finance

    def get_finance_kg_results(self, generated_query):
        formatted_time_list = []
        if 'datetime' in generated_query:
            datetime_list = generated_query['datetime'].split(' - ')
            for datetime in datetime_list:
                formatted_time_list.append(convert_to_standard_format(datetime.strip()))


        kg_results = []
        res = ""
        if "market_identifier" in generated_query.keys() and generated_query["market_identifier"] is not None:
            if isinstance(generated_query["market_identifier"], str):
                company_names = generated_query["market_identifier"].split(",")
            else:
                company_names = generated_query["market_identifier"]

            for company_name in company_names:
                try:
                    res = api.finance_get_company_name(company_name)["result"]

                    if res == []:
                        ticker_name = company_name.upper()
                    else:
                        ticker_name = api.finance_get_ticker_by_name(res[0])["result"]

                    if generated_query['metric'].lower().strip() == 'price':
                        response = api.finance_get_price_history(ticker_name)['result']
                    elif generated_query['metric'].lower().strip() == 'dividend':
                        response = api.finance_get_dividends_history(ticker_name)['result']
                    elif generated_query['metric'].lower().strip() == 'p/e ratio':
                        response = api.finance_get_pe_ratio(ticker_name)['result']
                    elif generated_query['metric'].lower().strip() == 'eps':
                        response = api.finance_get_eps(ticker_name)["result"]
                    elif generated_query['metric'].lower().strip() == 'marketcap' :
                        response = api.finance_get_market_capitalization(ticker_name)['result']
                    else:
                        response = api.finance_get_info(ticker_name)['result']
                        metric_value = get_metric_from_response(response, generated_query['metric'])
                        if metric_value is not None:
                            response = metric_value

                    try:
                        for formatted_time in formatted_time_list:
                            if formatted_time in response:
                                filtered_response = copy.deepcopy(response[formatted_time])
                            elif add_one_day(formatted_time) in response:
                                filtered_response = copy.deepcopy(response[add_one_day(formatted_time)])
                            elif subtract_one_day(formatted_time) in response:
                                filtered_response = copy.deepcopy(response[subtract_one_day(formatted_time)])
                            else:
                                filtered_response = copy.deepcopy(response)
                            kg_results.append({company_name + " " + generated_query["metric"]: filtered_response, 'time': formatted_time})
                    except:
                        kg_results.append({company_name + " " + generated_query["metric"]: response})

                except Exception as e:
                    print("Fail to parse the generated query")
                    pass

        kg_results = "<DOC>\n".join([str(res) for res in kg_results]) if len(kg_results) > 0 else ""
        return  kg_results

    def prompt_generator(self, query):
        user_message = ""
        user_message += f"Query: {query}\n"

        llm_input = [
          {"role": "system", "content": entity_extract_template},
          {"role": "user", "content": user_message},
        ]

        return llm_input
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — III. Implementing an LLM + Mock KG

원본 Cell `031`. 이 셀은 **III. Implementing an LLM + Mock KG** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class RAGWithKG:
    def __init__(self):
        self.kg_query_engine = KGQueryEngine()
        self.reader = Reader()

    def inference(self, query):
        # 1. retrieve relevant kg results
        kg_results, is_finance = self.kg_query_engine.query(query)

        # 2. answer the question based on the retrieved chunks
        answer = self.reader.generate_response(query, [kg_results])

        return answer, kg_results
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 6 — IV. Implementing an LLM + Web Search Results + Mock KG

원본 Cell `036`. 이 셀은 **IV. Implementing an LLM + Web Search Results + Mock KG** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### YOUR CODE HERE ###

class RAGWithSRKG:
    def __init__(self):
        self.retriever = LlamaIndexRetriever()
        self.kg_query_engine = KGQueryEngine()
        self.reader = Reader()

    def inference(self, query, search_results, topk):
        # 1. retrieve relevant chunks
        retrieved_results = self.retriever.retrieve(query, search_results, topk)

        # 2. retrieve relevant kg results
        kg_results, is_finance = self.kg_query_engine.query(query)

        # combined_results = [kg_results]
        # combined_results.extend(retrieved_results)
        if is_finance:
          combined_results = [kg_results]
        else:
          combined_results = retrieved_results

        # 3. answer the question based on the retrieved chunks
        answer = self.reader.generate_response(query, combined_results)

        return answer, combined_results
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?
