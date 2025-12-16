from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from src.model import get_llm
from src.prompt import prompt
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableSequence
class Agent:
    def __init__(self,retriever=None):
        self.prompt = prompt()
        self.parser = StrOutputParser()
        self.llm = get_llm()
        self.retriever = retriever or get_retriever()
    
    def get_context(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    def parallel_chain(self):
        chain = RunnableParallel(
            {
                'context': self.retriever | RunnableLambda(self.get_context),
                'question':RunnablePassthrough()
            }
        )
        return chain

    def invoke(self,question):
        chain = (
                self.parallel_chain()
                | self.prompt
                | self.llm
                | self.parser
            )

        res = chain.invoke(question)
        return res
    


if __name__ == "__main__":
    agent = Agent()
    print(agent.invoke("What is this document about"))