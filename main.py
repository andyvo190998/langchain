from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk [ ˈiːlɒn ˈɹiːv ˈmʌsk ] (born June 28, 1971 , in Pretoria , South Africa ) is a South African - Canadian - American entrepreneur. He is best known as the founder and chief technology officer of X.com, the predecessor to PayPal , and the aerospace company SpaceX , as well as the CEO and co-owner of the electric car manufacturer Tesla . He has also founded other companies and, since 2022, has held a majority stake in the microblogging service X (formerly Twitter).

With an estimated net worth of over $1 trillion, Musk is the richest person in the world and the first dollar trillionaire . [ 1 ] With his financial and media power, he significantly influences public political discourse worldwide. He holds libertarian views and (since 2022) predominantly right-wing political positions. In addition to his activities in the United States, he supports right-wing populist and far-right parties in Europe and South America. Through his posts on X, he has also become known for spreading conspiracy theories and making provocative statements, which have been criticized as scientifically unfounded, fear-mongering, antisemitic and transphobic , and as " trolling ."
    """

    summary_template = """
        given the information {information} about a person I want to create:
        1. A short summary
        2. Two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})


if __name__ == "__main__":
    main()
