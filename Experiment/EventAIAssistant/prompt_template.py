# prompt_template = """
# Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
# provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
# Context:\n {context}?\n
# Question: \n{question}\n

# Answer:
# """

prompt_template = """Question: {question}
    Answer according to the context {context}
    Answer: Give the answer as your are speaking to person in real time"""
