from langchain_core.prompts import ChatPromptTemplate



system_prompt = (
"You are an expert Data Scientist assistat of qestion-answering tasks."
"Use the following pieces of retrieved context to answer "
"the question. If you don't find any related context then say that you "
"don't know. Do not give any halusinating answer of this. Use the three sentece maximum and keep the "
"answer concise."
"\n\n"
"{context}"
 )

chat_prompt = ChatPromptTemplate.from_messages([
  ("system", system_prompt),
  ("user", "{input}" )]
)