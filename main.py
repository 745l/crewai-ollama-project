
from crewai import Agent, Task, Crew, Process
from langchain_ollama.llms import OllamaLLM

# Initialize the Ollama model
ollama_llm_mistral = OllamaLLM(model="mistral")
ollama_llm_llama3 = OllamaLLM(model="llama3")

# Define your agents
researcher = Agent(
  role='Researcher',
  goal='Research and provide information on a given topic.',
  backstory='You are an expert researcher, skilled in finding and summarizing relevant information.',
  verbose=True,
  allow_delegation=False,
  llm=ollama_llm_llama3
)
mirko = Agent(
  role=\'Mirko - Task Orchestrator\',
  goal=\'Orchestrate research and writing tasks to produce a comprehensive article.\',
  backstory=\'You are a seasoned project manager, expert in delegating tasks and ensuring timely, high-quality output from a team of specialized agents.\',
  verbose=True,
  allow_delegation=True,
 llm=ollama_llm_mistral # Using Mistral for orchestration
)

slavko = Agent(
  role=\'Slavko - Task Assistant\',
  goal=\'Evaluate task feasibility, provide insights, and communicate with the orchestrator.\',
  backstory=\'You are a diligent assistant, skilled in analyzing tasks, assessing their viability, and reporting back to the orchestrator.\',
  verbose=True,
  allow_delegation=False,
  llm=ollama_llm_llama3 # Using Llama3 for assistant tasks
)


writer = Agent(
  role=\'Writer\',  goal='Write a compelling and engaging article based on the research provided.',
  backstory='You are a professional writer, known for your clear and concise writing style.',
  verbose=True,
  allow_delegation=False,
  llm=ollama_llm_llama3
)

# Define your tasks
research_task = Task(
  description=\'Research the topic of "AI in 2025"\',
  expected_output=\'A summary of the latest trends and advancements in AI.\',
  agent=researcher
)

writing_task = Task(
  description=\'Write an article based on the research from the researcher.\',
  expected_output=\'A 500-word article on the topic of "AI in 2025".\',
  agent=writer
)

evaluate_task = Task(
  description=\'Evaluate the feasibility and potential challenges of creating an article on "AI in 2025". Communicate your findings to Mirko.\',
  expected_output=\'A detailed assessment of the task\'s feasibility, including potential risks and opportunities.\',
  agent=slavko
)

orchestrate_task = Task(
  description=\'Oversee the research and writing process for an article on "AI in 2025", taking into account Slavko\'s evaluation.\',
  expected_output=\'A final, comprehensive article on "AI in 2025" based on the delegated tasks and Slavko\'s insights.\',
  agent=mirko,
  context=[evaluate_task, research_task, writing_task]
)





# Create the crew
crew = Crew(
  agents=[researcher, writer, mirko, slavko],
  tasks=[orchestrate_task],
  process=Process.sequential # Mirko (orchestrator) will manage the flow
)



# Get the crew to work
result = crew.kickoff()

print("######################")
print(result)

