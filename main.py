
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

writer = Agent(
  role='Writer',
  goal='Write a compelling and engaging article based on the research provided.',
  backstory='You are a professional writer, known for your clear and concise writing style.',
  verbose=True,
  allow_delegation=False,
  llm=ollama_llm_llama3
)

# Define your tasks
task1 = Task(
  description='Research the topic of "AI in 2025"',
  expected_output='A summary of the latest trends and advancements in AI.',
  agent=researcher
)

task2 = Task(
  description='Write an article based on the research from the researcher.',
  expected_output='A 500-word article on the topic of "AI in 2025".',
  agent=writer
)

# Create the crew
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  process=Process.SEQUENTIAL
)

# Get the crew to work
result = crew.kickoff()

print("######################")
print(result)

