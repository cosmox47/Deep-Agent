import os
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Put your Groq API key here (Starts with gsk_)
os.environ["GROQ_API_KEY"] = "gsk_*****"

# 1. Define the Shared State Schema
class AgentState(TypedDict):
    user_request: str
    problem_statement: str
    prd: str
    architecture_design: str
    final_code: str


# 2. Initialize the LLM
# Ensure GROQ_API_KEY is replaced properly before running
try:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
except Exception as e:
    print(f"⚠️ Error initializing LLM. Did you set your API key? {e}")
    llm = None


# 3. Define the Node Functions
def supervisor_node(state: AgentState):
    print("\n--- 🤖 Supervisor is thinking... ---")
    sys_msg = SystemMessage(content="You are the Supervisor Agent in a software development team.\nYour job is to take raw user requests, clarify them, and break them down into a clear, actionable problem statement for the rest of the team.\nDo NOT write code. Just output a clear, professional summary and scoped definition of what needs to be built.")
    human_msg = HumanMessage(content=f"User Request: {state['user_request']}\n\nPlease refine this into a clear problem statement.")
    
    response = llm.invoke([sys_msg, human_msg])
    print(f"✅ Generated Problem Statement.\n")
    return {"problem_statement": response.content}

def pm_node(state: AgentState):
    print("\n--- 🤖 Product Manager is thinking... ---")
    sys_msg = SystemMessage(content="You are the Product Manager Agent.\nYou receive a finalized problem statement from the Supervisor.\nYour job is to write a detailed Product Requirement Document (PRD).\nInclude:\n1. Core Objectives & Features\n2. User Stories\n3. Acceptance Criteria\nDo NOT write code. Focus on the 'what' and 'why', not the 'how'.")
    human_msg = HumanMessage(content=f"Problem Statement:\n{state['problem_statement']}\n\nPlease create a comprehensive PRD.")
    
    response = llm.invoke([sys_msg, human_msg])
    print(f"✅ Generated PRD.\n")
    return {"prd": response.content}

def architect_node(state: AgentState):
    print("\n--- 🤖 Architect is thinking... ---")
    sys_msg = SystemMessage(content="You are the Software Architect Agent.\nYou receive a PRD from the Product Manager.\nYour job is to design the robust, scalable technical architecture to implement the PRD.\nInclude:\n1. Recommended Technology Stack\n2. File/Folder Structure\n3. High-level components, data models, and their interactions.\nDo NOT write the actual implementation code, just the technical architecture and design documents.")
    human_msg = HumanMessage(content=f"Product Requirement Document (PRD):\n{state['prd']}\n\nPlease create a technical architecture.")
    
    response = llm.invoke([sys_msg, human_msg])
    print(f"✅ Generated Architecture Design.\n")
    return {"architecture_design": response.content}

def engineer_node(state: AgentState):
    print("\n--- 🤖 Engineer is thinking... ---")
    sys_msg = SystemMessage(content="You are the Lead Software Engineer Agent.\nYou receive a PRD from the PM and an Architecture Design from the Architect.\nYour job is to write the complete, functional implementation code based on those documents.\nProvide clean, well-commented, and robust code.\nIf there are multiple files, clearly indicate the file names and the exact code that belongs in each file.\nThink step-by-step and deliver production-ready code.")
    human_msg = HumanMessage(content=f"PRD:\n{state['prd']}\n\nArchitecture Design:\n{state['architecture_design']}\n\nPlease write the complete implementation code.")
    
    response = llm.invoke([sys_msg, human_msg])
    print(f"✅ Generated Code.\n")
    return {"final_code": response.content}


# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("product_manager", pm_node)
workflow.add_node("architect", architect_node)
workflow.add_node("engineer", engineer_node)

# Add Edges (Linear Flow)
workflow.add_edge(START, "supervisor")
workflow.add_edge("supervisor", "product_manager")
workflow.add_edge("product_manager", "architect")
workflow.add_edge("architect", "engineer")
workflow.add_edge("engineer", END)

# Compile Graph
app = workflow.compile()


# 5. Execution Runner
def main():
    if not llm:
        return
        
    print("🚀 Deep Agent LangGraph Initialized!")
    print("Roles configured: Supervisor -> Product Manager -> Architect -> Engineer")
    
    while True:
        try:
            print("\n" + "="*50)
            user_input = input("👤 Enter your task (e.g., 'Build a classic snake game') or 'exit' to quit: \n> ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
                
            if not user_input.strip():
                continue

            # Initial State Data
            initial_state = {"user_request": user_input}
            
            # Execute the Graph
            final_state = app.invoke(initial_state)
            
            print("\n✅ Task Complete! Here is the final output from the Engineer:\n")
            print("=" * 50)
            print(final_state["final_code"])
            print("=" * 50)
            
            print("\n(Note: You can inspect 'final_state' variable if you want to see the PRD or Architecture as well!)")
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
