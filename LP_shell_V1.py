import streamlit as st
import pulp
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import re

# Set page config
st.set_page_config(
    page_title="LP Agent System with LangGraph & Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Ollama
@st.cache_resource
def init_llm():
    try:
        return Ollama(model="llama3.1")  # or "mistral", "codellama", etc.
    except Exception as e:
        st.error(f"Failed to connect to Ollama: {e}")
        return None

class AgentState(TypedDict):
    messages: List[Any]
    current_model: Optional[Any]
    model_data: Optional[Dict]
    query: str
    response: str
    status: str

class LPModelAgent:
    def __init__(self):
        self.models = {}
        self.model_counter = 0
    
    def create_model_from_json(self, lp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create LP model from structured JSON data"""
        try:
            model_id = f"model_{self.model_counter}"
            self.model_counter += 1
            
            # Create PuLP problem
            problem_type = lp_data.get('type', 'maximization')
            if problem_type == 'minimization':
                prob = pulp.LpProblem(f"LP_Problem_{model_id}", pulp.LpMinimize)
            else:
                prob = pulp.LpProblem(f"LP_Problem_{model_id}", pulp.LpMaximize)
            
            # Create variables
            variables = {}
            for var_data in lp_data.get('variables', []):
                name = var_data['name']
                low_bound = var_data.get('lowBound', 0)
                up_bound = var_data.get('upBound', None)
                cat = self._get_category(var_data.get('category', 'continuous'))
                variables[name] = pulp.LpVariable(name, low_bound, up_bound, cat)
            
            # Objective function
            objective_coeffs = lp_data.get('objective', {})
            objective_expr = pulp.lpSum([coeff * variables[name] 
                                       for name, coeff in objective_coeffs.items()])
            prob += objective_expr, "Objective_Function"
            
            # Constraints
            constraints = []
            for i, constr_data in enumerate(lp_data.get('constraints', [])):
                lhs_coeffs = constr_data.get('lhs', {})
                rhs = constr_data['rhs']
                sense = constr_data.get('sense', '<=')
                
                lhs_expr = pulp.lpSum([coeff * variables[name] 
                                     for name, coeff in lhs_coeffs.items()])
                
                if sense == '<=':
                    constraint = lhs_expr <= rhs
                elif sense == '>=':
                    constraint = lhs_expr >= rhs
                else:
                    constraint = lhs_expr == rhs
                
                prob += constraint, f"constraint_{i}"
                constraints.append({
                    'expression': str(constraint),
                    'sense': sense,
                    'rhs': rhs
                })
            
            model_info = {
                'model_id': model_id,
                'pulp_problem': prob,
                'variables': {name: str(var) for name, var in variables.items()},
                'objective': str(objective_expr),
                'constraints': constraints,
                'problem_type': problem_type,
                'status': 'created'
            }
            
            self.models[model_id] = model_info
            return model_info
            
        except Exception as e:
            return {'error': f"Model creation failed: {str(e)}"}
    
    def create_model_from_natural_language(self, description: str) -> Dict[str, Any]:
        """Create LP model from natural language description"""
        try:
            # Simple pattern matching for common LP problems
            if "production" in description.lower() or "profit" in description.lower():
                lp_data = self._create_production_example()
            elif "diet" in description.lower() or "nutrition" in description.lower():
                lp_data = self._create_diet_example()
            elif "transportation" in description.lower() or "shipping" in description.lower():
                lp_data = self._create_transportation_example()
            else:
                lp_data = self._create_default_example()
            
            return self.create_model_from_json(lp_data)
            
        except Exception as e:
            return {'error': f"Natural language model creation failed: {str(e)}"}  # Fixed the f-string syntax
    
    def _create_production_example(self) -> Dict[str, Any]:
        """Create a production planning example"""
        return {
            "type": "maximization",
            "variables": [
                {"name": "x1", "lowBound": 0, "category": "continuous"},
                {"name": "x2", "lowBound": 0, "category": "continuous"}
            ],
            "objective": {"x1": 300, "x2": 500},
            "constraints": [
                {"lhs": {"x1": 1, "x2": 2}, "rhs": 18, "sense": "<="},
                {"lhs": {"x1": 1, "x2": 1}, "rhs": 12, "sense": "<="},
                {"lhs": {"x1": 2, "x2": 1}, "rhs": 16, "sense": "<="}
            ]
        }
    
    def _create_diet_example(self) -> Dict[str, Any]:
        """Create a diet optimization example"""
        return {
            "type": "minimization",
            "variables": [
                {"name": "beef", "lowBound": 0, "category": "continuous"},
                {"name": "chicken", "lowBound": 0, "category": "continuous"}
            ],
            "objective": {"beef": 3.5, "chicken": 2.5},
            "constraints": [
                {"lhs": {"beef": 2, "chicken": 4}, "rhs": 10, "sense": ">="},
                {"lhs": {"beef": 3, "chicken": 2}, "rhs": 8, "sense": ">="}
            ]
        }
    
    def _create_transportation_example(self) -> Dict[str, Any]:
        """Create a transportation problem example"""
        return {
            "type": "minimization",
            "variables": [
                {"name": "x11", "lowBound": 0, "category": "continuous"},
                {"name": "x12", "lowBound": 0, "category": "continuous"},
                {"name": "x21", "lowBound": 0, "category": "continuous"},
                {"name": "x22", "lowBound": 0, "category": "continuous"}
            ],
            "objective": {"x11": 10, "x12": 20, "x21": 15, "x22": 25},
            "constraints": [
                {"lhs": {"x11": 1, "x12": 1}, "rhs": 100, "sense": "<="},
                {"lhs": {"x21": 1, "x22": 1}, "rhs": 150, "sense": "<="},
                {"lhs": {"x11": 1, "x21": 1}, "rhs": 120, "sense": ">="},
                {"lhs": {"x12": 1, "x22": 1}, "rhs": 80, "sense": ">="}
            ]
        }
    
    def _create_default_example(self) -> Dict[str, Any]:
        """Create a default LP example"""
        return {
            "type": "maximization",
            "variables": [
                {"name": "x1", "lowBound": 0, "category": "continuous"},
                {"name": "x2", "lowBound": 0, "category": "continuous"}
            ],
            "objective": {"x1": 3, "x2": 2},
            "constraints": [
                {"lhs": {"x1": 1, "x2": 1}, "rhs": 10, "sense": "<="},
                {"lhs": {"x1": 2, "x2": 1}, "rhs": 15, "sense": "<="}
            ]
        }
    
    def solve_model(self, model_id: str) -> Dict[str, Any]:
        """Solve the LP model"""
        if model_id not in self.models:
            return {'error': 'Model not found'}
        
        try:
            prob = self.models[model_id]['pulp_problem']
            prob.solve()
            
            results = {
                'status': pulp.LpStatus[prob.status],
                'objective_value': pulp.value(prob.objective),
                'variables': {var.name: var.varValue for var in prob.variables()}
            }
            
            self.models[model_id]['results'] = results
            self.models[model_id]['status'] = 'solved'
            return results
            
        except Exception as e:
            return {'error': f"Solution failed: {str(e)}"}
    
    def _get_category(self, cat_str: str) -> int:
        cat_map = {
            'continuous': pulp.LpContinuous,
            'integer': pulp.LpInteger,
            'binary': pulp.LpBinary
        }
        return cat_map.get(cat_str.lower(), pulp.LpContinuous)

class NLPAgent:
    def __init__(self, lp_agent: LPModelAgent):
        self.lp_agent = lp_agent
        self.llm = init_llm()
    
    def process_query(self, query: str) -> str:
        """Process natural language query"""
        query_lower = query.lower()
        
        # Model creation queries
        if any(word in query_lower for word in ['create', 'build', 'make', 'new model']):
            return self._handle_model_creation(query)
        
        # Model solving queries
        elif any(word in query_lower for word in ['solve', 'optimize', 'run']):
            return self._handle_solve_request(query)
        
        # Information queries
        elif any(word in query_lower for word in ['status', 'result', 'solution', 'value']):
            return self._handle_info_request(query)
        
        # List models
        elif any(word in query_lower for word in ['list', 'show models', 'what models']):
            return self._handle_list_models()
        
        # Help
        elif any(word in query_lower for word in ['help', 'what can you do']):
            return self._get_help_message()
        
        # General conversation with LLM
        else:
            return self._general_conversation(query)
    
    def _handle_model_creation(self, query: str) -> str:
        """Handle model creation requests"""
        try:
            model_info = self.lp_agent.create_model_from_natural_language(query)
            if 'error' in model_info:
                return f"Error creating model: {model_info['error']}"
            
            return f"""✅ Model created successfully!
            
**Model ID**: {model_info['model_id']}
**Type**: {model_info['problem_type']}
**Objective**: {model_info['objective']}
**Variables**: {', '.join(model_info['variables'].keys())}
**Constraints**: {len(model_info['constraints'])} constraints

You can now solve it by saying 'solve {model_info['model_id']}'"""
            
        except Exception as e:
            return f"Error creating model: {str(e)}"
    
    def _handle_solve_request(self, query: str) -> str:
        """Handle model solving requests"""
        # Extract model ID
        model_id = None
        for mid in self.lp_agent.models.keys():
            if mid in query:
                model_id = mid
                break
        
        if not model_id and self.lp_agent.models:
            model_id = list(self.lp_agent.models.keys())[-1]  # Use latest model
        
        if not model_id:
            return "No models available. Please create a model first."
        
        results = self.lp_agent.solve_model(model_id)
        if 'error' in results:
            return f"Error solving model: {results['error']}"
        
        # Format results nicely
        variables_text = "\n".join([f"- {var}: {value:.2f}" 
                                  for var, value in results['variables'].items()])
        
        return f"""🎯 Model {model_id} solved!

**Status**: {results['status']}
**Objective Value**: {results['objective_value']:.2f}

**Variable Values**:
{variables_text}"""
    
    def _handle_info_request(self, query: str) -> str:
        """Handle information requests"""
        if not self.lp_agent.models:
            return "No models available. Please create a model first."
        
        model_id = None
        for mid in self.lp_agent.models.keys():
            if mid in query:
                model_id = mid
                break
        
        if not model_id:
            model_id = list(self.lp_agent.models.keys())[-1]
        
        model = self.lp_agent.models[model_id]
        response = f"**Model {model_id}**\n"
        response += f"Type: {model['problem_type']}\n"
        response += f"Status: {model.get('status', 'created')}\n"
        response += f"Objective: {model['objective']}\n"
        response += f"Variables: {len(model['variables'])}\n"
        response += f"Constraints: {len(model['constraints'])}\n"
        
        if 'results' in model:
            response += f"\n**Results**\n"
            response += f"Objective Value: {model['results']['objective_value']:.2f}\n"
            for var, value in model['results']['variables'].items():
                response += f"{var}: {value:.2f}\n"
        
        return response
    
    def _handle_list_models(self) -> str:
        """Handle list models request"""
        if not self.lp_agent.models:
            return "No models available."
        
        response = "**Available Models:**\n"
        for model_id, model_info in self.lp_agent.models.items():
            response += f"- {model_id}: {model_info['problem_type']} ({model_info.get('status', 'created')})\n"
        
        return response
    
    def _get_help_message(self) -> str:
        """Return help message"""
        return """**I can help you with linear programming models!**

**Commands:**
- Create a model: "create a production model", "build diet optimization"
- Solve models: "solve model_0", "optimize the latest model"
- Get info: "show results", "what's the status?"
- List models: "list models", "show all models"

**Example problems:**
- Production planning
- Diet optimization  
- Transportation problems
- Resource allocation"""

    def _general_conversation(self, query: str) -> str:
        """Handle general conversation"""
        try:
            if self.llm:
                messages = [
                    SystemMessage(content="You are a helpful assistant for linear programming. Keep responses concise and focused on LP modeling."),
                    HumanMessage(content=query)
                ]
                response = self.llm.invoke(messages)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                return "I'm here to help with linear programming models. You can create models, solve them, and ask about results."
        except:
            return "I'm here to help with linear programming models. You can create models, solve them, and ask about results."

# Create LangGraph workflow
def create_agent_workflow(lp_agent: LPModelAgent, nlp_agent: NLPAgent):
    """Create the LangGraph workflow for multi-agent system"""
    
    def router(state: AgentState) -> str:
        """Route to appropriate agent based on query type"""
        query = state["query"].lower()
        
        if any(word in query for word in ['create', 'build', 'make', 'new']):
            return "model_creator"
        elif any(word in query for word in ['solve', 'optimize', 'run']):
            return "solver"
        else:
            return "nlp_agent"
    
    def model_creator(state: AgentState) -> AgentState:
        """Model creation agent node"""
        response = nlp_agent._handle_model_creation(state["query"])
        return {**state, "response": response, "status": "model_created"}
    
    def solver(state: AgentState) -> AgentState:
        """Model solving agent node"""
        response = nlp_agent._handle_solve_request(state["query"])
        return {**state, "response": response, "status": "model_solved"}
    
    def nlp_agent_node(state: AgentState) -> AgentState:
        """NLP agent node for general queries"""
        response = nlp_agent.process_query(state["query"])
        return {**state, "response": response, "status": "general_response"}
    
    # Build the graph using StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("model_creator", model_creator)
    workflow.add_node("solver", solver)
    workflow.add_node("nlp_agent", nlp_agent_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        START,
        router,
        {
            "model_creator": "model_creator",
            "solver": "solver", 
            "nlp_agent": "nlp_agent"
        }
    )
    
    workflow.add_edge("model_creator", END)
    workflow.add_edge("solver", END)
    workflow.add_edge("nlp_agent", END)
    
    return workflow.compile()

# Streamlit UI
def main():
    st.title("🤖 Multi-Agent LP System with LangGraph & Ollama")
    st.markdown("Create and solve linear programming models using natural language!")
    
    # Initialize agents
    if 'lp_agent' not in st.session_state:
        st.session_state.lp_agent = LPModelAgent()
    if 'nlp_agent' not in st.session_state:
        st.session_state.nlp_agent = NLPAgent(st.session_state.lp_agent)
    if 'workflow' not in st.session_state:
        st.session_state.workflow = create_agent_workflow(
            st.session_state.lp_agent, 
            st.session_state.nlp_agent
        )
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Ollama status
        llm = init_llm()
        if llm:
            st.success("✅ Ollama connected")
        else:
            st.warning("⚠️ Ollama not available - using rule-based system")
        
        st.subheader("Available Models")
        if st.session_state.lp_agent.models:
            for model_id, model_info in st.session_state.lp_agent.models.items():
                with st.expander(f"📊 {model_id}"):
                    st.write(f"**Type**: {model_info['problem_type']}")
                    st.write(f"**Status**: {model_info.get('status', 'created')}")
                    st.write(f"**Variables**: {len(model_info['variables'])}")
                    
                    if 'results' in model_info:
                        st.write("**Objective Value**:", 
                               f"{model_info['results']['objective_value']:.2f}")
        else:
            st.write("No models created yet")
        
        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            st.rerun()
        
        if st.button("Reset Models"):
            st.session_state.lp_agent = LPModelAgent()
            st.session_state.nlp_agent = NLPAgent(st.session_state.lp_agent)
            st.session_state.workflow = create_agent_workflow(
                st.session_state.lp_agent, 
                st.session_state.nlp_agent
            )
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with LP Agents")
        
        # Display conversation
        for i, (query, response, agent_type) in enumerate(st.session_state.conversation):
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                st.markdown(response)
                if agent_type == "model_created":
                    st.success("Model created successfully!")
                elif agent_type == "model_solved":
                    st.success("Model solved successfully!")
        
        # Chat input
        if prompt := st.chat_input("Describe your LP problem or ask a question..."):
            # Add user message
            st.session_state.conversation.append((prompt, "", ""))
            
            # Process with LangGraph workflow
            initial_state = AgentState(
                messages=[],
                current_model=None,
                model_data=None,
                query=prompt,
                response="",
                status="processing"
            )
            
            try:
                result = st.session_state.workflow.invoke(initial_state)
                response = result["response"]
                status = result["status"]
                
                # Update conversation
                st.session_state.conversation[-1] = (prompt, response, status)
                st.rerun()
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                st.session_state.conversation[-1] = (prompt, error_msg, "error")
                st.rerun()
    
    with col2:
        st.subheader("📋 Example Queries")
        
        examples = [
            "Create a production planning model",
            "Build a diet optimization model", 
            "Solve the latest model",
            "Show me all models",
            "Create transportation model"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                if prompt := example:
                    st.session_state.conversation.append((prompt, "", ""))
                    
                    initial_state = AgentState(
                        messages=[],
                        current_model=None,
                        model_data=None,
                        query=prompt,
                        response="",
                        status="processing"
                    )
                    
                    try:
                        result = st.session_state.workflow.invoke(initial_state)
                        response = result["response"]
                        status = result["status"]
                        
                        st.session_state.conversation[-1] = (prompt, response, status)
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"Error processing request: {str(e)}"
                        st.session_state.conversation[-1] = (prompt, error_msg, "error")
                        st.rerun()
        
        st.subheader("🔧 Quick Actions")
        
        if st.session_state.lp_agent.models:
            model_options = list(st.session_state.lp_agent.models.keys())
            selected_model = st.selectbox("Select model", model_options)
            
            if st.button("Solve Selected Model"):
                prompt = f"solve {selected_model}"
                st.session_state.conversation.append((prompt, "", ""))
                
                initial_state = AgentState(
                    messages=[],
                    current_model=None,
                    model_data=None,
                    query=prompt,
                    response="",
                    status="processing"
                )
                
                try:
                    result = st.session_state.workflow.invoke(initial_state)
                    response = result["response"]
                    status = result["status"]
                    
                    st.session_state.conversation[-1] = (prompt, response, status)
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)}"
                    st.session_state.conversation[-1] = (prompt, error_msg, "error")
                    st.rerun()
            
            if st.button("Show Model Details"):
                prompt = f"show details for {selected_model}"
                st.session_state.conversation.append((prompt, "", ""))
                
                initial_state = AgentState(
                    messages=[],
                    current_model=None,
                    model_data=None,
                    query=prompt,
                    response="",
                    status="processing"
                )
                
                try:
                    result = st.session_state.workflow.invoke(initial_state)
                    response = result["response"]
                    status = result["status"]
                    
                    st.session_state.conversation[-1] = (prompt, response, status)
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)}"
                    st.session_state.conversation[-1] = (prompt, error_msg, "error")
                    st.rerun()

if __name__ == "__main__":
    main()