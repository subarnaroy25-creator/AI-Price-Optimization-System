import streamlit as st
import pulp
import pandas as pd
import requests
import json
from typing import Dict, List, Any, Optional, TypedDict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Price Optimization System with MCP",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MCP Server Client
class MCPClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def query(self, query_type: str, params: Dict = None) -> Dict[str, Any]:
        """Query the MCP server"""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query_type, "params": params or {}}
            )
            return response.json().get('result', {})
        except Exception as e:
            return {"error": f"Failed to connect to MCP server: {str(e)}"}
    
    def get_products(self) -> List[str]:
        """Get list of available products"""
        try:
            response = requests.get(f"{self.base_url}/products")
            return response.json().get('products', [])
        except:
            return ["Product_A", "Product_B", "Product_C", "Product_D", "Product_E"]
    
    def health_check(self) -> bool:
        """Check if MCP server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

# Initialize services
@st.cache_resource
def init_mcp_client():
    return MCPClient()

@st.cache_resource
def init_llm():
    try:
        return Ollama(model="llama3.1")
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
    explanation: str
    current_agent: str
    handover_reason: str

class DataAnalysisAgent:
    """Agent specialized in analyzing historical data"""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        
    def analyze_historical_data(self, query: str) -> str:
        """Analyze historical data based on query"""
        try:
            # Determine what type of analysis is needed
            if "summary" in query.lower() or "overview" in query.lower():
                result = self.mcp_client.query("summary")
                return self._format_summary(result)
            elif "elasticity" in query.lower():
                result = self.mcp_client.query("elasticity")
                return self._format_elasticity(result)
            elif "trend" in query.lower() or "history" in query.lower():
                result = self.mcp_client.query("time_series", {"days": 90})
                return self._format_trends(result)
            elif "price range" in query.lower() or "feasible" in query.lower():
                result = self.mcp_client.query("price_ranges")
                return self._format_price_ranges(result)
            else:
                # General data exploration
                summary = self.mcp_client.query("summary")
                elasticity = self.mcp_client.query("elasticity")
                return self._format_comprehensive_analysis(summary, elasticity)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
    
    def _format_summary(self, summary: Dict) -> str:
        """Format summary statistics"""
        return f"""📊 **Historical Data Summary**

- **Total Products**: {summary.get('total_products', 'N/A')}
- **Analysis Period**: {summary.get('total_days', 'N/A')} days
- **Total Revenue**: ${summary.get('total_revenue', 0):,.2f}
- **Total Profit**: ${summary.get('total_profit', 0):,.2f}
- **Average Price**: ${summary.get('avg_price', 0):.2f}
- **Average Demand**: {summary.get('avg_demand', 0):.0f} units/day

This data provides a foundation for price optimization modeling."""
    
    def _format_elasticity(self, elasticity: Dict) -> str:
        """Format elasticity analysis"""
        if not elasticity or 'error' in elasticity:
            return "Elasticity data not available."
        
        analysis = "📈 **Price Elasticity Analysis**\n\n"
        for product, stats in elasticity.items():
            elast = stats.get('elasticity', 0)
            analysis += f"**{product}**: Elasticity = {elast:.3f}\n"
            if elast < -1:
                analysis += "  - Highly elastic (price sensitive)\n"
            elif elast > -1:
                analysis += "  - Inelastic (less price sensitive)\n"
            analysis += f"  - Avg Price: ${stats.get('avg_price', 0):.2f}\n"
            analysis += f"  - Avg Demand: {stats.get('avg_demand', 0):.0f} units\n\n"
        
        return analysis
    
    def _format_trends(self, trends: List[Dict]) -> str:
        """Format trend analysis"""
        if not trends or 'error' in trends:
            return "Trend data not available."
        
        df = pd.DataFrame(trends)
        recent_date = df['date'].max() if 'date' in df.columns else "N/A"
        
        return f"""📅 **Recent Trends Analysis**

- **Most Recent Data**: {recent_date}
- **Data Points**: {len(trends)} records
- **Coverage**: Last 90 days across all products

Trend analysis shows seasonal patterns and price-demand relationships."""
    
    def _format_price_ranges(self, ranges: Dict) -> str:
        """Format price range analysis"""
        if not ranges or 'error' in ranges:
            return "Price range data not available."
        
        analysis = "💰 **Feasible Price Ranges**\n\n"
        for product, stats in ranges.items():
            analysis += f"**{product}**:\n"
            analysis += f"  - Min Price: ${stats.get('min_price', 0):.2f}\n"
            analysis += f"  - Max Price: ${stats.get('max_price', 0):.2f}\n"
            analysis += f"  - Current Avg: ${stats.get('current_avg', 0):.2f}\n"
            analysis += f"  - Cost Price: ${stats.get('cost_price', 0):.2f}\n\n"
        
        return analysis
    
    def _format_comprehensive_analysis(self, summary: Dict, elasticity: Dict) -> str:
        """Format comprehensive data analysis"""
        return f"""🔍 **Comprehensive Data Analysis**

{self._format_summary(summary)}

{self._format_elasticity(elasticity)}

The historical data shows clear patterns for price optimization modeling."""

class PriceOptimizationAgent:
    """Agent for creating and solving price optimization models"""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.models = {}
        self.model_counter = 0
    
    def create_optimization_model(self, constraints: Dict = None) -> Dict[str, Any]:
        """Create price optimization model using historical data"""
        try:
            # Get data from MCP server
            price_ranges = self.mcp_client.query("price_ranges")
            elasticity_data = self.mcp_client.query("elasticity")
            summary_data = self.mcp_client.query("summary")
            
            if 'error' in price_ranges:
                return {'error': 'Failed to fetch data from MCP server'}
            
            model_id = f"price_model_{self.model_counter}"
            self.model_counter += 1
            
            # Create optimization problem
            prob = pulp.LpProblem("Price_Optimization", pulp.LpMaximize)
            
            # Create variables for each product
            variables = {}
            for product, data in price_ranges.items():
                var_name = f"price_{product}"
                variables[var_name] = pulp.LpVariable(
                    var_name, 
                    lowBound=data['min_price'], 
                    upBound=data['max_price'],
                    cat='Continuous'
                )
            
            # Objective function: Maximize total profit
            objective_terms = []
            for product, data in price_ranges.items():
                var_name = f"price_{product}"
                cost = data['cost_price']
                elasticity = elasticity_data.get(product, {}).get('elasticity', -2.0)
                base_demand = elasticity_data.get(product, {}).get('avg_demand', 100)
                base_price = data['current_avg']
                
                # Demand function based on elasticity
                # demand = base_demand * (price/base_price)^elasticity
                price_var = variables[var_name]
                demand_expr = base_demand * (price_var / base_price) ** elasticity
                profit_expr = demand_expr * (price_var - cost)
                objective_terms.append(profit_expr)
            
            prob += pulp.lpSum(objective_terms), "Total_Profit"
            
            # Add constraints
            constraints_list = []
            
            # Budget constraint (if provided)
            if constraints and 'max_total_price' in constraints:
                prob += pulp.lpSum([variables[var] for var in variables]) <= constraints['max_total_price'], "Total_Price_Limit"
            
            # Market share constraints
            for product in price_ranges.keys():
                var_name = f"price_{product}"
                # Ensure price is above cost
                prob += variables[var_name] >= price_ranges[product]['cost_price'] * 1.1, f"Min_Profit_Margin_{product}"
            
            model_info = {
                'model_id': model_id,
                'pulp_problem': prob,
                'variables': variables,
                'products': list(price_ranges.keys()),
                'price_ranges': price_ranges,
                'elasticity_data': elasticity_data,
                'status': 'created',
                'constraints': constraints or {}
            }
            
            self.models[model_id] = model_info
            return model_info
            
        except Exception as e:
            return {'error': f"Model creation failed: {str(e)}"}
    
    def solve_model(self, model_id: str) -> Dict[str, Any]:
        """Solve the price optimization model"""
        if model_id not in self.models:
            return {'error': 'Model not found'}
        
        try:
            prob = self.models[model_id]['pulp_problem']
            prob.solve()
            
            results = {
                'status': pulp.LpStatus[prob.status],
                'objective_value': pulp.value(prob.objective),
                'prices': {},
                'profits': {},
                'demands': {}
            }
            
            # Calculate detailed results
            model_info = self.models[model_id]
            for var_name, var in model_info['variables'].items():
                product = var_name.replace('price_', '')
                optimal_price = var.varValue
                results['prices'][product] = optimal_price
                
                # Calculate demand and profit
                elasticity = model_info['elasticity_data'].get(product, {}).get('elasticity', -2.0)
                base_demand = model_info['elasticity_data'].get(product, {}).get('avg_demand', 100)
                base_price = model_info['price_ranges'][product]['current_avg']
                cost = model_info['price_ranges'][product]['cost_price']
                
                demand = base_demand * (optimal_price / base_price) ** elasticity
                profit = demand * (optimal_price - cost)
                
                results['demands'][product] = demand
                results['profits'][product] = profit
            
            self.models[model_id]['results'] = results
            self.models[model_id]['status'] = 'solved'
            return results
            
        except Exception as e:
            return {'error': f"Solution failed: {str(e)}"}

class ExplanationAgent:
    """Agent for explaining results in natural language"""
    
    def __init__(self, optimization_agent: PriceOptimizationAgent):
        self.optimization_agent = optimization_agent
        self.llm = init_llm()
    
    def explain_optimization_results(self, model_id: str) -> str:
        """Explain optimization results in natural language"""
        if model_id not in self.optimization_agent.models:
            return "Model not found."
        
        model = self.optimization_agent.models[model_id]
        if 'results' not in model:
            return "Model not solved yet."
        
        results = model['results']
        
        # Use LLM for sophisticated explanation
        if self.llm:
            return self._explain_with_llm(model, results)
        else:
            return self._explain_with_rules(model, results)
    
    def _explain_with_llm(self, model: Dict, results: Dict) -> str:
        """Use LLM for natural language explanation"""
        try:
            prompt = f"""
            Explain these price optimization results in business-friendly language:
            
            Optimization Objective: Maximize total profit
            Status: {results['status']}
            Total Profit: ${results['objective_value']:.2f}
            
            Optimal Prices:
            {chr(10).join([f'{product}: ${price:.2f}' for product, price in results['prices'].items()])}
            
            Expected Demands:
            {chr(10).join([f'{product}: {demand:.0f} units' for product, demand in results['demands'].items()])}
            
            Please explain:
            1. What the optimal pricing strategy achieves
            2. How prices compare to historical averages
            3. Which products are most sensitive to price changes
            4. Business implications of the results
            5. Any recommendations for implementation
            
            Keep it practical and actionable for business decision-makers.
            """
            
            messages = [
                SystemMessage(content="You are a pricing analyst explaining optimization results to business executives."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return self._explain_with_rules(model, results)
    
    def _explain_with_rules(self, model: Dict, results: Dict) -> str:
        """Rule-based explanation"""
        explanation = "## 💰 Price Optimization Results Explained\n\n"
        
        explanation += f"**Optimization Status**: {results['status']}\n"
        explanation += f"**Expected Total Profit**: ${results['objective_value']:,.2f}\n\n"
        
        explanation += "### 📊 Optimal Pricing Strategy:\n"
        for product, price in results['prices'].items():
            current_avg = model['price_ranges'][product]['current_avg']
            change_pct = ((price - current_avg) / current_avg) * 100
            explanation += f"- **{product}**: ${price:.2f} "
            explanation += f"({change_pct:+.1f}% from current ${current_avg:.2f})\n"
        
        explanation += "\n### 🎯 Key Insights:\n"
        
        # Find most changed prices
        price_changes = []
        for product, price in results['prices'].items():
            current_avg = model['price_ranges'][product]['current_avg']
            change_pct = ((price - current_avg) / current_avg) * 100
            price_changes.append((product, change_pct))
        
        price_changes.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if price_changes:
            most_changed = price_changes[0]
            explanation += f"- **{most_changed[0]}** has the largest price adjustment ({most_changed[1]:+.1f}%)\n"
        
        explanation += "- Prices are optimized based on historical price elasticity\n"
        explanation += "- The model balances demand sensitivity with profit margins\n"
        
        explanation += "\n### 💡 Recommendations:\n"
        explanation += "- Implement prices gradually to monitor market response\n"
        explanation += "- Track actual demand vs. predicted demand\n"
        explanation += "- Consider seasonal adjustments beyond the model\n"
        
        return explanation

class MainOrchestrator:
    """Main agent that orchestrates between specialized agents"""
    
    def __init__(self, data_agent: DataAnalysisAgent, optimization_agent: PriceOptimizationAgent, explanation_agent: ExplanationAgent):
        self.data_agent = data_agent
        self.optimization_agent = optimization_agent
        self.explanation_agent = explanation_agent
        self.llm = init_llm()
    
    def process_query(self, query: str) -> tuple[str, str, str]:
        """Process query and determine which agent to use"""
        query_lower = query.lower()
        
        # Track handover reasoning
        handover_reason = ""
        current_agent = "orchestrator"
        
        # Data analysis queries
        if any(word in query_lower for word in ['data', 'historical', 'summary', 'elasticity', 'trend', 'analyze']):
            handover_reason = "Query involves historical data analysis"
            current_agent = "data_analyst"
            response = self.data_agent.analyze_historical_data(query)
        
        # Model creation queries
        elif any(word in query_lower for word in ['optimize', 'create model', 'build model', 'price optimization']):
            handover_reason = "Query requires price optimization modeling"
            current_agent = "optimizer"
            response = self._handle_model_creation(query)
        
        # Solution queries
        elif any(word in query_lower for word in ['solve', 'run', 'optimize model']):
            handover_reason = "Query requires model solving"
            current_agent = "optimizer"
            response = self._handle_solve_request(query)
        
        # Explanation queries
        elif any(word in query_lower for word in ['explain', 'interpret', 'what does it mean', 'analyze results']):
            handover_reason = "Query requires results explanation"
            current_agent = "explainer"
            response = self._handle_explanation_request(query)
        
        # General queries
        else:
            response = self._general_conversation(query)
        
        return response, current_agent, handover_reason
    
    def _handle_model_creation(self, query: str) -> str:
        """Handle model creation"""
        try:
            # Extract constraints from query
            constraints = {}
            if 'budget' in query.lower():
                constraints['max_total_price'] = 1000  # Example constraint
            
            model_info = self.optimization_agent.create_optimization_model(constraints)
            if 'error' in model_info:
                return f"Error creating model: {model_info['error']}"
            
            return f"""✅ **Price Optimization Model Created**

**Model ID**: {model_info['model_id']}
**Products**: {', '.join(model_info['products'])}
**Variables**: {len(model_info['variables'])} price variables
**Constraints**: {len(model_info['constraints'])} business constraints

The model uses historical elasticity data to optimize prices for maximum profit.
You can now solve it with 'solve {model_info['model_id']}'"""
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _handle_solve_request(self, query: str) -> str:
        """Handle model solving"""
        model_id = None
        for mid in self.optimization_agent.models.keys():
            if mid in query:
                model_id = mid
                break
        
        if not model_id and self.optimization_agent.models:
            model_id = list(self.optimization_agent.models.keys())[-1]
        
        if not model_id:
            return "No models available. Please create a model first."
        
        results = self.optimization_agent.solve_model(model_id)
        if 'error' in results:
            return f"Error solving model: {results['error']}"
        
        # Format results
        prices_text = "\n".join([f"- {product}: ${price:.2f}" 
                               for product, price in results['prices'].items()])
        
        return f"""🎯 **Model {model_id} Solved Successfully**

**Status**: {results['status']}
**Expected Total Profit**: ${results['objective_value']:,.2f}

**Optimal Prices**:
{prices_text}

Ask me to 'explain the results' for a detailed interpretation."""
    
    def _handle_explanation_request(self, query: str) -> str:
        """Handle explanation requests"""
        model_id = None
        for mid in self.optimization_agent.models.keys():
            if mid in query:
                model_id = mid
                break
        
        if not model_id and self.optimization_agent.models:
            model_id = list(self.optimization_agent.models.keys())[-1]
        
        if not model_id:
            return "No solved models available for explanation."
        
        return self.explanation_agent.explain_optimization_results(model_id)
    
    def _general_conversation(self, query: str) -> str:
        """Handle general conversation"""
        try:
            if self.llm:
                messages = [
                    SystemMessage(content="You are a price optimization expert. Help users with data analysis, modeling, and results interpretation."),
                    HumanMessage(content=query)
                ]
                response = self.llm.invoke(messages)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                return "I can help you with price optimization using historical data. Ask me about data analysis, creating models, or explaining results."
        except:
            return "I can help you with price optimization using historical data. Ask me about data analysis, creating models, or explaining results."

# Create LangGraph workflow
def create_agent_workflow(orchestrator: MainOrchestrator):
    """Create workflow showing agent handovers"""
    
    def router(state: AgentState) -> str:
        query = state["query"].lower()
        
        if any(word in query for word in ['data', 'historical', 'summary', 'elasticity']):
            return "data_analyst"
        elif any(word in query for word in ['optimize', 'create model', 'build model']):
            return "optimizer"
        elif any(word in query for word in ['solve', 'run model']):
            return "solver"
        elif any(word in query for word in ['explain', 'interpret', 'analyze results']):
            return "explainer"
        else:
            return "orchestrator"
    
    def data_analyst_node(state: AgentState) -> AgentState:
        response, agent, reason = orchestrator.process_query(state["query"])
        return {**state, "response": response, "current_agent": agent, "handover_reason": reason}
    
    def optimizer_node(state: AgentState) -> AgentState:
        response, agent, reason = orchestrator.process_query(state["query"])
        return {**state, "response": response, "current_agent": agent, "handover_reason": reason}
    
    def solver_node(state: AgentState) -> AgentState:
        response, agent, reason = orchestrator.process_query(state["query"])
        return {**state, "response": response, "current_agent": agent, "handover_reason": reason}
    
    def explainer_node(state: AgentState) -> AgentState:
        response, agent, reason = orchestrator.process_query(state["query"])
        return {**state, "response": response, "current_agent": agent, "handover_reason": reason}
    
    def orchestrator_node(state: AgentState) -> AgentState:
        response, agent, reason = orchestrator.process_query(state["query"])
        return {**state, "response": response, "current_agent": agent, "handover_reason": reason}
    
    # Build workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("optimizer", optimizer_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("explainer", explainer_node)
    workflow.add_node("orchestrator", orchestrator_node)
    
    # Add routing
    workflow.add_conditional_edges(
        START,
        router,
        {
            "data_analyst": "data_analyst",
            "optimizer": "optimizer",
            "solver": "solver",
            "explainer": "explainer",
            "orchestrator": "orchestrator"
        }
    )
    
    workflow.add_edge("data_analyst", END)
    workflow.add_edge("optimizer", END)
    workflow.add_edge("solver", END)
    workflow.add_edge("explainer", END)
    workflow.add_edge("orchestrator", END)
    
    return workflow.compile()

# Streamlit UI
def main():
    st.title("💰 Multi-Agent Price Optimization System")
    st.markdown("Historical Data → MCP Server → Multi-Agent Analysis → Optimization → Explanation")
    
    # Initialize services
    mcp_client = init_mcp_client()
    data_agent = DataAnalysisAgent(mcp_client)
    optimization_agent = PriceOptimizationAgent(mcp_client)
    explanation_agent = ExplanationAgent(optimization_agent)
    orchestrator = MainOrchestrator(data_agent, optimization_agent, explanation_agent)
    
    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = create_agent_workflow(orchestrator)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'agent_handovers' not in st.session_state:
        st.session_state.agent_handovers = []
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # MCP Server status
        mcp_status = mcp_client.health_check()
        if mcp_status:
            st.success("✅ MCP Server Connected")
            # Show data summary
            summary = mcp_client.query("summary")
            st.write(f"**Products**: {summary.get('total_products', 'N/A')}")
            st.write(f"**Days of Data**: {summary.get('total_days', 'N/A')}")
        else:
            st.error("❌ MCP Server Offline")
            st.info("Run: `python mcp_server.py` in another terminal")
        
        # Ollama status
        llm = init_llm()
        if llm:
            st.success("✅ Ollama Connected")
        else:
            st.warning("⚠️ Ollama Offline - Using rule-based explanations")
        
        st.header("📊 Agent Handovers")
        if st.session_state.agent_handovers:
            for i, (query, agent, reason) in enumerate(st.session_state.agent_handovers[-5:]):  # Last 5
                with st.expander(f"Step {i+1}: {agent}"):
                    st.write(f"**Query**: {query}")
                    st.write(f"**Agent**: {agent}")
                    st.write(f"**Reason**: {reason}")
        else:
            st.write("No handovers yet")
        
        if st.button("Clear History"):
            st.session_state.conversation = []
            st.session_state.agent_handovers = []
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Price Optimization Chat")
        
        # Display conversation with agent indicators
        for i, (query, response, agent, reason) in enumerate(st.session_state.conversation):
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                # Show which agent handled this
                st.caption(f"🤖 Handled by: {agent}")
                if reason:
                    st.caption(f"📋 Reason: {reason}")
                
                st.markdown(response)
                
                # Add visual indicators based on agent type
                if agent == "data_analyst":
                    st.success("📊 Data Analysis Complete")
                elif agent == "optimizer":
                    st.success("⚡ Optimization Model Ready")
                elif agent == "solver":
                    st.success("🎯 Model Solved")
                elif agent == "explainer":
                    st.info("💡 Explanation Provided")
        
        # Chat input
        if prompt := st.chat_input("Ask about historical data, create models, or explain results..."):
            # Process through workflow
            initial_state = AgentState(
                messages=[],
                current_model=None,
                model_data=None,
                query=prompt,
                response="",
                status="processing",
                explanation="",
                current_agent="",
                handover_reason=""
            )
            
            try:
                result = st.session_state.workflow.invoke(initial_state)
                response = result["response"]
                current_agent = result["current_agent"]
                handover_reason = result["handover_reason"]
                
                # Store conversation and handover
                st.session_state.conversation.append((prompt, response, current_agent, handover_reason))
                st.session_state.agent_handovers.append((prompt, current_agent, handover_reason))
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.conversation.append((prompt, error_msg, "error", "Processing failed"))
                st.rerun()
    
    with col2:
        st.subheader("🚀 Quick Actions")
        
        st.info("Try these example workflows:")
        
        # Example workflows
        workflow_steps = [
            ("1. Analyze Historical Data", "Show me summary of historical sales data"),
            ("2. Check Price Elasticity", "What are the price elasticity estimates?"),
            ("3. Create Optimization Model", "Create a price optimization model"),
            ("4. Solve the Model", "Solve the latest optimization model"),
            ("5. Explain Results", "Explain the optimization results")
        ]
        
        for label, query in workflow_steps:
            if st.button(label, use_container_width=True):
                st.session_state.conversation.append((query, "", "pending", ""))
                
                initial_state = AgentState(
                    messages=[],
                    current_model=None,
                    model_data=None,
                    query=query,
                    response="",
                    status="processing",
                    explanation="",
                    current_agent="",
                    handover_reason=""
                )
                
                try:
                    result = st.session_state.workflow.invoke(initial_state)
                    response = result["response"]
                    current_agent = result["current_agent"]
                    handover_reason = result["handover_reason"]
                    
                    st.session_state.conversation[-1] = (query, response, current_agent, handover_reason)
                    st.session_state.agent_handovers.append((query, current_agent, handover_reason))
                    
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.conversation[-1] = (query, error_msg, "error", "Processing failed")
                    st.rerun()
        
        st.subheader("📈 Model Results")
        
        if optimization_agent.models:
            for model_id, model in optimization_agent.models.items():
                with st.expander(f"Model: {model_id}"):
                    st.write(f"Status: {model.get('status', 'created')}")
                    st.write(f"Products: {len(model.get('products', []))}")
                    
                    if 'results' in model:
                        results = model['results']
                        st.write(f"Total Profit: ${results.get('objective_value', 0):,.2f}")
                        
                        # Show price changes
                        for product, price in results.get('prices', {}).items():
                            current_avg = model['price_ranges'][product]['current_avg']
                            change_pct = ((price - current_avg) / current_avg) * 100
                            st.write(f"{product}: ${price:.2f} ({change_pct:+.1f}%)")

if __name__ == "__main__":
    # Check if MCP server is running, if not show instructions
    mcp_client = MCPClient()
    if not mcp_client.health_check():
        st.warning("""
        **MCP Server Not Detected**
        
        To run the complete system:
        
        1. Generate historical data:
        ```bash
        python data_creator.py
        ```
        
        2. Start MCP server (in separate terminal):
        ```bash
        python mcp_server.py
        ```
        
        3. Then refresh this page
        """)
    
    main()