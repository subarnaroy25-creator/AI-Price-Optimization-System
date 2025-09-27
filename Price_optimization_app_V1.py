import streamlit as st
import pulp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import sqlite3
import os

# Set page config
st.set_page_config(
    page_title="Price Optimization System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Ollama
@st.cache_resource
def init_llm():
    try:
        return Ollama(model="llama3.1")
    except Exception as e:
        st.warning(f"Ollama not available: {e}")
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

class DataManager:
    """Manages historical data directly without MCP server"""
    
    def __init__(self):
        self.data = None
        self.db_path = "price_optimization.db"
        self._ensure_data_exists()
    
    def _ensure_data_exists(self):
        """Generate data if it doesn't exist"""
        if not os.path.exists(self.db_path):
            st.info("📊 Generating historical price data...")
            self._generate_historical_data()
        self._load_data()
    
    def _generate_historical_data(self):
        """Generate synthetic historical sales data"""
        np.random.seed(42)
        
        products_list = [f"Product_{chr(65+i)}" for i in range(5)]
        base_prices = [100, 150, 200, 80, 120]
        cost_prices = [60, 90, 120, 50, 80]
        
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
        
        data = []
        for date in dates:
            for i, product in enumerate(products_list):
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                current_price = float(base_prices[i] * (0.8 + 0.4 * np.random.random()))
                price_ratio = current_price / base_prices[i]
                elasticity_effect = max(0.1, 1.2 - 0.8 * price_ratio)
                noise = 0.9 + 0.2 * np.random.random()
                
                base_demand = 1000 * (i + 1)
                demand = int(base_demand * seasonal_factor * elasticity_effect * noise)
                revenue = demand * current_price
                profit = demand * (current_price - cost_prices[i])
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product': product,
                    'price': current_price,
                    'demand': max(10, demand),
                    'revenue': revenue,
                    'cost_price': cost_prices[i],
                    'profit': profit,
                    'weekday': date.weekday(),
                    'month': date.month,
                })
        
        self.data = pd.DataFrame(data)
        
        # Save to SQLite
        conn = sqlite3.connect(self.db_path)
        self.data.to_sql('historical_sales', conn, if_exists='replace', index=False)
        conn.close()
        
        st.success("✅ Historical data generated successfully!")
    
    def _load_data(self):
        """Load data from SQLite"""
        conn = sqlite3.connect(self.db_path)
        self.data = pd.read_sql("SELECT * FROM historical_sales", conn)
        conn.close()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_products": int(self.data['product'].nunique()),
            "total_days": int(self.data['date'].nunique()),
            "total_revenue": float(self.data['revenue'].sum()),
            "total_profit": float(self.data['profit'].sum()),
            "avg_price": float(self.data['price'].mean()),
            "avg_demand": float(self.data['demand'].mean())
        }
    
    def get_elasticity_data(self) -> Dict[str, Any]:
        """Calculate price elasticity estimates using linear approximation"""
        elasticity_data = {}
        
        for product in self.data['product'].unique():
            product_df = self.data[self.data['product'] == product].sort_values('date')
            if len(product_df) > 10:
                # Use linear regression for elasticity approximation
                X = product_df['price'].values.reshape(-1, 1)
                y = product_df['demand'].values
                
                # Simple linear approximation: demand = a + b*price
                price_mean = X.mean()
                demand_mean = y.mean()
                
                # Calculate slope (simplified)
                if len(X) > 1:
                    # Use correlation-based approximation
                    correlation = np.corrcoef(X.flatten(), y)[0, 1]
                    if not np.isnan(correlation):
                        # Elasticity ≈ correlation * (price_mean/demand_mean)
                        elasticity = correlation * (price_mean / demand_mean) * 10  # Scale factor
                    else:
                        elasticity = -0.5  # Default slightly elastic
                else:
                    elasticity = -0.5
            else:
                elasticity = -0.5  # Default elasticity
            
            elasticity_data[product] = {
                'elasticity': float(elasticity),
                'avg_price': float(product_df['price'].mean()),
                'avg_demand': float(product_df['demand'].mean()),
                'max_demand': float(product_df['demand'].max()),
                'min_demand': float(product_df['demand'].min())
            }
        
        return elasticity_data
    
    def get_price_ranges(self) -> Dict[str, Any]:
        """Get feasible price ranges"""
        ranges = {}
        
        for product in self.data['product'].unique():
            product_df = self.data[self.data['product'] == product]
            cost_price = float(product_df['cost_price'].iloc[0])
            current_avg = float(product_df['price'].mean())
            price_std = float(product_df['price'].std())
            
            ranges[product] = {
                'min_price': float(max(cost_price * 1.05, current_avg * 0.7)),
                'max_price': float(current_avg * 1.3),
                'current_avg': current_avg,
                'cost_price': cost_price,
                'price_std': price_std
            }
        
        return ranges
    
    def get_products(self) -> List[str]:
        """Get list of products"""
        return self.data['product'].unique().tolist()

class DataAnalysisAgent:
    """Agent specialized in analyzing historical data"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    def analyze_historical_data(self, query: str) -> str:
        """Analyze historical data based on query"""
        try:
            query_lower = query.lower()
            
            if "summary" in query_lower or "overview" in query_lower:
                result = self.data_manager.get_summary_stats()
                return self._format_summary(result)
            elif "elasticity" in query_lower:
                result = self.data_manager.get_elasticity_data()
                return self._format_elasticity(result)
            elif "price range" in query_lower or "feasible" in query_lower:
                result = self.data_manager.get_price_ranges()
                return self._format_price_ranges(result)
            else:
                summary = self.data_manager.get_summary_stats()
                elasticity = self.data_manager.get_elasticity_data()
                return self._format_comprehensive_analysis(summary, elasticity)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
    
    def _format_summary(self, summary: Dict) -> str:
        return f"""📊 **Historical Data Summary**

- **Products Analyzed**: {summary['total_products']}
- **Time Period**: {summary['total_days']} days
- **Total Revenue**: ${summary['total_revenue']:,.2f}
- **Total Profit**: ${summary['total_profit']:,.2f}
- **Average Price**: ${summary['avg_price']:.2f}
- **Average Demand**: {summary['avg_demand']:.0f} units/day

Ready for price optimization modeling!"""
    
    def _format_elasticity(self, elasticity: Dict) -> str:
        analysis = "📈 **Price Elasticity Analysis**\n\n"
        for product, stats in elasticity.items():
            elast = stats['elasticity']
            analysis += f"**{product}**: Elasticity = {elast:.3f}\n"
            if elast < -1.0:
                analysis += "  - 🎯 Highly elastic (very price sensitive)\n"
            elif elast < -0.5:
                analysis += "  - ⚖️ Moderately elastic\n"
            else:
                analysis += "  - 💪 Inelastic (less price sensitive)\n"
            analysis += f"  - Avg Price: ${stats['avg_price']:.2f}\n"
            analysis += f"  - Avg Demand: {stats['avg_demand']:.0f} units\n\n"
        
        return analysis
    
    def _format_price_ranges(self, ranges: Dict) -> str:
        analysis = "💰 **Feasible Price Ranges**\n\n"
        for product, stats in ranges.items():
            analysis += f"**{product}**:\n"
            analysis += f"  - Minimum: ${stats['min_price']:.2f}\n"
            analysis += f"  - Maximum: ${stats['max_price']:.2f}\n"
            analysis += f"  - Current: ${stats['current_avg']:.2f}\n"
            analysis += f"  - Cost: ${stats['cost_price']:.2f}\n"
            analysis += f"  - Margin: {((stats['current_avg'] - stats['cost_price']) / stats['cost_price'] * 100):.1f}%\n\n"
        
        return analysis
    
    def _format_comprehensive_analysis(self, summary: Dict, elasticity: Dict) -> str:
        return f"""🔍 **Comprehensive Data Analysis**

{self._format_summary(summary)}

{self._format_elasticity(elasticity)}

The data shows clear patterns for effective price optimization!"""

class PriceOptimizationAgent:
    """Agent for creating and solving price optimization models using LINEAR programming"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.models = {}
        self.model_counter = 0
    
    def create_optimization_model(self, constraints: Dict = None) -> Dict[str, Any]:
        """Create LINEAR price optimization model"""
        try:
            price_ranges = self.data_manager.get_price_ranges()
            elasticity_data = self.data_manager.get_elasticity_data()
            
            model_id = f"price_model_{self.model_counter}"
            self.model_counter += 1
            
            # Create LINEAR optimization problem
            prob = pulp.LpProblem("Price_Optimization", pulp.LpMaximize)
            variables = {}
            
            # Create variables for prices and demands (both linear)
            price_vars = {}
            demand_vars = {}
            revenue_vars = {}
            
            for product, data in price_ranges.items():
                # Price variable
                price_var_name = f"price_{product}"
                price_vars[product] = pulp.LpVariable(
                    price_var_name, 
                    lowBound=data['min_price'], 
                    upBound=data['max_price'],
                    cat='Continuous'
                )
                
                # Demand variable with linear elasticity approximation
                elasticity = elasticity_data[product]['elasticity']
                avg_demand = elasticity_data[product]['avg_demand']
                avg_price = data['current_avg']
                
                # Linear demand function: demand = base_demand + slope * (price - avg_price)
                # Where slope = elasticity * (avg_demand / avg_price)
                slope = elasticity * (avg_demand / avg_price)
                
                demand_var_name = f"demand_{product}"
                # Demand constraints based on linear approximation
                min_demand = avg_demand + slope * (data['min_price'] - avg_price)
                max_demand = avg_demand + slope * (data['max_price'] - avg_price)
                
                demand_vars[product] = pulp.LpVariable(
                    demand_var_name,
                    lowBound=max(0, min(min_demand, max_demand)),
                    upBound=max(min_demand, max_demand),
                    cat='Continuous'
                )
                
                # Revenue variable
                revenue_var_name = f"revenue_{product}"
                revenue_vars[product] = pulp.LpVariable(
                    revenue_var_name,
                    lowBound=0,
                    cat='Continuous'
                )
            
            # OBJECTIVE: Maximize total profit (revenue - cost)
            profit_terms = []
            for product in price_ranges.keys():
                cost = price_ranges[product]['cost_price']
                # Profit = revenue - cost * demand = price * demand - cost * demand
                profit_expr = revenue_vars[product] - cost * demand_vars[product]
                profit_terms.append(profit_expr)
            
            prob += pulp.lpSum(profit_terms), "Total_Profit"
            
            # CONSTRAINTS
            
            # 1. Revenue definition: revenue = price * demand
            for product in price_ranges.keys():
                # This is nonlinear, so we need to linearize or use approximation
                # For simplicity, we'll use bounds and linear relationships
                prob += revenue_vars[product] <= price_vars[product] * demand_vars[product].upBound, f"Max_Revenue_{product}"
                prob += revenue_vars[product] >= price_vars[product] * demand_vars[product].lowBound, f"Min_Revenue_{product}"
            
            # 2. Linear demand relationship
            for product in price_ranges.keys():
                elasticity = elasticity_data[product]['elasticity']
                avg_demand = elasticity_data[product]['avg_demand']
                avg_price = price_ranges[product]['current_avg']
                slope = elasticity * (avg_demand / avg_price)
                
                # Demand = avg_demand + slope * (price - avg_price)
                prob += demand_vars[product] == avg_demand + slope * (price_vars[product] - avg_price), f"Demand_Function_{product}"
            
            # 3. Business constraints
            if constraints and 'max_total_price' in constraints:
                prob += pulp.lpSum([price_vars[product] for product in price_ranges.keys()]) <= constraints['max_total_price'], "Total_Price_Limit"
            
            for product in price_ranges.keys():
                # Minimum profit margin constraint
                min_margin = 0.05  # 5% minimum margin
                prob += price_vars[product] >= price_ranges[product]['cost_price'] * (1 + min_margin), f"Min_Margin_{product}"
                
                # Demand non-negativity
                prob += demand_vars[product] >= 0, f"Non_Neg_Demand_{product}"
            
            model_info = {
                'model_id': model_id,
                'pulp_problem': prob,
                'price_variables': price_vars,
                'demand_variables': demand_vars,
                'revenue_variables': revenue_vars,
                'products': list(price_ranges.keys()),
                'price_ranges': price_ranges,
                'elasticity_data': elasticity_data,
                'status': 'created',
                'constraints': constraints or {},
                'model_type': 'linear_approximation'
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
                'objective_value': pulp.value(prob.objective) or 0,
                'prices': {},
                'demands': {},
                'revenues': {},
                'profits': {}
            }
            
            model_info = self.models[model_id]
            
            # Extract results
            for product in model_info['products']:
                price_var = model_info['price_variables'][product]
                demand_var = model_info['demand_variables'][product]
                revenue_var = model_info['revenue_variables'][product]
                
                optimal_price = price_var.varValue if price_var.varValue else model_info['price_ranges'][product]['current_avg']
                optimal_demand = demand_var.varValue if demand_var.varValue else model_info['elasticity_data'][product]['avg_demand']
                optimal_revenue = revenue_var.varValue if revenue_var.varValue else optimal_price * optimal_demand
                
                results['prices'][product] = optimal_price
                results['demands'][product] = optimal_demand
                results['revenues'][product] = optimal_revenue
                results['profits'][product] = optimal_revenue - (model_info['price_ranges'][product]['cost_price'] * optimal_demand)
            
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
        
        if self.llm:
            return self._explain_with_llm(model, results)
        else:
            return self._explain_with_rules(model, results)
    
    def _explain_with_llm(self, model: Dict, results: Dict) -> str:
        """Use LLM for natural language explanation"""
        try:
            prompt = f"""
            Explain these price optimization results in simple business terms:
            
            Optimization Goal: Maximize total profit using linear programming
            Status: {results['status']}
            Expected Profit: ${results['objective_value']:,.2f}
            
            Optimal Prices:
            {chr(10).join([f'{product}: ${price:.2f}' for product, price in results['prices'].items()])}
            
            Please explain what these results mean and suggest implementation strategy.
            """
            
            messages = [
                SystemMessage(content="You are a pricing expert explaining optimization results to business managers."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return self._explain_with_rules(model, results)
    
    def _explain_with_rules(self, model: Dict, results: Dict) -> str:
        """Rule-based explanation"""
        explanation = "## 💰 Linear Price Optimization Results\n\n"
        
        explanation += f"**Status**: {results['status']}\n"
        explanation += f"**Expected Profit**: ${results['objective_value']:,.2f}\n\n"
        
        explanation += "### 🎯 Recommended Prices:\n"
        total_profit = 0
        for product, price in results['prices'].items():
            current_avg = model['price_ranges'][product]['current_avg']
            change_pct = ((price - current_avg) / current_avg) * 100
            profit = results['profits'][product]
            total_profit += profit
            
            explanation += f"- **{product}**: ${price:.2f} "
            explanation += f"({change_pct:+.1f}% from ${current_avg:.2f})\n"
            explanation += f"  - Expected demand: {results['demands'][product]:.0f} units\n"
            explanation += f"  - Expected profit: ${profit:,.0f}\n"
        
        explanation += f"\n**Total Expected Profit**: ${total_profit:,.0f}\n\n"
        
        explanation += "### 📈 Optimization Approach:\n"
        explanation += "- Uses linear programming with price elasticity approximation\n"
        explanation += "- Balances price increases with expected demand changes\n"
        explanation += "- Ensures minimum profit margins are maintained\n"
        
        explanation += "\n### 💡 Implementation Strategy:\n"
        explanation += "- Implement price changes gradually\n"
        explanation += "- Monitor actual vs. predicted demand closely\n"
        explanation += "- Adjust based on market response and competitor actions\n"
        
        return explanation

class MainOrchestrator:
    """Main agent that orchestrates between specialized agents"""
    
    def __init__(self, data_agent: DataAnalysisAgent, optimization_agent: PriceOptimizationAgent, explanation_agent: ExplanationAgent):
        self.data_agent = data_agent
        self.optimization_agent = optimization_agent
        self.explanation_agent = explanation_agent
    
    def process_query(self, query: str) -> tuple[str, str, str]:
        """Process query and determine which agent to use"""
        query_lower = query.lower()
        
        handover_reason = ""
        current_agent = "orchestrator"
        
        if any(word in query_lower for word in ['data', 'historical', 'summary', 'elasticity', 'trend', 'analyze']):
            handover_reason = "Query involves historical data analysis"
            current_agent = "data_analyst"
            response = self.data_agent.analyze_historical_data(query)
        
        elif any(word in query_lower for word in ['optimize', 'create model', 'build model', 'price optimization']):
            handover_reason = "Query requires price optimization modeling"
            current_agent = "optimizer"
            response = self._handle_model_creation(query)
        
        elif any(word in query_lower for word in ['solve', 'run', 'optimize model']):
            handover_reason = "Query requires model solving"
            current_agent = "optimizer"
            response = self._handle_solve_request(query)
        
        elif any(word in query_lower for word in ['explain', 'interpret', 'what does it mean', 'analyze results']):
            handover_reason = "Query requires results explanation"
            current_agent = "explainer"
            response = self._handle_explanation_request(query)
        
        else:
            response = self._general_conversation(query)
        
        return response, current_agent, handover_reason
    
    def _handle_model_creation(self, query: str) -> str:
        """Handle model creation"""
        try:
            constraints = {}
            if 'budget' in query.lower():
                constraints['max_total_price'] = 500
            
            model_info = self.optimization_agent.create_optimization_model(constraints)
            if 'error' in model_info:
                return f"Error creating model: {model_info['error']}"
            
            return f"""✅ **Linear Price Optimization Model Created**

**Model ID**: {model_info['model_id']}
**Products**: {', '.join(model_info['products'])}
**Model Type**: Linear programming with elasticity approximation
**Variables**: {len(model_info['price_variables'])} price variables + {len(model_info['demand_variables'])} demand variables

The model uses linear approximations of price elasticity to maximize profit.
Use 'solve {model_info['model_id']}' to find optimal prices."""
            
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
        
        prices_text = "\n".join([f"- {product}: ${price:.2f}" 
                               for product, price in results['prices'].items()])
        
        return f"""🎯 **Model {model_id} Solved Successfully**

**Status**: {results['status']}
**Expected Profit**: ${results['objective_value']:,.2f}

**Optimal Prices**:
{prices_text}

**Total Expected Profit**: ${results['objective_value']:,.0f}

Ask 'explain the results' for detailed interpretation."""
    
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
        return """I'm your price optimization assistant! I can help you:

📊 **Analyze Historical Data** - Understand past sales patterns
⚡ **Create Linear Optimization Models** - Build mathematical pricing models  
🎯 **Solve for Optimal Prices** - Find profit-maximizing prices
💡 **Explain Results** - Understand the business implications

Try asking me to:
- "Show me historical data summary"
- "Create a price optimization model" 
- "Solve the latest model"
- "Explain the optimization results"
"""

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
    st.title("💰 Linear Price Optimization System")
    st.markdown("**Multi-Agent System: Data Analysis → Linear Optimization → Explanation**")
    
    # Initialize all components
    data_manager = DataManager()
    data_agent = DataAnalysisAgent(data_manager)
    optimization_agent = PriceOptimizationAgent(data_manager)
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
        
        # Data status
        summary = data_manager.get_summary_stats()
        st.success("✅ Data Manager Ready")
        st.write(f"**Products**: {summary['total_products']}")
        st.write(f"**Days of Data**: {summary['total_days']}")
        st.write(f"**Total Revenue**: ${summary['total_revenue']:,.0f}")
        
        # Ollama status
        llm = init_llm()
        if llm:
            st.success("✅ Ollama Connected")
        else:
            st.warning("⚠️ Ollama Offline - Using rule-based explanations")
        
        st.header("🤖 Agent Handovers")
        if st.session_state.agent_handovers:
            for i, (query, agent, reason) in enumerate(st.session_state.agent_handovers[-5:]):
                with st.expander(f"Step {i+1}: {agent}"):
                    st.write(f"**Query**: {query[:50]}...")
                    st.write(f"**Agent**: {agent}")
                    st.write(f"**Reason**: {reason}")
        else:
            st.write("No handovers yet")
        
        if st.button("🔄 Clear History"):
            st.session_state.conversation = []
            st.session_state.agent_handovers = []
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Optimization Chat")
        
        # Display conversation
        for i, (query, response, agent, reason) in enumerate(st.session_state.conversation):
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                # Agent indicator
                agent_icons = {
                    "data_analyst": "📊",
                    "optimizer": "⚡", 
                    "solver": "🎯",
                    "explainer": "💡",
                    "orchestrator": "🤖"
                }
                st.caption(f"{agent_icons.get(agent, '🤖')} Handled by: {agent}")
                if reason:
                    st.caption(f"📋 {reason}")
                
                st.markdown(response)
        
        # Chat input
        if prompt := st.chat_input("Ask about data, create models, or explain results..."):
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
        
        workflow_steps = [
            ("1. Data Analysis", "Show me historical data summary"),
            ("2. Price Elasticity", "Analyze price elasticity"),
            ("3. Create Model", "Create price optimization model"),
            ("4. Solve Model", "Solve the optimization model"),
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
        
        st.subheader("📈 Current Models")
        
        if optimization_agent.models:
            for model_id, model in optimization_agent.models.items():
                with st.expander(f"Model: {model_id}"):
                    st.write(f"Status: {model.get('status', 'created')}")
                    st.write(f"Products: {len(model.get('products', []))}")
                    st.write(f"Type: {model.get('model_type', 'N/A')}")
                    
                    if 'results' in model:
                        results = model['results']
                        st.write(f"Profit: ${results.get('objective_value', 0):,.0f}")
                        
                        # Show price changes
                        for product in model.get('products', [])[:3]:
                            if product in results.get('prices', {}):
                                price = results['prices'][product]
                                current = model['price_ranges'][product]['current_avg']
                                change = ((price - current) / current) * 100
                                st.write(f"{product}: ${price:.1f} ({change:+.1f}%)")
        else:
            st.write("No models created yet")

if __name__ == "__main__":
    main()