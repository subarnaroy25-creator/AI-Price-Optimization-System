🤖 AI Price Optimization System

A multi-agent system that analyzes historical data, creates linear programming models for price optimization, and provides natural language explanations of the results.
🌟 Features
🤖 Multi-Agent Architecture

    Data Analysis Agent: Analyzes historical sales data and calculates price elasticity

    Optimization Agent: Creates and solves linear programming models for price optimization

    Explanation Agent: Provides natural language explanations of optimization results

    Orchestrator Agent: Routes queries to appropriate specialists

💰 Core Capabilities

    Historical Data Analysis: Summary statistics, price elasticity calculations, trend analysis

    Price Optimization: Linear programming models to maximize profit while considering price elasticity

    Natural Language Interface: Chat-based interaction with the AI system

    Visual Agent Handovers: See which agent handles each query and why

📊 Technical Features

    Streamlit Web Interface: Modern, interactive web application

    LangGraph Workflow Management: Intelligent routing between specialized agents

    Ollama Integration: Local LLM for sophisticated natural language processing

    PuLP Solver: Mathematical optimization using linear programming

    SQLite Database: Efficient historical data storage and retrieval

🚀 Quick Start
Prerequisites

    Python 3.8+

    Ollama (for LLM capabilities)

    Required Python packages

Installation

    Clone or download the project

bash

git clone <repository-url>
cd price-optimization-system

    Install dependencies

bash

pip install -r requirements.txt

    Set up Ollama (optional but recommended)

bash

# Install Ollama from https://ollama.ai/
ollama pull llama3.1  # or any other model you prefer

    Run the application

bash

streamlit run price_optimization_app.py

The application will automatically:

    Generate sample historical data (if not exists)

    Start the web interface on http://localhost:8501

📖 Usage Guide
🏠 Home Interface

The application features a two-column layout:

    Left Column: Chat interface for natural language queries

    Right Column: Quick actions and model management

    Sidebar: System status, agent handovers, and model tracking

💬 Example Queries

Try these natural language commands:
Data Analysis

    "Show me historical data summary"

    "Analyze price elasticity"

    "What are the feasible price ranges?"

    "Show me recent trends"

Model Operations

    "Create a price optimization model"

    "Solve the optimization model"

    "Explain the optimization results"

    "What models do I have?"

🔄 Workflow Example

    Data Analysis: "Show me historical data summary"

        Handled by: Data Analysis Agent 📊

        Output: Summary statistics and data overview

    Model Creation: "Create price optimization model"

        Handled by: Optimization Agent ⚡

        Output: Model created with ID and product list

    Model Solving: "Solve the model"

        Handled by: Optimization Agent ⚡

        Output: Optimal prices and expected profit

    Results Explanation: "Explain the results"

        Handled by: Explanation Agent 💡

        Output: Business insights and implementation guidance

🏗️ System Architecture
Agent Responsibilities
Data Analysis Agent 📊

    Analyzes historical sales data

    Calculates price elasticity coefficients

    Provides statistical summaries

    Identifies trends and patterns

Optimization Agent ⚡

    Creates linear programming models

    Defines objective function (profit maximization)

    Sets constraints (price ranges, margins)

    Solves optimization problems

Explanation Agent 💡

    Interprets optimization results

    Provides business recommendations

    Explains pricing strategies

    Suggests implementation approaches

Orchestrator Agent 🤖

    Routes queries to appropriate agents

    Manages workflow between agents

    Provides fallback responses

Data Flow

    Historical Data → Data Analysis Agent → Insights

    Insights + Constraints → Optimization Agent → Optimal Prices

    Optimal Prices + Business Context → Explanation Agent → Recommendations

🔧 Configuration
Model Parameters

The system uses reasonable defaults but can be customized:

    Price Elasticity: Calculated from historical data

    Price Ranges: ±30% from current prices with cost-based minimums

    Margin Constraints: Minimum 5-10% profit margins

    Optimization Objective: Maximize total profit

Ollama Settings

Edit the init_llm() function to use different models:
python

# Available models: llama3.1, mistral, codellama, etc.
return Ollama(model="llama3.1")

📈 Sample Output
Optimization Results Example
text

🎯 Model price_model_0 Solved Successfully!

Status: Optimal
Expected Profit: $45,250.00

Optimal Prices (change from current):
- Product_A: $115.50 (+15.5%)
- Product_B: $162.75 (+8.5%)
- Product_C: $185.20 (-7.4%)
- Product_D: $88.90 (+11.1%)
- Product_E: $135.60 (+13.0%)

Explanation Example
text

💰 Price Optimization Results Analysis

Executive Summary:
- Expected Profit Increase: $45,250
- 3 products recommended for price increases
- 2 products recommended for price decreases

Key Insights:
- Product_A shows high elasticity - sensitive to price changes
- Product_C has limited competition - can sustain higher prices
- Overall strategy balances margin improvement with volume retention

🐛 Troubleshooting
Common Issues

Ollama Connection Failed
text

Ollama not available - using rule-based explanations

    Solution: Install Ollama or check if it's running

No Models Displayed
text

No models created yet

    Solution: Use "Create price optimization model" command first

Optimization Errors
text

Model creation failed: unsupported operand type(s)

    Solution: The system will automatically use simplified models

Debug Mode

Enable debug information in the sidebar to see:

    Agent handover details

    Model creation steps

    Data retrieval status

🔮 Future Enhancements
Planned Features

    Competitor Price Integration: Incorporate competitor pricing data

    Seasonal Adjustments: Account for seasonal demand patterns

    Multiple Objective Functions: Balance profit, market share, etc.

    Sensitivity Analysis: Show how results change with different assumptions

    Export Capabilities: Download optimization results as CSV/PDF

    API Endpoints: REST API for integration with other systems

Technical Improvements

    Database Backends: Support for PostgreSQL, MySQL

    Caching: Improved performance for large datasets

    Authentication: User management and access controls

    Deployment: Docker containerization

🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines for details.
Development Setup
bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    PuLP: Linear programming solver

    Streamlit: Web application framework

    LangGraph: Multi-agent workflow management

    Ollama: Local LLM infrastructure

    Pandas & NumPy: Data analysis libraries

📞 Support

For questions or issues:

    Check the troubleshooting section above

    Review the agent handovers in the sidebar for debugging

    Open an issue on GitHub with detailed description

Note: This is a demonstration system for educational purposes. Always validate optimization results with domain experts before implementing pricing changes in production environments.
