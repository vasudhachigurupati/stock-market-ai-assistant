import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import traceback
import os


load_dotenv()


st.set_page_config(
    page_title="Stock Analysis Agent",
    page_icon="📈",
    layout="wide"
)


st.title("🤖 Stock Analysis Assistant")
st.markdown("""
This app uses an AI agent to analyze stocks and provide detailed information using YFinance data and web search results.
""")

# Initialize session state for the agent if it doesn't exist
if 'agent' not in st.session_state:
    st.session_state.agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[
            YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
            DuckDuckGo()
        ],
        show_tool_calls=True,
        markdown=True,
        instructions=[
            "Use the table to display the data",
            "When comparing stocks, analyze one stock at a time",
            "Present data in a clear, structured format"
        ],
        
    )

# Create the input form
with st.form("query_form"):
    query = st.text_input("Enter your stock-related query:", placeholder="e.g., Give me the details of META")
    submit_button = st.form_submit_button("Get Analysis")

# Handle form submission
if submit_button and query:
    try:
        # Show a spinner while processing
        with st.spinner("Analyzing your request..."):
            # Get response from agent
            response = st.session_state.agent.run(query)
            
            # Extract the actual content from the response
            if hasattr(response, 'content'):
                display_content = response.content
            elif hasattr(response, 'messages'):
                # Get the last assistant message content
                assistant_messages = [msg for msg in response.messages if msg.role == 'assistant' and msg.content]
                if assistant_messages:
                    display_content = assistant_messages[-1].content
                else:
                    display_content = "No response content available"
            else:
                display_content = str(response)
            
        # Display response
        st.markdown("### Analysis Results")
        st.markdown(display_content)
        
        # Optionally display debug information in an expander
        with st.expander("Debug Information"):
            st.code(str(response))
        
    except Exception as e:
        # Display error message
        st.error(f"An error occurred: {str(e)}")
        # Display traceback in expander
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your stock-related query in the text box
2. Click "Get Analysis" to receive detailed information
3. The results will include stock data and relevant web information

**Tips:**
- For stock comparisons, try asking about one stock at a time
- Be specific in your queries for better results
- Check the Debug Information if you need more details

**Note:** The analysis may take a few moments to complete depending on the complexity of your query.
""")

# Add sidebar with additional information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool combines:
    - Real-time stock data from YFinance
    - Web search results from DuckDuckGo
    - AI analysis using Groq LLM
    
    Made with Streamlit and PHI framework.
    """)
