#!/usr/bin/env python3
"""
PRSM Tokenomics Dashboard
Interactive Streamlit dashboard for FTNS tokenomics simulation and analysis

Features:
- Real-time economic simulation visualization
- Stress testing scenario management
- Agent behavior analysis
- Economic fairness metrics
- Interactive parameter tuning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

try:
    from tokenomics_simulation import (
        FTNSEconomicSimulation, MarketCondition, AgentType,
        run_stress_test_scenarios
    )
    TOKENOMICS_AVAILABLE = True
except ImportError:
    TOKENOMICS_AVAILABLE = False
    st.error("Tokenomics simulation not available. Please ensure tokenomics_simulation.py is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="PRSM Tokenomics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
    margin-bottom: 1rem;
}
.success-metric { border-left-color: #28a745; }
.warning-metric { border-left-color: #ffc107; }
.danger-metric { border-left-color: #dc3545; }
.agent-card {
    background-color: #f8f9fa;
    padding: 0.8rem;
    border-radius: 0.3rem;
    border: 1px solid #dee2e6;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

class TokenomicsDashboard:
    """FTNS Tokenomics Dashboard Controller"""
    
    def __init__(self):
        self.simulation = None
        self.simulation_results = None
        self.stress_test_results = None
        
        # Initialize session state
        if 'simulation_run' not in st.session_state:
            st.session_state.simulation_run = False
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = None
        if 'stress_test_data' not in st.session_state:
            st.session_state.stress_test_data = None
    
    def create_simulation(self, num_agents: int, simulation_days: int, 
                         initial_supply: float) -> bool:
        """Create new economic simulation"""
        if TOKENOMICS_AVAILABLE:
            self.simulation = FTNSEconomicSimulation(
                num_agents=num_agents,
                simulation_days=simulation_days,
                initial_token_supply=initial_supply
            )
            return True
        return False
    
    def run_simulation(self, stress_scenarios=None):
        """Run the economic simulation"""
        if self.simulation:
            results = self.simulation.run_simulation(stress_scenarios)
            st.session_state.simulation_data = results
            st.session_state.simulation_run = True
            return results
        return None
    
    def run_stress_tests(self):
        """Run comprehensive stress tests"""
        if TOKENOMICS_AVAILABLE:
            results = run_stress_test_scenarios()
            st.session_state.stress_test_data = results
            return results
        return None

def render_economic_metrics(results):
    """Render key economic metrics"""
    if not results:
        return
    
    metrics = results['key_metrics']
    validation = results['validation_results']
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_delta = metrics['final_token_price'] - 1.0
        st.metric(
            "Final Token Price",
            f"${metrics['final_token_price']:.3f}",
            f"{price_delta:+.3f}"
        )
    
    with col2:
        gini_status = "🟢" if validation['wealth_distribution_fair'] else "🔴"
        st.metric(
            "Wealth Distribution",
            f"{metrics['final_gini_coefficient']:.3f} {gini_status}",
            "Fair" if validation['wealth_distribution_fair'] else "Unfair"
        )
    
    with col3:
        quality_status = "🟢" if validation['quality_maintained'] else "🔴"
        st.metric(
            "Average Quality",
            f"{metrics['average_quality']:.1%} {quality_status}",
            "Good" if validation['quality_maintained'] else "Poor"
        )
    
    with col4:
        stability_status = "🟢" if validation['price_stable'] else "🔴"
        st.metric(
            "Price Stability",
            f"{metrics['price_stability']:.1%} {stability_status}",
            "Stable" if validation['price_stable'] else "Volatile"
        )
    
    # Validation results summary
    st.subheader("🎯 Economic Validation")
    
    validation_col1, validation_col2 = st.columns(2)
    
    with validation_col1:
        st.markdown("**Validation Criteria:**")
        for criterion, passed in validation.items():
            icon = "✅" if passed else "❌"
            criterion_name = criterion.replace('_', ' ').title()
            st.write(f"{icon} {criterion_name}")
    
    with validation_col2:
        success_rate = sum(validation.values()) / len(validation)
        overall_status = "🟢 PASS" if results['overall_success'] else "🔴 FAIL"
        
        st.markdown("**Overall Assessment:**")
        st.write(f"Success Rate: {success_rate:.1%}")
        st.write(f"Status: {overall_status}")
        
        if results['overall_success']:
            st.success("🏆 Tokenomics system meets all validation criteria!")
        else:
            st.error("⚠️ Tokenomics system requires improvements")

def render_time_series_charts(results):
    """Render time series analysis charts"""
    if not results or not results.get('daily_metrics'):
        st.warning("No time series data available")
        return
    
    daily_data = pd.DataFrame(results['daily_metrics'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Token Price Evolution',
            'Wealth Distribution (Gini Coefficient)',
            'Network Activity & Quality',
            'Transaction Volume'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # 1. Token price evolution
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['token_price'],
            mode='lines+markers',
            name='Token Price',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # 2. Gini coefficient with threshold
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['gini_coefficient'],
            mode='lines+markers',
            name='Gini Coefficient',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=4)
        ),
        row=1, col=2
    )
    
    # Add fairness threshold
    fig.add_hline(
        y=0.7, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Fairness Threshold (0.7)",
        row=1, col=2
    )
    
    # 3. Activity rate and quality (dual y-axis)
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['activity_rate'],
            mode='lines+markers',
            name='Activity Rate',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=3)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['avg_contribution_quality'],
            mode='lines+markers',
            name='Avg Quality',
            line=dict(color='#d62728', width=2),
            marker=dict(size=3),
            yaxis='y4'
        ),
        row=2, col=1, secondary_y=True
    )
    
    # 4. Transaction volume
    fig.add_trace(
        go.Scatter(
            x=daily_data['day'],
            y=daily_data['total_transaction_volume'],
            mode='lines+markers',
            name='Transaction Volume',
            line=dict(color='#9467bd', width=3),
            marker=dict(size=4),
            fill='tonexty'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="📊 FTNS Tokenomics Time Series Analysis",
        showlegend=True,
        height=700,
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Simulation Day")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Gini Coefficient", row=1, col=2)
    fig.update_yaxes(title_text="Activity Rate", row=2, col=1)
    fig.update_yaxes(title_text="Quality Score", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Volume (FTNS)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def render_agent_analysis(results):
    """Render agent behavior and performance analysis"""
    if not results or not results.get('agent_analysis'):
        st.warning("No agent data available")
        return
    
    agent_data = results['agent_analysis']
    
    # Convert to DataFrame for analysis
    agent_df = pd.DataFrame([
        {
            'agent_id': agent_id,
            **data
        }
        for agent_id, data in agent_data.items()
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Agent Type Distribution")
        
        # Agent type summary
        type_summary = agent_df.groupby('type').agg({
            'final_balance': ['mean', 'median', 'std'],
            'roi': ['mean', 'median'],
            'reputation': ['mean'],
            'contributions': ['sum']
        }).round(3)
        
        # Flatten column names
        type_summary.columns = ['_'.join(col).strip() for col in type_summary.columns]
        
        st.dataframe(type_summary, use_container_width=True)
        
        # ROI by agent type
        fig_roi = px.box(
            agent_df, 
            x='type', 
            y='roi',
            title='Return on Investment by Agent Type',
            color='type'
        )
        fig_roi.update_xaxes(title='Agent Type')
        fig_roi.update_yaxes(title='ROI')
        fig_roi.update_layout(showlegend=False)
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col2:
        st.subheader("💰 Wealth Distribution")
        
        # Wealth distribution histogram
        fig_wealth = px.histogram(
            agent_df,
            x='final_balance',
            nbins=20,
            title='Final Balance Distribution',
            color='type',
            marginal='rug'
        )
        fig_wealth.update_xaxes(title='Final Balance (FTNS)')
        fig_wealth.update_yaxes(title='Number of Agents')
        st.plotly_chart(fig_wealth, use_container_width=True)
        
        # Top and bottom performers
        st.markdown("**🏆 Top Performers (by ROI):**")
        top_performers = agent_df.nlargest(5, 'roi')[['agent_id', 'type', 'roi', 'final_balance']]
        for _, agent in top_performers.iterrows():
            st.markdown(
                f"<div class='agent-card'>"
                f"<strong>{agent['agent_id']}</strong> ({agent['type']})<br>"
                f"ROI: {agent['roi']:.1%} | Balance: {agent['final_balance']:.1f} FTNS"
                f"</div>",
                unsafe_allow_html=True
            )
        
        st.markdown("**📉 Bottom Performers (by ROI):**")
        bottom_performers = agent_df.nsmallest(3, 'roi')[['agent_id', 'type', 'roi', 'final_balance']]
        for _, agent in bottom_performers.iterrows():
            st.markdown(
                f"<div class='agent-card'>"
                f"<strong>{agent['agent_id']}</strong> ({agent['type']})<br>"
                f"ROI: {agent['roi']:.1%} | Balance: {agent['final_balance']:.1f} FTNS"
                f"</div>",
                unsafe_allow_html=True
            )

def render_stress_test_comparison(stress_results):
    """Render stress test scenario comparison"""
    if not stress_results:
        st.warning("No stress test data available")
        return
    
    st.subheader("🧪 Stress Test Scenario Comparison")
    
    # Extract key metrics from each scenario
    scenario_metrics = {}
    for scenario_name, results in stress_results.items():
        metrics = results['key_metrics']
        validation = results['validation_results']
        
        scenario_metrics[scenario_name] = {
            'Final Price': metrics['final_token_price'],
            'Gini Coefficient': metrics['final_gini_coefficient'],
            'Avg Quality': metrics['average_quality'],
            'Price Stability': metrics['price_stability'],
            'Activity Rate': metrics['average_activity_rate'],
            'Success Rate': sum(validation.values()) / len(validation),
            'Overall Success': results['overall_success']
        }
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(scenario_metrics).T
    
    # Display comparison table
    st.dataframe(
        comparison_df.style.format({
            'Final Price': '${:.3f}',
            'Gini Coefficient': '{:.3f}',
            'Avg Quality': '{:.1%}',
            'Price Stability': '{:.1%}',
            'Activity Rate': '{:.1%}',
            'Success Rate': '{:.1%}'
        }).background_gradient(subset=['Success Rate'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate comparison
        fig_success = px.bar(
            x=comparison_df.index,
            y=comparison_df['Success Rate'],
            title='Success Rate by Scenario',
            color=comparison_df['Success Rate'],
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        fig_success.update_xaxes(title='Scenario')
        fig_success.update_yaxes(title='Success Rate', tickformat='.1%')
        fig_success.update_layout(showlegend=False)
        st.plotly_chart(fig_success, use_container_width=True)
    
    with col2:
        # Price stability comparison
        fig_stability = px.bar(
            x=comparison_df.index,
            y=comparison_df['Price Stability'],
            title='Price Stability by Scenario',
            color=comparison_df['Price Stability'],
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        fig_stability.update_xaxes(title='Scenario')
        fig_stability.update_yaxes(title='Price Stability', tickformat='.1%')
        fig_stability.update_layout(showlegend=False)
        st.plotly_chart(fig_stability, use_container_width=True)
    
    # Radar chart for multi-dimensional comparison
    st.subheader("🎯 Multi-Dimensional Performance Radar")
    
    # Prepare radar chart data
    categories = ['Price Stability', 'Avg Quality', 'Activity Rate', 'Success Rate']
    
    fig_radar = go.Figure()
    
    for scenario_name in comparison_df.index:
        values = [comparison_df.loc[scenario_name, cat] for cat in categories]
        values += [values[0]]  # Close the radar chart
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=scenario_name,
            line=dict(width=2)
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.1%'
            )
        ),
        showlegend=True,
        title="Performance Comparison Across All Metrics"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

def render_economic_insights(results):
    """Render economic insights and recommendations"""
    if not results:
        return
    
    st.subheader("💡 Economic Insights & Recommendations")
    
    metrics = results['key_metrics']
    validation = results['validation_results']
    
    insights = []
    recommendations = []
    
    # Wealth distribution analysis
    if metrics['final_gini_coefficient'] > 0.7:
        insights.append("⚠️ High wealth inequality detected (Gini > 0.7)")
        recommendations.append("Implement progressive taxation or wealth redistribution mechanisms")
    else:
        insights.append("✅ Wealth distribution is relatively fair")
    
    # Price stability analysis
    if metrics['price_stability'] < 0.8:
        insights.append("⚠️ Price volatility is high")
        recommendations.append("Consider automated market makers or price stabilization mechanisms")
    else:
        insights.append("✅ Token price is stable")
    
    # Quality analysis
    if metrics['average_quality'] < 0.6:
        insights.append("⚠️ Average contribution quality is low")
        recommendations.append("Increase quality incentives and improve validation mechanisms")
    else:
        insights.append("✅ Contribution quality is maintained")
    
    # Activity analysis
    if metrics['average_activity_rate'] < 0.5:
        insights.append("⚠️ Low network activity")
        recommendations.append("Increase participation incentives and reduce transaction costs")
    else:
        insights.append("✅ Good network participation")
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Key Insights:**")
        for insight in insights:
            st.write(insight)
    
    with col2:
        st.markdown("**💭 Recommendations:**")
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Economic health score
    health_score = sum(validation.values()) / len(validation)
    
    if health_score >= 0.8:
        health_status = "🟢 Excellent"
        health_color = "success"
    elif health_score >= 0.6:
        health_status = "🟡 Good"
        health_color = "warning"
    else:
        health_status = "🔴 Needs Improvement"
        health_color = "error"
    
    st.markdown(f"**Overall Economic Health: {health_status}** ({health_score:.1%})")

def main():
    """Main dashboard application"""
    st.title("💰 PRSM Tokenomics Dashboard")
    st.markdown("Interactive analysis of FTNS token economy simulation and stress testing")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TokenomicsDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.title("🔧 Simulation Controls")
    
    # Simulation parameters
    st.sidebar.subheader("Parameters")
    num_agents = st.sidebar.slider("Number of Agents", min_value=10, max_value=100, value=30, step=5)
    simulation_days = st.sidebar.slider("Simulation Days", min_value=10, max_value=60, value=30, step=5)
    initial_supply = st.sidebar.number_input("Initial Token Supply", min_value=100000, max_value=10000000, value=1000000, step=100000)
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Run Simulation", disabled=not TOKENOMICS_AVAILABLE):
            if not TOKENOMICS_AVAILABLE:
                st.error("Tokenomics simulation not available")
            else:
                with st.spinner("Running economic simulation..."):
                    dashboard.create_simulation(num_agents, simulation_days, initial_supply)
                    results = dashboard.run_simulation()
                    if results:
                        st.success("Simulation completed!")
                        st.rerun()
    
    with col2:
        if st.button("🧪 Stress Tests"):
            if not TOKENOMICS_AVAILABLE:
                st.error("Tokenomics simulation not available")
            else:
                with st.spinner("Running stress test scenarios..."):
                    stress_results = dashboard.run_stress_tests()
                    if stress_results:
                        st.success("Stress tests completed!")
                        st.rerun()
    
    # Stress scenario configuration
    st.sidebar.subheader("Custom Stress Scenario")
    stress_condition = st.sidebar.selectbox(
        "Market Condition",
        ["normal", "bull_market", "bear_market", "volatility_spike", "compute_shortage", "data_flood"]
    )
    stress_start = st.sidebar.slider("Start Day", 0, 25, 5)
    stress_end = st.sidebar.slider("End Day", stress_start + 1, 30, 15)
    
    if st.sidebar.button("Run Custom Stress Test"):
        if dashboard.simulation:
            custom_scenario = [{
                'start_day': stress_start,
                'end_day': stress_end,
                'condition': stress_condition
            }]
            
            with st.spinner(f"Running custom stress test: {stress_condition}..."):
                results = dashboard.run_simulation(custom_scenario)
                if results:
                    st.success(f"Custom stress test completed!")
                    st.rerun()
    
    # Main content area
    if not st.session_state.simulation_run and not st.session_state.stress_test_data:
        st.info("👆 Configure parameters and run a simulation or stress test to begin analysis")
        
        # Show simulation overview
        st.subheader("📖 About FTNS Tokenomics Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Simulation Features:**
            - **Multi-Agent Economy**: 5 agent types with realistic behaviors
            - **Quality-Based Rewards**: Contribution quality affects token rewards
            - **Market Dynamics**: Supply/demand price discovery mechanism
            - **Reputation System**: Agent reputation influences earning potential
            - **Stress Testing**: Various economic shock scenarios
            """)
        
        with col2:
            st.markdown("""
            **📊 Validation Criteria:**
            - **Wealth Distribution**: Gini coefficient < 0.7 (fairness)
            - **Quality Maintenance**: Average quality > 60%
            - **Price Stability**: Price variance < 20%
            - **High Participation**: >50% daily activity
            - **Economic Sustainability**: Positive network growth
            """)
        
        st.markdown("""
        **🏛️ Agent Types:**
        - **Data Contributors**: Upload datasets, earn based on quality and usage
        - **Model Creators**: Develop AI models, higher rewards but higher costs
        - **Query Users**: Consume services, pay transaction fees
        - **Validators**: Validate content quality, earn consistent fees
        - **Freeloaders**: Bad actors attempting to game the system (5% of population)
        """)
        
    else:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📈 Time Series", 
            "👥 Agent Analysis", 
            "🧪 Stress Tests",
            "💡 Insights"
        ])
        
        # Determine which data to display
        current_results = st.session_state.simulation_data
        stress_results = st.session_state.stress_test_data
        
        with tab1:
            if current_results:
                render_economic_metrics(current_results)
            else:
                st.info("Run a simulation to see overview metrics")
        
        with tab2:
            if current_results:
                render_time_series_charts(current_results)
            else:
                st.info("Run a simulation to see time series analysis")
        
        with tab3:
            if current_results:
                render_agent_analysis(current_results)
            else:
                st.info("Run a simulation to see agent analysis")
        
        with tab4:
            if stress_results:
                render_stress_test_comparison(stress_results)
            else:
                st.info("Run stress tests to see scenario comparison")
        
        with tab5:
            if current_results:
                render_economic_insights(current_results)
            else:
                st.info("Run a simulation to see economic insights")
    
    # Footer
    st.markdown("---")
    st.markdown("**PRSM Tokenomics Dashboard** | Built with Streamlit | 💰 Economic Simulation & Analysis")

if __name__ == "__main__":
    main()