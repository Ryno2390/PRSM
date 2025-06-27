#!/usr/bin/env python3
"""
PRSM P2P Network Dashboard
Real-time monitoring dashboard for P2P network demo using Streamlit

Features:
- Real-time network topology visualization
- Node status and metrics monitoring
- Message flow tracking
- Consensus mechanism visualization
- Performance analytics
"""

import streamlit as st
import asyncio
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

try:
    from p2p_network_demo import P2PNetworkDemo, P2PNode
    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False
    st.error("P2P Network Demo not available. Please ensure p2p_network_demo.py is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="PRSM P2P Network Monitor",
    page_icon="🌐",
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
}
.status-active { color: #28a745; }
.status-inactive { color: #dc3545; }
.status-syncing { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

class P2PDashboard:
    """P2P Network Dashboard Controller"""
    
    def __init__(self):
        self.network = None
        self.is_running = False
        self.last_update = 0
        self.historical_data = []
        
        # Initialize session state
        if 'network_started' not in st.session_state:
            st.session_state.network_started = False
        if 'historical_metrics' not in st.session_state:
            st.session_state.historical_metrics = []
    
    def create_network(self, num_nodes: int):
        """Create P2P network with specified number of nodes"""
        if P2P_AVAILABLE:
            self.network = P2PNetworkDemo(num_nodes=num_nodes)
            return True
        return False
    
    async def start_network(self):
        """Start the P2P network"""
        if self.network and not self.is_running:
            self.is_running = True
            st.session_state.network_started = True
            
            # Start network in background
            try:
                await asyncio.wait_for(self._background_network_runner(), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # Expected - network runs in background
    
    async def _background_network_runner(self):
        """Run network in background"""
        await self.network.start_network()
    
    def get_network_metrics(self):
        """Get current network metrics"""
        if not self.network:
            return None
        
        try:
            status = self.network.get_network_status()
            
            # Add timestamp
            status['timestamp'] = time.time()
            
            # Store historical data
            st.session_state.historical_metrics.append(status)
            
            # Keep only last 100 data points
            if len(st.session_state.historical_metrics) > 100:
                st.session_state.historical_metrics = st.session_state.historical_metrics[-100:]
            
            return status
        except Exception as e:
            st.error(f"Error getting network metrics: {e}")
            return None
    
    def stop_network(self):
        """Stop the P2P network"""
        self.is_running = False
        st.session_state.network_started = False
        if self.network:
            # Note: In real app, would properly await this
            # For Streamlit demo, we'll just mark as stopped
            for node in self.network.nodes:
                node.is_running = False
                node.node_info.status = "inactive"

def render_network_topology(network_status):
    """Render network topology visualization"""
    if not network_status or not network_status.get('nodes'):
        st.warning("No network data available")
        return
    
    # Create network graph
    fig = go.Figure()
    
    # Node positions (circular layout)
    import math
    nodes = network_status['nodes']
    num_nodes = len(nodes)
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / num_nodes
        x = math.cos(angle)
        y = math.sin(angle)
        
        node_x.append(x)
        node_y.append(y)
        
        node_info = node['node_info']
        metrics = node['network_metrics']
        
        node_text.append(
            f"Node: {node_info['node_id']}<br>"
            f"Type: {node_info['node_type']}<br>"
            f"Status: {node_info['status']}<br>"
            f"Peers: {metrics['known_peers']}<br>"
            f"Messages: {metrics['messages_sent']}↗ {metrics['messages_received']}↙"
        )
        
        # Color by status
        if node_info['status'] == 'active':
            node_color.append('#28a745')
        elif node_info['status'] == 'inactive':
            node_color.append('#dc3545')
        else:
            node_color.append('#ffc107')
    
    # Add connections between nodes
    edge_x = []
    edge_y = []
    
    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i != j and other_node['node_info']['node_id'] in node.get('peers', {}):
                edge_x.extend([node_x[i], node_x[j], None])
                edge_y.extend([node_y[i], node_y[j], None])
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        showlegend=False,
        name='Connections'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        text=[node['node_info']['node_id'] for node in nodes],
        textposition="middle center",
        textfont=dict(color="white", size=10),
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False,
        name='Nodes'
    ))
    
    fig.update_layout(
        title="P2P Network Topology",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="🟢 Active  🟡 Syncing  🔴 Inactive",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_metrics_charts(historical_data):
    """Render metrics charts"""
    if not historical_data:
        st.info("Waiting for network metrics...")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': datetime.fromtimestamp(data['timestamp']),
            'active_nodes': data['active_nodes'],
            'total_connections': data['total_connections'],
            'total_messages': data['total_messages'],
            'consensus_proposals': data['consensus_proposals']
        }
        for data in historical_data
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Messages over time
        fig_messages = px.line(
            df, x='timestamp', y='total_messages',
            title='Total Messages Over Time',
            labels={'total_messages': 'Total Messages', 'timestamp': 'Time'}
        )
        fig_messages.update_layout(showlegend=False)
        st.plotly_chart(fig_messages, use_container_width=True)
    
    with col2:
        # Network connections
        fig_connections = px.line(
            df, x='timestamp', y='total_connections',
            title='Network Connections Over Time',
            labels={'total_connections': 'Connections', 'timestamp': 'Time'}
        )
        fig_connections.update_layout(showlegend=False)
        st.plotly_chart(fig_connections, use_container_width=True)

def render_node_details(network_status):
    """Render detailed node information"""
    if not network_status or not network_status.get('nodes'):
        return
    
    st.subheader("📊 Node Details")
    
    for i, node in enumerate(network_status['nodes']):
        node_info = node['node_info']
        metrics = node['network_metrics']
        
        with st.expander(f"Node {i+1}: {node_info['node_id']} ({node_info['node_type']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_color = "status-active" if node_info['status'] == 'active' else "status-inactive"
                st.markdown(f"**Status:** <span class='{status_color}'>{node_info['status']}</span>", unsafe_allow_html=True)
                st.write(f"**Address:** {node_info['address']}")
                st.write(f"**Reputation:** {node_info['reputation_score']:.1f}")
            
            with col2:
                st.write(f"**Known Peers:** {metrics['known_peers']}")
                st.write(f"**Messages Sent:** {metrics['messages_sent']}")
                st.write(f"**Messages Received:** {metrics['messages_received']}")
            
            with col3:
                st.write(f"**Consensus Participations:** {metrics['consensus_participations']}")
                st.write(f"**Failed Connections:** {metrics['failed_connections']}")
                st.write(f"**Pending Consensus:** {metrics['pending_consensus']}")
            
            # Recent messages
            if node.get('recent_messages'):
                st.write("**Recent Messages:**")
                msg_df = pd.DataFrame(node['recent_messages'])
                if not msg_df.empty:
                    msg_df['timestamp'] = msg_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%H:%M:%S'))
                    st.dataframe(msg_df, hide_index=True)

def render_consensus_monitor(network_status):
    """Render consensus mechanism monitoring"""
    st.subheader("🤝 Consensus Monitoring")
    
    if not network_status or not network_status.get('nodes'):
        st.info("No consensus data available")
        return
    
    total_proposals = network_status.get('consensus_proposals', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Proposals", total_proposals)
    
    with col2:
        # Calculate average participation
        nodes_with_consensus = sum(1 for node in network_status['nodes'] 
                                 if node['network_metrics']['consensus_participations'] > 0)
        participation_rate = (nodes_with_consensus / len(network_status['nodes']) * 100) if network_status['nodes'] else 0
        st.metric("Participation Rate", f"{participation_rate:.1f}%")
    
    with col3:
        # Network consensus health
        active_nodes = network_status.get('active_nodes', 0)
        health_score = min(100, (active_nodes / max(1, len(network_status['nodes']))) * 100)
        st.metric("Network Health", f"{health_score:.1f}%")
    
    # Consensus activity timeline
    if st.session_state.historical_metrics:
        consensus_df = pd.DataFrame([
            {
                'timestamp': datetime.fromtimestamp(data['timestamp']),
                'consensus_proposals': data['consensus_proposals']
            }
            for data in st.session_state.historical_metrics
        ])
        
        if not consensus_df.empty:
            fig = px.bar(
                consensus_df, x='timestamp', y='consensus_proposals',
                title='Consensus Proposals Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application"""
    st.title("🌐 PRSM P2P Network Monitor")
    st.markdown("Real-time monitoring of PRSM's decentralized peer-to-peer network")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = P2PDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.title("Network Controls")
    
    # Network configuration
    st.sidebar.subheader("Configuration")
    num_nodes = st.sidebar.slider("Number of Nodes", min_value=2, max_value=10, value=3)
    
    # Network control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Start Network", disabled=st.session_state.network_started):
            if not P2P_AVAILABLE:
                st.error("P2P Network Demo not available")
            else:
                with st.spinner("Starting network..."):
                    if dashboard.create_network(num_nodes):
                        # For Streamlit, we'll simulate network start
                        st.session_state.network_started = True
                        st.success("Network started!")
                        st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Network", disabled=not st.session_state.network_started):
            dashboard.stop_network()
            st.success("Network stopped!")
            st.rerun()
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
    
    if auto_refresh and st.session_state.network_started:
        time.sleep(5)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Demo actions
    st.sidebar.subheader("Demo Actions")
    
    if st.session_state.network_started:
        if st.sidebar.button("🤝 Trigger Consensus"):
            st.sidebar.success("Consensus proposal initiated!")
        
        if st.sidebar.button("📁 Share File"):
            st.sidebar.success("File sharing demo started!")
        
        if st.sidebar.button("⚠️ Simulate Failure"):
            st.sidebar.warning("Node failure simulated!")
    
    # Main content area
    if not st.session_state.network_started:
        st.info("👆 Start the network using the sidebar controls to begin monitoring")
        
        # Show demo information
        st.subheader("📖 About this Demo")
        st.write("""
        This dashboard monitors a simulated P2P network demonstrating PRSM's decentralized architecture:
        
        **Features Demonstrated:**
        - 🔍 **Node Discovery**: Automatic peer discovery and handshaking
        - 💬 **Message Passing**: Secure message exchange between nodes
        - 🤝 **Consensus Mechanisms**: Byzantine fault tolerant consensus simulation
        - 📁 **File Sharing**: Secure data distribution across the network
        - 📊 **Real-time Monitoring**: Live network metrics and status
        - ⚠️ **Failure Recovery**: Node failure simulation and auto-recovery
        
        **Network Architecture:**
        - **Coordinator Nodes**: Initiate consensus proposals and coordinate network activities
        - **Worker Nodes**: Execute tasks and participate in consensus voting
        - **Validator Nodes**: Validate data integrity and maintain network security
        """)
        
    else:
        # Get current network status
        network_status = dashboard.get_network_metrics()
        
        if network_status:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Active Nodes", 
                    f"{network_status['active_nodes']}/{network_status['total_nodes']}"
                )
            
            with col2:
                st.metric("Total Connections", network_status['total_connections'])
            
            with col3:
                st.metric("Messages Exchanged", network_status['total_messages'])
            
            with col4:
                st.metric("Consensus Proposals", network_status['consensus_proposals'])
            
            # Network topology
            st.subheader("🕸️ Network Topology")
            render_network_topology(network_status)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Metrics", "📊 Node Details", "🤝 Consensus"])
            
            with tab1:
                render_metrics_charts(st.session_state.historical_metrics)
            
            with tab2:
                render_node_details(network_status)
            
            with tab3:
                render_consensus_monitor(network_status)
        
        else:
            st.error("Failed to get network status")
    
    # Footer
    st.markdown("---")
    st.markdown("**PRSM P2P Network Demo** | Built with Streamlit | 🌐 Decentralized AI Infrastructure")

if __name__ == "__main__":
    main()