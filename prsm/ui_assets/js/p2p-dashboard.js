/**
 * P2P Network Dashboard JavaScript
 * Handles real-time updates, interactions, and data visualization
 */

class P2PNetworkDashboard {
    constructor() {
        this.updateInterval = 5000; // 5 seconds
        this.charts = {};
        this.networkData = {
            peers: [],
            metrics: {},
            security: {},
            alerts: []
        };
        
        this.initializeComponents();
        this.startRealTimeUpdates();
    }

    initializeComponents() {
        this.initializeCharts();
        this.bindEventListeners();
        this.loadInitialData();
    }

    initializeCharts() {
        // Initialize performance charts
        const chartConfigs = [
            { id: 'throughput-chart', type: 'line', color: '#10b981' },
            { id: 'latency-chart', type: 'line', color: '#f59e0b' },
            { id: 'success-chart', type: 'line', color: '#3b82f6' }
        ];

        chartConfigs.forEach(config => {
            const canvas = document.getElementById(config.id);
            if (canvas) {
                this.charts[config.id] = this.createChart(canvas, config);
            }
        });
    }

    createChart(canvas, config) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Generate sample data
        const dataPoints = this.generateSampleData(20);
        
        return {
            canvas: canvas,
            ctx: ctx,
            data: dataPoints,
            config: config,
            draw: () => this.drawChart(ctx, dataPoints, config, width, height)
        };
    }

    drawChart(ctx, data, config, width, height) {
        ctx.clearRect(0, 0, width, height);
        
        // Set up the chart
        const padding = 20;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Find min/max values
        const minValue = Math.min(...data);
        const maxValue = Math.max(...data);
        const range = maxValue - minValue;
        
        // Draw grid lines
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;
        
        // Horizontal grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding + (i * chartHeight / 4);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }
        
        // Draw the line
        ctx.strokeStyle = config.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.forEach((value, index) => {
            const x = padding + (index * chartWidth / (data.length - 1));
            const y = padding + chartHeight - ((value - minValue) / range * chartHeight);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Fill area under the curve
        ctx.fillStyle = config.color + '20';
        ctx.lineTo(width - padding, height - padding);
        ctx.lineTo(padding, height - padding);
        ctx.closePath();
        ctx.fill();
    }

    generateSampleData(points) {
        const data = [];
        let value = 50 + Math.random() * 50;
        
        for (let i = 0; i < points; i++) {
            value += (Math.random() - 0.5) * 10;
            value = Math.max(0, Math.min(100, value));
            data.push(value);
        }
        
        return data;
    }

    bindEventListeners() {
        // Network topology view selector
        const topologyView = document.getElementById('topology-view');
        if (topologyView) {
            topologyView.addEventListener('change', (e) => {
                this.updateTopologyView(e.target.value);
            });
        }

        // Peer filters
        const regionFilter = document.getElementById('region-filter');
        const reputationFilter = document.getElementById('reputation-filter');
        
        if (regionFilter) {
            regionFilter.addEventListener('change', () => this.filterPeers());
        }
        
        if (reputationFilter) {
            reputationFilter.addEventListener('change', () => this.filterPeers());
        }

        // Search input
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchPeers(e.target.value);
            });
        }

        // Node interactions
        this.bindNodeInteractions();
    }

    bindNodeInteractions() {
        const networkNodes = document.querySelectorAll('.network-node');
        networkNodes.forEach(node => {
            node.addEventListener('click', (e) => {
                const nodeId = e.currentTarget.dataset.nodeId;
                this.showNodeDetails(nodeId);
            });
            
            node.addEventListener('mouseenter', (e) => {
                this.showNodeTooltip(e.currentTarget);
            });
            
            node.addEventListener('mouseleave', (e) => {
                this.hideNodeTooltip(e.currentTarget);
            });
        });
    }

    loadInitialData() {
        // Simulate loading initial network data
        this.networkData = {
            peers: [
                {
                    id: 'stanford-lab-01',
                    name: 'Stanford-Lab-01',
                    region: 'us-west',
                    reputation: 4.8,
                    latency: 23,
                    bandwidth: { upload: 45, download: 89 },
                    shards: 127,
                    status: 'online'
                },
                {
                    id: 'mit-research-03',
                    name: 'MIT-Research-03',
                    region: 'us-east',
                    reputation: 4.2,
                    latency: 67,
                    bandwidth: { upload: 23, download: 45 },
                    shards: 89,
                    status: 'online'
                },
                {
                    id: 'duke-medical-02',
                    name: 'Duke-Medical-02',
                    region: 'us-east',
                    reputation: 4.9,
                    latency: 12,
                    bandwidth: { upload: 67, download: 123 },
                    shards: 203,
                    status: 'online'
                }
            ],
            metrics: {
                activePeers: 47,
                shardCount: 1248,
                securityScore: 98.7,
                avgLatency: 45
            }
        };

        this.updateDashboard();
    }

    startRealTimeUpdates() {
        setInterval(() => {
            this.updateMetrics();
            this.updateCharts();
        }, this.updateInterval);
    }

    updateMetrics() {
        // Simulate real-time metric updates
        const metrics = this.networkData.metrics;
        
        // Add some random variation
        metrics.activePeers += Math.floor((Math.random() - 0.5) * 2);
        metrics.avgLatency += Math.floor((Math.random() - 0.5) * 5);
        metrics.securityScore += (Math.random() - 0.5) * 0.1;
        
        // Update DOM elements
        this.updateElement('active-peers', metrics.activePeers);
        this.updateElement('shard-count', metrics.shardCount);
        this.updateElement('security-score', metrics.securityScore.toFixed(1) + '%');
        this.updateElement('avg-latency', metrics.avgLatency + 'ms');
    }

    updateCharts() {
        Object.values(this.charts).forEach(chart => {
            // Add new data point and remove oldest
            chart.data.shift();
            chart.data.push(50 + Math.random() * 50);
            chart.draw();
        });
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateTopologyView(viewType) {
        console.log(`Switching to ${viewType} topology view`);
        
        // Update the topology visualization based on view type
        const topologyMap = document.getElementById('topology-map');
        if (topologyMap) {
            topologyMap.className = `topology-map ${viewType}-view`;
            
            // You could implement different layouts here
            switch (viewType) {
                case 'geographic':
                    this.arrangeNodesGeographically();
                    break;
                case 'logical':
                    this.arrangeNodesLogically();
                    break;
                case 'performance':
                    this.arrangeNodesByPerformance();
                    break;
            }
        }
    }

    arrangeNodesGeographically() {
        // Implement geographic arrangement logic
        console.log('Arranging nodes geographically');
    }

    arrangeNodesLogically() {
        // Implement logical arrangement logic
        console.log('Arranging nodes logically');
    }

    arrangeNodesByPerformance() {
        // Implement performance-based arrangement logic
        console.log('Arranging nodes by performance');
    }

    filterPeers() {
        const regionFilter = document.getElementById('region-filter')?.value;
        const reputationFilter = document.getElementById('reputation-filter')?.value;
        
        const peerRows = document.querySelectorAll('.peer-row');
        
        peerRows.forEach(row => {
            let show = true;
            
            // Apply region filter
            if (regionFilter && regionFilter !== 'all') {
                const peerRegion = row.dataset.region;
                if (peerRegion !== regionFilter) {
                    show = false;
                }
            }
            
            // Apply reputation filter
            if (reputationFilter && reputationFilter !== 'all') {
                const reputationScore = parseFloat(row.querySelector('.reputation-score .score')?.textContent);
                
                switch (reputationFilter) {
                    case 'high':
                        if (reputationScore < 4.0) show = false;
                        break;
                    case 'medium':
                        if (reputationScore < 3.0 || reputationScore >= 4.0) show = false;
                        break;
                    case 'low':
                        if (reputationScore >= 3.0) show = false;
                        break;
                }
            }
            
            row.style.display = show ? 'table-row' : 'none';
        });
    }

    searchPeers(query) {
        const peerRows = document.querySelectorAll('.peer-row');
        const searchTerm = query.toLowerCase();
        
        peerRows.forEach(row => {
            const peerName = row.querySelector('.peer-info strong')?.textContent.toLowerCase();
            const show = !searchTerm || peerName.includes(searchTerm);
            row.style.display = show ? 'table-row' : 'none';
        });
    }

    showNodeDetails(nodeId) {
        console.log(`Showing details for node: ${nodeId}`);
        
        // Create and show a detailed modal for the node
        const modal = this.createNodeDetailsModal(nodeId);
        document.body.appendChild(modal);
        
        // Show the modal
        modal.style.display = 'flex';
    }

    createNodeDetailsModal(nodeId) {
        const peer = this.networkData.peers.find(p => p.id === nodeId);
        
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.id = `node-details-${nodeId}`;
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h4><i class="fas fa-info-circle"></i> Node Details: ${peer?.name || nodeId}</h4>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="node-details-grid">
                        <div class="detail-item">
                            <label>Node ID:</label>
                            <span>${nodeId}</span>
                        </div>
                        <div class="detail-item">
                            <label>Region:</label>
                            <span>${peer?.region || 'Unknown'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Reputation:</label>
                            <span>${peer?.reputation || 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Latency:</label>
                            <span>${peer?.latency || 'N/A'}ms</span>
                        </div>
                        <div class="detail-item">
                            <label>Bandwidth:</label>
                            <span>↑${peer?.bandwidth.upload || 'N/A'} ↓${peer?.bandwidth.download || 'N/A'} MB/s</span>
                        </div>
                        <div class="detail-item">
                            <label>Shards Hosted:</label>
                            <span>${peer?.shards || 'N/A'}</span>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                </div>
            </div>
        `;
        
        return modal;
    }

    showNodeTooltip(node) {
        // Show detailed stats on hover
        const stats = node.querySelector('.node-stats');
        if (stats) {
            stats.style.opacity = '1';
        }
    }

    hideNodeTooltip(node) {
        const stats = node.querySelector('.node-stats');
        if (stats) {
            stats.style.opacity = '0';
        }
    }

    updateDashboard() {
        // Update all dashboard components with current data
        this.updateMetrics();
        this.updatePeerTable();
        this.updateSecurityStatus();
    }

    updatePeerTable() {
        const tableBody = document.getElementById('peer-table-body');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Add rows for each peer
        this.networkData.peers.forEach(peer => {
            const row = this.createPeerRow(peer);
            tableBody.appendChild(row);
        });
    }

    createPeerRow(peer) {
        const row = document.createElement('tr');
        row.className = 'peer-row';
        row.dataset.peerId = peer.id;
        row.dataset.region = peer.region;
        
        const reputationClass = peer.reputation >= 4.0 ? 'high' : 
                               peer.reputation >= 3.0 ? 'medium' : 'low';
        
        const latencyClass = peer.latency <= 50 ? 'excellent' :
                            peer.latency <= 100 ? 'good' :
                            peer.latency <= 200 ? 'fair' : 'poor';
        
        const stars = '★'.repeat(Math.floor(peer.reputation)) + 
                     '☆'.repeat(5 - Math.floor(peer.reputation));
        
        row.innerHTML = `
            <td>
                <div class="peer-info">
                    <i class="fas fa-laptop peer-icon"></i>
                    <div>
                        <strong>${peer.name}</strong>
                        <small>${peer.region}</small>
                    </div>
                </div>
            </td>
            <td>
                <div class="reputation-score ${reputationClass}">
                    <span class="score">${peer.reputation}</span>
                    <div class="stars">${stars}</div>
                </div>
            </td>
            <td><span class="latency ${latencyClass}">${peer.latency}ms</span></td>
            <td><span class="bandwidth high">↑${peer.bandwidth.upload} ↓${peer.bandwidth.download} MB/s</span></td>
            <td><span class="shard-count">${peer.shards}</span></td>
            <td><span class="status-badge ${peer.status}">${peer.status}</span></td>
        `;
        
        return row;
    }

    updateSecurityStatus() {
        // Update security metrics and alerts
        console.log('Updating security status');
    }
}

// Global functions for UI interactions
window.refreshNetworkData = function() {
    console.log('Refreshing network data...');
    dashboard.loadInitialData();
};

window.togglePeerFilters = function() {
    const filters = document.getElementById('peer-filters');
    if (filters) {
        filters.style.display = filters.style.display === 'none' ? 'flex' : 'none';
    }
};

window.optimizeNetwork = function() {
    console.log('Optimizing network performance...');
    // Implement network optimization logic
};

window.exportNetworkData = function() {
    console.log('Exporting network data...');
    // Implement data export functionality
};

window.showNetworkSettings = function() {
    const modal = document.getElementById('network-settings-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
};

window.emergencyDisconnect = function() {
    if (confirm('Are you sure you want to disconnect from the P2P network? This will stop all file sharing and collaboration.')) {
        console.log('Emergency disconnect initiated...');
        // Implement emergency disconnect logic
    }
};

window.runSecurityScan = function() {
    console.log('Running security scan...');
    // Implement security scan functionality
};

window.closeModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
};

window.saveNetworkSettings = function() {
    console.log('Saving network settings...');
    // Implement settings save logic
    closeModal('network-settings-modal');
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new P2PNetworkDashboard();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = P2PNetworkDashboard;
}