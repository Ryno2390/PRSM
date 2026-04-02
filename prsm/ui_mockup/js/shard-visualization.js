/**
 * Shard Distribution Visualization JavaScript
 * Handles real-time shard monitoring and distribution management
 */

class ShardVisualization {
    constructor() {
        this.updateInterval = 8000; // 8 seconds
        this.shardData = {
            totalShards: 1248,
            activeFiles: 178,
            distributionNodes: 47,
            redundancyScore: 98.7,
            files: new Map(),
            nodes: new Map(),
            distributionMetrics: {
                geographic: {
                    northAmerica: 65,
                    europe: 25,
                    asia: 10
                },
                security: {
                    high: 30,
                    medium: 20,
                    standard: 50
                }
            }
        };
        
        this.initializeComponents();
        this.startRealTimeUpdates();
    }

    initializeComponents() {
        this.loadShardData();
        this.bindEventListeners();
        this.updateMetrics();
        this.initializeMap();
    }

    bindEventListeners() {
        // Map view selector
        const mapView = document.getElementById('map-view');
        if (mapView) {
            mapView.addEventListener('change', (e) => {
                this.updateMapView(e.target.value);
            });
        }

        // Security filter
        const securityFilter = document.getElementById('security-filter');
        if (securityFilter) {
            securityFilter.addEventListener('change', (e) => {
                this.filterFilesBySecurity(e.target.value);
            });
        }

        // Search input
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchFiles(e.target.value);
            });
        }

        // Shard node interactions
        this.bindNodeInteractions();
    }

    bindNodeInteractions() {
        const shardNodes = document.querySelectorAll('.shard-node');
        shardNodes.forEach(node => {
            node.addEventListener('click', (e) => {
                const nodeId = e.currentTarget.dataset.node;
                this.showNodeDetails(nodeId);
            });
        });
    }

    loadShardData() {
        // Simulate loading shard data from backend
        this.shardData.files.set('prop-algo-v2', {
            name: 'Proprietary_Algorithm_v2.pdf',
            size: '2.4 MB',
            security: 'high',
            shardCount: 7,
            shards: [
                { id: 1, location: 'Stanford Lab-01', status: 'verified' },
                { id: 2, location: 'MIT Research-03', status: 'verified' },
                { id: 3, location: 'Duke Medical-02', status: 'verified' },
                { id: 4, location: 'Oxford Quantum-Lab', status: 'verified' },
                { id: 5, location: 'ETH Zurich-Main', status: 'verified' },
                { id: 6, location: 'Tokyo Tech-AI', status: 'verified' },
                { id: 7, location: 'Backup Storage', status: 'replicating' }
            ]
        });

        this.shardData.files.set('research-notebook', {
            name: 'research_notebook.ipynb',
            size: '856 KB',
            security: 'medium',
            shardCount: 5,
            shards: [
                { id: 1, location: 'Stanford Lab-01', status: 'verified' },
                { id: 2, location: 'MIT Research-03', status: 'verified' },
                { id: 3, location: 'Duke Medical-02', status: 'verified' },
                { id: 4, location: 'ETH Zurich-Main', status: 'verified' },
                { id: 5, location: 'Tokyo Tech-AI', status: 'verified' }
            ]
        });

        this.shardData.nodes.set('stanford', {
            name: 'Stanford Lab-01',
            region: 'North America',
            shardCount: 127,
            loadPercentage: 85,
            status: 'online'
        });

        this.shardData.nodes.set('mit', {
            name: 'MIT Research-03',
            region: 'North America',
            shardCount: 89,
            loadPercentage: 72,
            status: 'online'
        });

        this.shardData.nodes.set('duke', {
            name: 'Duke Medical-02',
            region: 'North America',
            shardCount: 203,
            loadPercentage: 92,
            status: 'online'
        });
    }

    updateMetrics() {
        // Add some randomness to simulate real-time updates
        this.shardData.totalShards += Math.floor((Math.random() - 0.5) * 10);
        this.shardData.redundancyScore += (Math.random() - 0.5) * 0.2;
        this.shardData.redundancyScore = Math.max(95, Math.min(100, this.shardData.redundancyScore));

        // Update DOM elements
        this.updateElement('total-shards', this.shardData.totalShards.toLocaleString());
        this.updateElement('active-files', this.shardData.activeFiles);
        this.updateElement('distribution-nodes', this.shardData.distributionNodes);
        this.updateElement('redundancy-score', this.shardData.redundancyScore.toFixed(1) + '%');
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    initializeMap() {
        // Initialize the world map with shard distribution
        this.updateMapView('geographic');
    }

    updateMapView(viewType) {
        console.log(`Switching to ${viewType} map view`);
        
        const worldMap = document.getElementById('shard-world-map');
        if (worldMap) {
            worldMap.className = `world-map ${viewType}-view`;
            
            switch (viewType) {
                case 'geographic':
                    this.arrangeNodesGeographically();
                    break;
                case 'cluster':
                    this.arrangeNodesByCluster();
                    break;
                case 'security':
                    this.arrangeNodesBySecurity();
                    break;
            }
        }
    }

    arrangeNodesGeographically() {
        // Implement geographic arrangement logic
        console.log('Arranging nodes geographically');
        // Nodes are already positioned geographically in CSS
    }

    arrangeNodesByCluster() {
        // Implement cluster arrangement logic
        console.log('Arranging nodes by cluster');
        
        const nodes = document.querySelectorAll('.shard-node');
        nodes.forEach((node, index) => {
            const angle = (index * 60) * (Math.PI / 180); // 60 degrees apart
            const radius = 35; // Percentage from center
            const centerX = 50;
            const centerY = 50;
            
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            
            node.style.left = x + '%';
            node.style.top = y + '%';
        });
    }

    arrangeNodesBySecurity() {
        // Implement security-based arrangement logic
        console.log('Arranging nodes by security level');
        
        const highSecurityNodes = document.querySelectorAll('.shard-node.primary');
        const mediumSecurityNodes = document.querySelectorAll('.shard-node.secondary');
        const standardSecurityNodes = document.querySelectorAll('.shard-node.tertiary');
        
        // Arrange high security nodes in inner circle
        highSecurityNodes.forEach((node, index) => {
            const angle = (index * 120) * (Math.PI / 180);
            const radius = 20;
            const centerX = 50;
            const centerY = 50;
            
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            
            node.style.left = x + '%';
            node.style.top = y + '%';
        });
        
        // Arrange medium security nodes in middle circle
        mediumSecurityNodes.forEach((node, index) => {
            const angle = (index * 90) * (Math.PI / 180);
            const radius = 35;
            const centerX = 50;
            const centerY = 50;
            
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            
            node.style.left = x + '%';
            node.style.top = y + '%';
        });
    }

    filterFilesBySecurity(securityLevel) {
        const fileItems = document.querySelectorAll('.file-item');
        
        fileItems.forEach(item => {
            if (securityLevel === 'all') {
                item.style.display = 'block';
            } else {
                const itemSecurity = item.classList.contains(`${securityLevel}-security`);
                item.style.display = itemSecurity ? 'block' : 'none';
            }
        });
    }

    searchFiles(query) {
        const fileItems = document.querySelectorAll('.file-item');
        const searchTerm = query.toLowerCase();
        
        fileItems.forEach(item => {
            const fileName = item.querySelector('.file-name').textContent.toLowerCase();
            const show = !searchTerm || fileName.includes(searchTerm);
            item.style.display = show ? 'block' : 'none';
        });
    }

    showNodeDetails(nodeId) {
        console.log(`Showing details for node: ${nodeId}`);
        
        const nodeData = this.shardData.nodes.get(nodeId);
        if (!nodeData) return;
        
        const modal = this.createNodeDetailsModal(nodeId, nodeData);
        document.body.appendChild(modal);
        modal.style.display = 'flex';
    }

    createNodeDetailsModal(nodeId, nodeData) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.id = `node-details-${nodeId}`;
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h4><i class="fas fa-info-circle"></i> Node Details: ${nodeData.name}</h4>
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
                            <span>${nodeData.region}</span>
                        </div>
                        <div class="detail-item">
                            <label>Shard Count:</label>
                            <span>${nodeData.shardCount}</span>
                        </div>
                        <div class="detail-item">
                            <label>Load:</label>
                            <span>${nodeData.loadPercentage}%</span>
                        </div>
                        <div class="detail-item">
                            <label>Status:</label>
                            <span class="status-badge ${nodeData.status}">${nodeData.status}</span>
                        </div>
                    </div>
                    <div class="node-actions">
                        <button class="action-btn primary" onclick="redistributeFromNode('${nodeId}')">
                            <i class="fas fa-sync-alt"></i> Redistribute Shards
                        </button>
                        <button class="action-btn secondary" onclick="viewNodeLogs('${nodeId}')">
                            <i class="fas fa-file-alt"></i> View Logs
                        </button>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                </div>
            </div>
        `;
        
        return modal;
    }

    startRealTimeUpdates() {
        setInterval(() => {
            this.updateMetrics();
            this.updateShardStatus();
        }, this.updateInterval);
    }

    updateShardStatus() {
        // Simulate shard status updates
        const shardItems = document.querySelectorAll('.shard-item');
        shardItems.forEach(item => {
            const indicator = item.querySelector('.shard-indicator');
            const status = item.querySelector('.shard-status');
            
            // Small chance of status change
            if (Math.random() < 0.05) {
                if (status.textContent === 'Replicating') {
                    status.textContent = 'Verified';
                    status.className = 'shard-status verified';
                    indicator.className = 'shard-indicator active';
                }
            }
        });
    }
}

// Global functions for UI interactions
window.toggleFileDetails = function(fileId) {
    const details = document.getElementById(`details-${fileId}`);
    const expandBtn = event.target;
    
    if (details.style.display === 'block') {
        details.style.display = 'none';
        expandBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
    } else {
        details.style.display = 'block';
        expandBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
    }
};

window.reconstructFile = function(fileId) {
    console.log(`Reconstructing file: ${fileId}`);
    
    const reconstructBtn = event.target;
    const originalText = reconstructBtn.innerHTML;
    
    reconstructBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Reconstructing...';
    reconstructBtn.disabled = true;
    
    setTimeout(() => {
        reconstructBtn.innerHTML = originalText;
        reconstructBtn.disabled = false;
        console.log('File reconstruction completed successfully');
        alert('File has been reconstructed from distributed shards');
    }, 3000);
};

window.redistributeShards = function(fileId) {
    console.log(`Redistributing shards for file: ${fileId}`);
    
    const redistributeBtn = event.target;
    const originalText = redistributeBtn.innerHTML;
    
    redistributeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Redistributing...';
    redistributeBtn.disabled = true;
    
    setTimeout(() => {
        redistributeBtn.innerHTML = originalText;
        redistributeBtn.disabled = false;
        console.log('Shard redistribution completed');
        alert('Shards have been optimally redistributed across the network');
    }, 4000);
};

window.validateIntegrity = function(fileId) {
    console.log(`Validating integrity for file: ${fileId}`);
    
    const validateBtn = event.target;
    const originalText = validateBtn.innerHTML;
    
    validateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
    validateBtn.disabled = true;
    
    setTimeout(() => {
        validateBtn.innerHTML = originalText;
        validateBtn.disabled = false;
        console.log('Integrity validation completed');
        alert('All shards validated successfully - integrity confirmed');
    }, 2500);
};

window.refreshShardData = function() {
    console.log('Refreshing shard distribution data...');
    
    const refreshBtn = event.target;
    const originalText = refreshBtn.innerHTML;
    
    refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
    refreshBtn.disabled = true;
    
    setTimeout(() => {
        refreshBtn.innerHTML = originalText;
        refreshBtn.disabled = false;
        window.shardVisualization.updateMetrics();
        console.log('Shard data refreshed successfully');
    }, 1500);
};

window.optimizeDistribution = function() {
    console.log('Starting distribution optimization...');
    
    const optimizeBtn = event.target;
    const originalText = optimizeBtn.innerHTML;
    
    optimizeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Optimizing...';
    optimizeBtn.disabled = true;
    
    setTimeout(() => {
        optimizeBtn.innerHTML = originalText;
        optimizeBtn.disabled = false;
        console.log('Distribution optimization completed');
        alert('Network distribution has been optimized for performance and security');
    }, 5000);
};

window.rebalanceShards = function() {
    console.log('Starting shard rebalancing...');
    
    const rebalanceBtn = event.target;
    const originalText = rebalanceBtn.innerHTML;
    
    rebalanceBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Rebalancing...';
    rebalanceBtn.disabled = true;
    
    setTimeout(() => {
        rebalanceBtn.innerHTML = originalText;
        rebalanceBtn.disabled = false;
        
        // Update load percentages to show rebalancing effect
        const loadFills = document.querySelectorAll('.load-fill');
        const loadPercents = document.querySelectorAll('.load-percent');
        
        loadFills.forEach((fill, index) => {
            const newLoad = 60 + Math.random() * 20; // 60-80%
            fill.style.width = newLoad + '%';
            if (loadPercents[index]) {
                loadPercents[index].textContent = Math.round(newLoad) + '%';
            }
        });
        
        console.log('Shard rebalancing completed');
        alert('Shards have been rebalanced to optimize load distribution');
    }, 4500);
};

window.exportDistributionReport = function() {
    console.log('Generating distribution report...');
    
    const exportBtn = event.target;
    const originalText = exportBtn.innerHTML;
    
    exportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    exportBtn.disabled = true;
    
    setTimeout(() => {
        exportBtn.innerHTML = originalText;
        exportBtn.disabled = false;
        console.log('Distribution report generated');
        alert('Distribution report has been generated and saved to secure storage');
    }, 2000);
};

window.emergencyRedistribute = function() {
    if (confirm('Are you sure you want to perform an emergency redistribution? This will temporarily disrupt file access.')) {
        console.log('Initiating emergency redistribution...');
        
        const emergencyBtn = event.target;
        const originalText = emergencyBtn.innerHTML;
        
        emergencyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Emergency Mode...';
        emergencyBtn.disabled = true;
        
        setTimeout(() => {
            emergencyBtn.innerHTML = originalText;
            emergencyBtn.disabled = false;
            console.log('Emergency redistribution completed');
            alert('Emergency redistribution completed - all files are now secure');
        }, 6000);
    }
};

window.redistributeFromNode = function(nodeId) {
    console.log(`Redistributing shards from node: ${nodeId}`);
    
    // Close the modal
    document.getElementById(`node-details-${nodeId}`).remove();
    
    alert(`Shards from ${nodeId} are being redistributed to optimize network load`);
};

window.viewNodeLogs = function(nodeId) {
    console.log(`Viewing logs for node: ${nodeId}`);
    
    // Close the current modal
    document.getElementById(`node-details-${nodeId}`).remove();
    
    // Create logs modal
    const logsModal = document.createElement('div');
    logsModal.className = 'modal-overlay';
    logsModal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h4><i class="fas fa-file-alt"></i> Node Logs: ${nodeId}</h4>
                <button class="modal-close" onclick="this.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div class="logs-container" style="background: #f8fafc; padding: 16px; border-radius: 6px; font-family: monospace; max-height: 400px; overflow-y: auto;">
                    <div>[2024-01-15 10:32:15] INFO: Shard replication completed for file_chunk_127</div>
                    <div>[2024-01-15 10:31:45] INFO: New connection established from peer 192.168.1.45</div>
                    <div>[2024-01-15 10:30:22] INFO: Integrity validation passed for 15 shards</div>
                    <div>[2024-01-15 10:29:18] WARN: High CPU usage detected (87%)</div>
                    <div>[2024-01-15 10:28:35] INFO: Bandwidth optimization applied</div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                <button class="btn primary">Download Full Logs</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(logsModal);
    logsModal.style.display = 'flex';
};

// Initialize visualization when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.shardVisualization = new ShardVisualization();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ShardVisualization;
}