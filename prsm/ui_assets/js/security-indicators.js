/**
 * Security Status Indicators JavaScript
 * Handles real-time security monitoring and user interactions
 */

class SecurityDashboard {
    constructor() {
        this.updateInterval = 10000; // 10 seconds
        this.securityData = {
            overallScore: 98.7,
            postQuantumStatus: 'active',
            signatures: {
                count: 1247,
                verificationRate: 99.9
            },
            accessControl: {
                activeRules: 847,
                authorizedUsers: 23,
                pendingApprovals: 2
            },
            shardIntegrity: {
                verified: 1247,
                corrupted: 1
            },
            networkSecurity: {
                tlsEncryption: true,
                ddosProtection: true,
                anonymousRouting: true
            },
            threatDetection: {
                activeAlerts: 2,
                recentThreats: []
            }
        };
        
        this.initializeComponents();
        this.startRealTimeUpdates();
    }

    initializeComponents() {
        this.updateSecurityScore();
        this.bindEventListeners();
        this.loadSecurityData();
        this.initializeTimelineFilters();
    }

    bindEventListeners() {
        // Timeline filter buttons
        const filterButtons = document.querySelectorAll('.filter-btn');
        filterButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.filterTimeline(e.target.dataset.filter);
            });
        });

        // Threat investigation buttons
        document.querySelectorAll('.investigate-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const threatId = e.target.closest('.alert-item').dataset.threatId;
                this.investigateThreat(threatId);
            });
        });
    }

    updateSecurityScore() {
        const scoreValue = document.getElementById('overall-score');
        const scoreProgress = document.querySelector('.score-progress');
        
        if (scoreValue && scoreProgress) {
            scoreValue.textContent = this.securityData.overallScore;
            
            // Update circle progress (circumference = 2 * π * radius = 2 * π * 40 = 251.2)
            const circumference = 251.2;
            const offset = circumference - (this.securityData.overallScore / 100) * circumference;
            scoreProgress.style.strokeDashoffset = offset;
            
            // Update color based on score
            if (this.securityData.overallScore >= 95) {
                scoreProgress.style.stroke = '#10b981'; // Green
            } else if (this.securityData.overallScore >= 80) {
                scoreProgress.style.stroke = '#f59e0b'; // Yellow
            } else {
                scoreProgress.style.stroke = '#ef4444'; // Red
            }
        }
    }

    loadSecurityData() {
        // Simulate loading real-time security data
        this.updateMetrics();
    }

    updateMetrics() {
        // Add some randomness to simulate real-time updates
        this.securityData.signatures.count += Math.floor(Math.random() * 5);
        this.securityData.signatures.verificationRate += (Math.random() - 0.5) * 0.1;
        this.securityData.signatures.verificationRate = Math.max(99.0, 
            Math.min(100.0, this.securityData.signatures.verificationRate));

        // Update shard integrity
        if (Math.random() < 0.1) { // 10% chance of change
            this.securityData.shardIntegrity.verified += Math.floor((Math.random() - 0.5) * 10);
        }

        // Update overall security score based on various factors
        this.calculateOverallScore();
        this.updateSecurityScore();
        this.updateDisplayedMetrics();
    }

    calculateOverallScore() {
        let score = 100;
        
        // Deduct for corrupted shards
        if (this.securityData.shardIntegrity.corrupted > 0) {
            score -= this.securityData.shardIntegrity.corrupted * 2;
        }
        
        // Deduct for pending approvals
        if (this.securityData.accessControl.pendingApprovals > 0) {
            score -= this.securityData.accessControl.pendingApprovals * 0.5;
        }
        
        // Deduct for active threats
        if (this.securityData.threatDetection.activeAlerts > 0) {
            score -= this.securityData.threatDetection.activeAlerts * 1;
        }
        
        // Factor in signature verification rate
        score = score * (this.securityData.signatures.verificationRate / 100);
        
        this.securityData.overallScore = Math.max(0, Math.min(100, score));
    }

    updateDisplayedMetrics() {
        // Update signature metrics
        const sigCountElements = document.querySelectorAll('.metric-value');
        sigCountElements.forEach(element => {
            if (element.textContent.includes('1,247')) {
                element.textContent = this.securityData.signatures.count.toLocaleString();
            }
            if (element.textContent.includes('99.9%')) {
                element.textContent = this.securityData.signatures.verificationRate.toFixed(1) + '%';
            }
        });

        // Update shard integrity
        const verifiedShardsElements = document.querySelectorAll('.stat-value');
        verifiedShardsElements.forEach(element => {
            if (element.textContent === '1,247') {
                element.textContent = this.securityData.shardIntegrity.verified.toLocaleString();
            }
        });

        // Update health bar
        const healthFill = document.querySelector('.health-fill');
        if (healthFill) {
            healthFill.style.width = this.securityData.signatures.verificationRate + '%';
        }
    }

    startRealTimeUpdates() {
        setInterval(() => {
            this.updateMetrics();
        }, this.updateInterval);
    }

    filterTimeline(filter) {
        // Update active filter button
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-filter="${filter}"]`).classList.add('active');

        // Filter timeline items
        const timelineItems = document.querySelectorAll('.timeline-item');
        timelineItems.forEach(item => {
            if (filter === 'all') {
                item.style.display = 'flex';
            } else {
                const itemType = item.classList.contains(filter) ? 'flex' : 'none';
                item.style.display = itemType;
            }
        });
    }

    initializeTimelineFilters() {
        // Set default filter
        this.filterTimeline('all');
    }

    investigateThreat(threatId) {
        console.log(`Investigating threat: ${threatId}`);
        
        // Create investigation modal
        const modal = this.createInvestigationModal(threatId);
        document.body.appendChild(modal);
        modal.style.display = 'flex';
    }

    createInvestigationModal(threatId) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.id = `threat-investigation-${threatId}`;
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h4><i class="fas fa-search"></i> Threat Investigation: ${threatId}</h4>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="investigation-details">
                        <div class="detail-section">
                            <h6>Threat Details</h6>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <label>Threat ID:</label>
                                    <span>${threatId}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Severity:</label>
                                    <span class="severity high">High</span>
                                </div>
                                <div class="detail-item">
                                    <label>Source IP:</label>
                                    <span>192.168.1.***</span>
                                </div>
                                <div class="detail-item">
                                    <label>Detection Time:</label>
                                    <span>2 minutes ago</span>
                                </div>
                            </div>
                        </div>
                        <div class="detail-section">
                            <h6>Mitigation Actions</h6>
                            <div class="action-buttons">
                                <button class="action-btn danger" onclick="blockThreat('${threatId}')">
                                    <i class="fas fa-ban"></i> Block Source
                                </button>
                                <button class="action-btn warning" onclick="quarantineThreat('${threatId}')">
                                    <i class="fas fa-shield-alt"></i> Quarantine
                                </button>
                                <button class="action-btn secondary" onclick="ignoreThreat('${threatId}')">
                                    <i class="fas fa-eye-slash"></i> Ignore
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                    <button class="btn primary" onclick="generateThreatReport('${threatId}')">Generate Report</button>
                </div>
            </div>
        `;
        
        return modal;
    }
}

// Global functions for UI interactions
window.rotateKeys = function() {
    console.log('Initiating post-quantum key rotation...');
    
    // Simulate key rotation process
    const rotateBtn = event.target;
    const originalText = rotateBtn.innerHTML;
    
    rotateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Rotating...';
    rotateBtn.disabled = true;
    
    setTimeout(() => {
        rotateBtn.innerHTML = originalText;
        rotateBtn.disabled = false;
        
        // Update last rotation time
        const rotationElements = document.querySelectorAll('.metric-value');
        rotationElements.forEach(element => {
            if (element.textContent.includes('hours ago')) {
                element.textContent = 'Just now';
            }
        });
        
        console.log('Key rotation completed successfully');
    }, 3000);
};

window.viewKeyDetails = function() {
    console.log('Viewing key details...');
    
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h4><i class="fas fa-key"></i> Post-Quantum Key Details</h4>
                <button class="modal-close" onclick="this.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div class="key-details">
                    <div class="key-detail">
                        <label>Algorithm:</label>
                        <span>Kyber-1024 KEM</span>
                    </div>
                    <div class="key-detail">
                        <label>Key Size:</label>
                        <span>1568 bytes</span>
                    </div>
                    <div class="key-detail">
                        <label>Security Level:</label>
                        <span>NIST Level 5 (highest)</span>
                    </div>
                    <div class="key-detail">
                        <label>Quantum Resistance:</label>
                        <span>Yes (against Shor's algorithm)</span>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    modal.style.display = 'flex';
};

window.approveAccess = function(userId) {
    console.log(`Approving access for user: ${userId}`);
    
    // Remove the approval item from the UI
    const approvalItem = event.target.closest('.approval-item');
    approvalItem.style.opacity = '0.5';
    
    setTimeout(() => {
        approvalItem.remove();
        
        // Update pending count
        const pendingElements = document.querySelectorAll('.stat-number');
        pendingElements.forEach(element => {
            if (element.textContent === '2') {
                element.textContent = '1';
            }
        });
    }, 500);
};

window.denyAccess = function(userId) {
    console.log(`Denying access for user: ${userId}`);
    
    // Remove the approval item from the UI
    const approvalItem = event.target.closest('.approval-item');
    approvalItem.style.opacity = '0.5';
    
    setTimeout(() => {
        approvalItem.remove();
        
        // Update pending count
        const pendingElements = document.querySelectorAll('.stat-number');
        pendingElements.forEach(element => {
            if (element.textContent === '2') {
                element.textContent = '1';
            }
        });
    }, 500);
};

window.validateAllShards = function() {
    console.log('Starting comprehensive shard validation...');
    
    const validateBtn = event.target;
    const originalText = validateBtn.innerHTML;
    
    validateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
    validateBtn.disabled = true;
    
    setTimeout(() => {
        validateBtn.innerHTML = originalText;
        validateBtn.disabled = false;
        console.log('Shard validation completed - all shards verified');
    }, 5000);
};

window.repairCorrupted = function() {
    console.log('Initiating shard repair process...');
    
    const repairBtn = event.target;
    const originalText = repairBtn.innerHTML;
    
    repairBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Repairing...';
    repairBtn.disabled = true;
    
    setTimeout(() => {
        repairBtn.innerHTML = originalText;
        repairBtn.disabled = false;
        
        // Update corrupted count
        const corruptedElements = document.querySelectorAll('.stat-value');
        corruptedElements.forEach(element => {
            if (element.textContent === '1') {
                element.textContent = '0';
            }
        });
        
        console.log('Corrupted shards repaired successfully');
    }, 4000);
};

window.blockThreat = function(threatId) {
    console.log(`Blocking threat: ${threatId}`);
    
    // Close the investigation modal
    document.getElementById(`threat-investigation-${threatId}`).remove();
    
    // Show confirmation
    alert('Threat source has been blocked successfully.');
};

window.quarantineThreat = function(threatId) {
    console.log(`Quarantining threat: ${threatId}`);
    
    // Close the investigation modal
    document.getElementById(`threat-investigation-${threatId}`).remove();
    
    // Show confirmation
    alert('Threat has been quarantined for further analysis.');
};

window.ignoreThreat = function(threatId) {
    console.log(`Ignoring threat: ${threatId}`);
    
    // Close the investigation modal
    document.getElementById(`threat-investigation-${threatId}`).remove();
    
    // Show confirmation
    alert('Threat has been marked as false positive.');
};

window.generateThreatReport = function(threatId) {
    console.log(`Generating threat report for: ${threatId}`);
    
    // Simulate report generation
    alert('Threat analysis report has been generated and saved to secure storage.');
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.securityDashboard = new SecurityDashboard();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SecurityDashboard;
}