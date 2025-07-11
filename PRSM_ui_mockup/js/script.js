// === PRSM UI MOCKUP INITIALIZATION ===
console.log('🔄 PRSM UI Mockup - Script loading...');

// Test function to check if buttons are working
window.testButtonFunctionality = function() {
    console.log('🔍 Testing button functionality...');
    
    // Test tab buttons
    const tabs = document.querySelectorAll('.tab-navigation .tab');
    console.log('📝 Found', tabs.length, 'tabs');
    tabs.forEach((tab, index) => {
        console.log(`  Tab ${index}: ${tab.textContent.trim()} -> ${tab.dataset.target}`);
    });
    
    // Test asset type cards
    const assetCards = document.querySelectorAll('.asset-type-card');
    console.log('📊 Found', assetCards.length, 'asset type cards');
    assetCards.forEach((card, index) => {
        console.log(`  Asset card ${index}: ${card.textContent.trim()} -> ${card.dataset.type}`);
    });
    
    // Test profile button
    const profileBtn = document.getElementById('profile-button');
    console.log('👤 Profile button:', profileBtn ? 'found' : 'not found');
    
    // Test data work functionality
    const dataWorkCard = document.querySelector('.asset-type-card[data-type="data_work"]');
    const dataWorkHub = document.getElementById('data-work-hub');
    console.log('💼 Data work card:', dataWorkCard ? 'found' : 'not found');
    console.log('💼 Data work hub:', dataWorkHub ? 'found' : 'not found');
    
    return '✅ Test complete - check console logs above';
};

// Quick test functions
window.testTabs = function() {
    const tabs = document.querySelectorAll('.tab-navigation .tab');
    if (tabs.length === 0) {
        console.error('No tabs found!');
        return;
    }
    
    console.log('Testing first tab click...');
    tabs[0].click();
    console.log('Tab click test completed.');
};

window.testMarketplace = function() {
    const marketplaceTab = document.querySelector('.tab-navigation .tab[data-target="marketplace-content"]');
    if (marketplaceTab) {
        console.log('Testing marketplace tab...');
        marketplaceTab.click();
        console.log('Marketplace tab test completed.');
    } else {
        console.error('Marketplace tab not found!');
    }
};

window.testDataWork = function() {
    const dataWorkCard = document.querySelector('.asset-type-card[data-type="data_work"]');
    if (dataWorkCard) {
        console.log('Testing data work card...');
        dataWorkCard.click();
        console.log('Data work card test completed.');
    } else {
        console.error('Data work card not found!');
    }
};

// Test function to verify discovery title updates
window.testDiscoveryTitle = function() {
    const assetTypes = ['ai_model', 'dataset', 'agent_workflow', 'mcp_tool', 'data_work'];
    const discoveryTitle = document.getElementById('discovery-title');
    
    console.log('Testing discovery title updates...');
    console.log('Current title:', discoveryTitle?.textContent || 'not found');
    
    assetTypes.forEach(type => {
        const card = document.querySelector(`[data-type="${type}"]`);
        if (card) {
            console.log(`Testing ${type}...`);
            updateMarketplaceStats(type, card);
            console.log('Title after update:', discoveryTitle?.textContent);
        }
    });
    
    return 'Discovery title test completed - check console';
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 PRSM UI Mockup - DOM ready, initializing...');
    
    // Add test function to global scope for debugging
    window.testButtons = window.testButtonFunctionality;
    // --- DOM References ---
    const leftPanel = document.getElementById('left-panel');
    const rightPanel = document.getElementById('right-panel');
    const resizer = document.getElementById('drag-handle');
    const leftPanelToggleBtn = document.getElementById('left-panel-toggle');
    const logoImage = document.getElementById('logo-img'); // Logo in right header
    const rightPanelTabs = document.querySelectorAll('.tab-navigation .tab');
    const rightPanelContentSections = document.querySelectorAll('.right-panel-content-area .content-section');
    
    // Debug: Check if elements exist
    console.log('✅ Elements found:', {
        leftPanel: !!leftPanel,
        rightPanel: !!rightPanel,
        rightPanelTabs: rightPanelTabs.length,
        leftPanelToggleBtn: !!leftPanelToggleBtn
    });

    // Profile Dropdown Elements
    const profileButton = document.getElementById('profile-button');
    const profileDropdown = document.getElementById('profile-dropdown');
    const dropdownThemeToggle = document.getElementById('dropdown-theme-toggle');
    const dropdownSettings = document.getElementById('dropdown-settings');
    const dropdownSignOut = document.getElementById('dropdown-sign-out');

    // Upload Button Elements
    const uploadButton = document.getElementById('upload-button');
    const uploadDropdown = document.getElementById('upload-dropdown');
    const uploadComputer = document.getElementById('upload-computer');
    const uploadMyFiles = document.getElementById('upload-my-files');

    // Conversation History Elements
    const historyToggleBtn = document.getElementById('history-toggle-btn');
    const historySidebar = document.getElementById('conversation-history-sidebar');


    // Settings Elements
    const settingsContent = document.getElementById('settings-content');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    let apiSettingFields = []; // Will populate later
    let hasUnsavedChanges = false;

    // Integration Elements - Initialize references to avoid null errors
    let integrationElements = {};
    // --- Local Storage Keys ---
    const themeKey = 'prsmThemePreference';
    const leftPanelWidthKey = 'prsmLeftPanelWidth';
    const leftPanelCollapsedKey = 'prsmLeftPanelCollapsed';
    const historySidebarHiddenKey = 'prsmHistorySidebarHidden'; // New key

    // --- Helper: Get CSS Variable ---
    const getCssVariable = (variableName) => getComputedStyle(document.documentElement).getPropertyValue(variableName).trim();

    // --- Theme Management ---
    const applyTheme = (theme) => {
        const themeToggleLink = document.getElementById('dropdown-theme-toggle');
        if (themeToggleLink) {
            if (theme === 'light') {
                document.body.classList.add('light-theme');
                themeToggleLink.innerHTML = '<i class="fas fa-moon fa-fw"></i> Toggle Dark Mode';
            } else { // Dark theme
                document.body.classList.remove('light-theme');
                themeToggleLink.innerHTML = '<i class="fas fa-sun fa-fw"></i> Toggle Light Mode';
            }
        }
        if (logoImage) {
            logoImage.src = theme === 'light' ? 'assets/PRSM_Logo_Light.png' : 'assets/PRSM_Logo_Dark.png';
        }
    };

    const toggleTheme = () => {
        const isLight = document.body.classList.contains('light-theme');
        const newTheme = isLight ? 'dark' : 'light';
        applyTheme(newTheme);
        localStorage.setItem(themeKey, newTheme);
    };

    // Apply saved theme on initial load
    const savedTheme = localStorage.getItem(themeKey) || 'dark';
    console.log('Applying saved theme:', savedTheme);
    applyTheme(savedTheme);


    // --- Profile Dropdown Logic ---
    const toggleProfileDropdown = (show) => {
        if (!profileDropdown) return;
        const shouldShow = show ?? !profileDropdown.classList.contains('show');
        profileDropdown.classList.toggle('show', shouldShow);
    };

    if (profileButton) {
        console.log('Profile button found, adding click handler');
        profileButton.addEventListener('click', (e) => {
            e.stopPropagation();
            console.log('Profile button clicked');
            toggleProfileDropdown();
        });
    } else {
        console.error('Profile button not found');
    }

    document.addEventListener('click', (e) => {
        if (profileDropdown && profileDropdown.classList.contains('show')) {
            if (!profileButton.contains(e.target) && !profileDropdown.contains(e.target)) {
                toggleProfileDropdown(false);
            }
        }
    });

    if (dropdownThemeToggle) {
        dropdownThemeToggle.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('🌙 Theme toggle clicked');
            toggleTheme();
            toggleProfileDropdown(false);
        });
    } else {
        console.error('Theme toggle not found');
    }

    if (dropdownSettings) {
        dropdownSettings.addEventListener('click', (e) => {
            e.preventDefault();
            console.log("Settings dropdown item clicked");
            showRightPanelSection('settings-content');
            toggleProfileDropdown(false);
            // queryApiSettingFields(); // This is already called by showRightPanelSection logic if target is settings
        });
    } else {
        console.error('Settings dropdown not found');
    }

    if (dropdownSignOut) {
        dropdownSignOut.addEventListener('click', (e) => {
            e.preventDefault();
            console.log("Sign Out clicked");
            toggleProfileDropdown(false);
        });
    }

    // --- Upload Button Dropdown Logic ---
    const toggleUploadDropdown = (show) => {
        if (!uploadDropdown) return;
        const shouldShow = show ?? !uploadDropdown.classList.contains('show');
        uploadDropdown.classList.toggle('show', shouldShow);
    };

    if (uploadButton) {
        uploadButton.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleUploadDropdown();
        });
    }

    // Handle upload options clicks
    if (uploadComputer) {
        uploadComputer.addEventListener('click', (e) => {
            e.preventDefault();
            console.log("Upload from Computer clicked");
            // Implement file upload from computer functionality here
            toggleUploadDropdown(false);
        });
    }

    if (uploadMyFiles) {
        uploadMyFiles.addEventListener('click', (e) => {
            e.preventDefault();
            console.log("Upload from My Files clicked");
            // Implement file selection from My Files functionality here
            toggleUploadDropdown(false);
            // Optionally show the My Files tab
            showRightPanelSection('my-files-content');
        });
    }

    // Close upload dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (uploadDropdown && uploadDropdown.classList.contains('show')) {
            if (!uploadButton.contains(e.target) && !uploadDropdown.contains(e.target)) {
                toggleUploadDropdown(false);
            }
        }
    });

    // --- Conversation History Sidebar Toggle ---
    const toggleHistorySidebar = (hide) => {
        if (!historySidebar) {
            console.error('History sidebar element not found');
            return;
        }
        
        const shouldHide = hide ?? !historySidebar.classList.contains('hidden');
        console.log(`Toggling history sidebar. Should hide: ${shouldHide}`);
        
        // Toggle the hidden class
        historySidebar.classList.toggle('hidden', shouldHide);
        
        // Update the toggle button icon and title
        if (historyToggleBtn) {
            const icon = historyToggleBtn.querySelector('i');
            if (icon) {
                if (shouldHide) {
                    icon.className = 'fas fa-chevron-right';
                    historyToggleBtn.title = 'Show History';
                    // Add tooltip to the entire sidebar when hidden
                    historySidebar.title = 'Click to show conversation history';
                } else {
                    icon.className = 'fas fa-chevron-left';
                    historyToggleBtn.title = 'Hide History';
                    // Remove tooltip when visible
                    historySidebar.removeAttribute('title');
                }
            }
        }
        
        // Save state to localStorage
        localStorage.setItem(historySidebarHiddenKey, shouldHide ? 'true' : 'false');
        console.log(`History sidebar toggled. Hidden: ${shouldHide}`);
    };

    if (historyToggleBtn) {
        console.log('Attaching history toggle event listener');
        historyToggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('History toggle button clicked');
            toggleHistorySidebar();
        });
    } else {
        console.error('History toggle button not found');
    }

    // Make the entire hidden history sidebar clickable
    if (historySidebar) {
        historySidebar.addEventListener('click', (e) => {
            // Only trigger if the sidebar is hidden and click is on header area
            if (historySidebar.classList.contains('hidden')) {
                // Make sure we're not clicking on content that might still be visible
                if (e.target.closest('.history-persistent-header') || e.target === historySidebar) {
                    console.log('Hidden history sidebar clicked');
                    toggleHistorySidebar();
                }
            }
        });
    }

    // Apply saved history state on load, defaulting to visible
    const savedHistoryHidden = localStorage.getItem(historySidebarHiddenKey) === 'true';
    if (historySidebar) {
        // Explicitly set the class based on the saved state or default (visible)
        historySidebar.classList.toggle('hidden', savedHistoryHidden);
        // Set correct icon based on state
        if (historyToggleBtn) {
            const icon = historyToggleBtn.querySelector('i');
            if (icon) {
                icon.className = savedHistoryHidden ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
            }
            historyToggleBtn.title = savedHistoryHidden ? 'Show History' : 'Hide History';
        }
        // Add tooltip to sidebar if it starts hidden
        if (savedHistoryHidden) {
            historySidebar.title = 'Click to show conversation history';
        }
    }


    // --- Left Panel Toggle (UPDATED to handle history) ---
    const toggleLeftPanel = (collapse) => {
        if (!leftPanel || !rightPanel || !resizer) {
            console.error('Panel elements not found:', { leftPanel, rightPanel, resizer });
            return;
        }

        const shouldCollapse = collapse ?? !leftPanel.classList.contains('collapsed');
        const collapsedWidth = getCssVariable('--left-panel-collapsed-width') || '45px'; // Fallback
        const resizerWidth = resizer.offsetWidth;

        console.log(`Toggling left panel. Should collapse: ${shouldCollapse}`);
        if (shouldCollapse) {
            // Store current width before collapsing (if not already collapsed)
            if (!leftPanel.classList.contains('collapsed')) {
                 const currentWidth = leftPanel.offsetWidth > 0 ? `${leftPanel.offsetWidth}px` : (localStorage.getItem(leftPanelWidthKey) || '50%');
                 localStorage.setItem(leftPanelWidthKey, currentWidth);
            }

            leftPanel.classList.add('collapsed');
            leftPanel.style.width = collapsedWidth;
            leftPanel.style.minWidth = collapsedWidth;
            leftPanel.style.maxWidth = collapsedWidth;
            rightPanel.style.width = `calc(100% - ${collapsedWidth})`;
            rightPanel.style.minWidth = 'auto';
            resizer.style.display = 'none';
            localStorage.setItem(leftPanelCollapsedKey, 'true');
            // Add helpful tooltip for collapsed state
            leftPanel.title = 'Click to expand conversation panel';
            // Update toggle button icon
            if (leftPanelToggleBtn) {
                const icon = leftPanelToggleBtn.querySelector('i');
                if (icon) {
                    icon.className = 'fas fa-chevron-right';
                }
            }
            // Note: History sidebar maintains its own state when main panel collapses

        } else {
            leftPanel.classList.remove('collapsed');
            const restoredWidth = localStorage.getItem(leftPanelWidthKey) || '50%';
            leftPanel.style.width = restoredWidth;
            leftPanel.style.minWidth = '250px'; // Restore minimum width
            leftPanel.style.maxWidth = 'none'; // Remove max width restriction
            rightPanel.style.width = `calc(100% - ${restoredWidth} - ${resizerWidth}px)`;
            rightPanel.style.minWidth = '350px'; // Restore minimum width
            resizer.style.display = 'block';
            localStorage.setItem(leftPanelCollapsedKey, 'false');
            // Remove tooltip when expanded
            leftPanel.removeAttribute('title');
            // Update toggle button icon
            if (leftPanelToggleBtn) {
                const icon = leftPanelToggleBtn.querySelector('i');
                if (icon) {
                    icon.className = 'fas fa-chevron-left';
                }
            }
            // Restore history sidebar state (it might have been open before collapsing)
             if (historySidebar && localStorage.getItem(historySidebarHiddenKey) !== 'true') {
                 historySidebar.classList.remove('hidden');
             }
        }
    };

    if (leftPanelToggleBtn) {
        console.log('Attaching left panel toggle event listener');
        leftPanelToggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Left panel toggle button clicked');
            toggleLeftPanel();
        });
    } else {
        console.error('Left panel toggle button not found');
    }

    // Make the entire collapsed left panel clickable for better UX
    if (leftPanel) {
        leftPanel.addEventListener('click', (e) => {
            // Only trigger if the panel is collapsed and the click is on the panel itself or header
            if (leftPanel.classList.contains('collapsed')) {
                // Prevent bubbling if clicked on specific interactive elements
                if (!e.target.closest('.conversation-main-area')) {
                    console.log('Collapsed left panel clicked');
                    toggleLeftPanel();
                }
            }
        });
    }

     // Apply saved collapsed state on load (UPDATED)
     const savedCollapsed = localStorage.getItem(leftPanelCollapsedKey) === 'true';
     if (savedCollapsed && leftPanel && rightPanel && resizer) {
         const collapsedWidth = getCssVariable('--left-panel-collapsed-width') || '45px';
         leftPanel.classList.add('collapsed');
         leftPanel.style.width = collapsedWidth;
         leftPanel.style.minWidth = collapsedWidth;
         leftPanel.style.maxWidth = collapsedWidth;
         rightPanel.style.width = `calc(100% - ${collapsedWidth})`;
         rightPanel.style.minWidth = 'auto';
         resizer.style.display = 'none';
         // Add helpful tooltip for collapsed state
         leftPanel.title = 'Click to expand conversation panel';
         // Set correct icon for collapsed state
         if (leftPanelToggleBtn) {
             const icon = leftPanelToggleBtn.querySelector('i');
             if (icon) {
                 icon.className = 'fas fa-chevron-right';
             }
         }
         // Note: History sidebar maintains its own state independent of main panel
     }


    // --- Right Panel Tab Switching ---
    const showRightPanelSection = (targetId) => {
        console.log('Switching to section:', targetId);
        rightPanelTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.target === targetId);
        });
        rightPanelContentSections.forEach(section => {
            section.classList.toggle('active', section.id === targetId);
        });
        if (rightPanel) {
            const contentArea = rightPanel.querySelector('.right-panel-content-area');
            if (contentArea) contentArea.scrollTop = 0;
        }
    };

    console.log(`Found ${rightPanelTabs.length} right panel tabs`);
    rightPanelTabs.forEach(tab => {
        console.log('Adding click handler to tab:', tab.dataset.target);
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = tab.dataset.target; // Get targetId here
            console.log(`Tab clicked: ${targetId}`);
            // --- Unsaved Changes Check ---
            // Updated check: Only prompt if leaving settings with unsaved changes
            if (hasUnsavedChanges && settingsContent.classList.contains('active') && targetId !== 'settings-content') {
                const proceed = confirm("You have unsaved changes in Settings. Do you want to discard them and switch tabs?");
                if (!proceed) {
                    // console.log("Tab switch cancelled due to unsaved changes."); // Removed log
                    return; // Stop the tab switch if user cancels
                }
                // If user proceeds (discards changes)
                resetUnsavedChangesState();
                // console.log("Unsaved changes discarded."); // Removed log
            }
            // --- End Unsaved Changes Check ---

            console.log(`Showing section: ${targetId}`);
            showRightPanelSection(targetId); // Keep only ONE call
            // If switching *to* settings, ensure fields are queried
            if (targetId === 'settings-content') {
                queryApiSettingFields();
            }
        });
    });

    if (!document.querySelector('.right-panel-content-area .content-section.active')) {
        showRightPanelSection('tasks-content');
    }


    // --- Panel Resizing ---
    let isResizing = false;
    let startX, startWidthLeft;

    if (resizer && leftPanel && rightPanel) {
        // Apply saved width on load (only if not collapsed)
        if (!savedCollapsed) {
            const savedWidth = localStorage.getItem(leftPanelWidthKey);
            if (savedWidth) {
                leftPanel.style.width = savedWidth;
                rightPanel.style.width = `calc(100% - ${savedWidth} - ${resizer.offsetWidth}px)`;
            }
        }

        resizer.addEventListener('mousedown', (e) => {
            if (leftPanel.classList.contains('collapsed')) return; // Prevent dragging when collapsed

            isResizing = true;
            startX = e.clientX;
            startWidthLeft = leftPanel.offsetWidth;

            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        });

        const handleMouseMove = (e) => {
            if (!isResizing) return;

            const currentX = e.clientX;
            const deltaX = currentX - startX;
            let newLeftWidth = startWidthLeft + deltaX;

            const minLeftWidth = parseInt(getCssVariable('--left-panel-min-width') || '250'); // Use CSS var or fallback
            const minRightWidth = parseInt(getCssVariable('--right-panel-min-width') || '350');
            const containerWidth = leftPanel.parentElement.offsetWidth;
            const resizerWidth = resizer.offsetWidth;
            const collapsedWidthThreshold = parseInt(getCssVariable('--left-panel-collapsed-width') || '50');

            // Prevent making left panel smaller than its collapsed width during resize
            if (newLeftWidth < collapsedWidthThreshold + 20) { // Add a small buffer
                 newLeftWidth = collapsedWidthThreshold + 20;
            }
            // Clamp based on min widths
            if (newLeftWidth < minLeftWidth) {
                newLeftWidth = minLeftWidth;
            }
            if (containerWidth - newLeftWidth - resizerWidth < minRightWidth) {
                newLeftWidth = containerWidth - minRightWidth - resizerWidth;
            }

            leftPanel.style.width = `${newLeftWidth}px`;
            rightPanel.style.width = `calc(100% - ${newLeftWidth}px - ${resizerWidth}px)`;
        };

        const handleMouseUp = () => {
            if (isResizing) {
                isResizing = false;
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
                document.body.style.cursor = '';
                document.body.style.userSelect = '';

                if (!leftPanel.classList.contains('collapsed')) {
                    localStorage.setItem(leftPanelWidthKey, leftPanel.style.width);
                }
            }
        };
    }



    // --- API Settings Save Logic ---
    const queryApiSettingFields = () => {
        if (settingsContent) {
            // Query fields only when settings tab is potentially visible
            apiSettingFields = settingsContent.querySelectorAll('.api-setting-field');
            // Add listeners only if they haven't been added before or if fields changed
            // Simple approach: remove and re-add listeners
            apiSettingFields.forEach(field => {
                field.removeEventListener('input', handleSettingChange);
                field.removeEventListener('change', handleSettingChange);
                field.addEventListener('input', handleSettingChange);
                field.addEventListener('change', handleSettingChange); // For select elements
            });
        }
    };

    const handleSettingChange = () => {
        if (!hasUnsavedChanges) {
            hasUnsavedChanges = true;
            if (saveSettingsBtn) {
                saveSettingsBtn.classList.add('unsaved-changes');
                // Optional: Update button text or add an indicator
                // saveSettingsBtn.textContent = "Save API Settings*"; 
            }
        }
    };

    const resetUnsavedChangesState = () => {
        hasUnsavedChanges = false;
        if (saveSettingsBtn) {
            saveSettingsBtn.classList.remove('unsaved-changes');
            // Optional: Reset button text if changed
            // saveSettingsBtn.textContent = "Save API Settings"; 
        }
    };

    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', () => {
            console.log("Saving API Settings..."); // Placeholder for actual save logic
            // Simulate save
            resetUnsavedChangesState();
            alert("API Settings Saved (mock)!"); // User feedback
        });
    }

    // Initial query in case settings tab is active on load
    if (document.querySelector('#settings-content.active')) {
        queryApiSettingFields();
    }

    // Add listener to the settings dropdown item to ensure fields are queried when tab is shown - REDUNDANT LISTENER REMOVED
    // if (dropdownSettings) {
    //     dropdownSettings.addEventListener('click', queryApiSettingFields); // Keep commented out
    // }

    // === PRSM API Integration ===
    
    // Wait for API client to be available
    const initializeAPIIntegration = () => {
        if (typeof window.prsmAPI === 'undefined') {
            setTimeout(initializeAPIIntegration, 100);
            return;
        }

        console.log('🚀 Initializing PRSM API integration');
        
        // Initialize conversation interface
        initializeConversationInterface();
        
        // Load conversation history
        loadConversationHistory();
        
        // Load tokenomics data for display
        loadTokenomicsData();
        
        // Load tasks data
        loadTasksData();
        
        // Load files data
        loadFilesData();
    };

    const initializeConversationInterface = () => {
        const sendButton = document.getElementById('send-message-btn');
        const promptTextarea = document.getElementById('prompt-input');
        const responseArea = document.querySelector('.response-area');

        if (sendButton && promptTextarea && responseArea) {
            // Send message on button click
            sendButton.addEventListener('click', () => sendMessage());
            
            // Send message on Ctrl+Enter or Cmd+Enter
            promptTextarea.addEventListener('keydown', (e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    e.preventDefault();
                    sendMessage();
                }
            });

            console.log('✅ Conversation interface initialized');
        }

        // Set up WebSocket event handlers for real-time features
        setupWebSocketEventHandlers();
    };

    const setupWebSocketEventHandlers = () => {
        if (!window.prsmAPI) return;

        // Handle real-time AI response chunks
        window.prsmAPI.onWebSocketMessage('ai_response_chunk', (data) => {
            console.log('🔄 Received AI response chunk');
            // Handled automatically by the API client
        });

        // Handle complete AI responses
        window.prsmAPI.onWebSocketMessage('ai_response_complete', (data) => {
            console.log('✅ AI response complete');
            // Update conversation history or other UI elements as needed
            loadConversationHistory();
        });

        // Handle real-time notifications
        window.prsmAPI.onWebSocketMessage('notification', (data) => {
            console.log('📢 Received notification:', data);
            // Additional UI handling if needed beyond the built-in notifications
        });

        // Handle real-time tokenomics updates
        window.prsmAPI.onWebSocketMessage('tokenomics_update', (data) => {
            console.log('💰 Received tokenomics update');
            // Updates are handled automatically by the API client
        });

        // Handle real-time task updates
        window.prsmAPI.onWebSocketMessage('task_update', (data) => {
            console.log('📋 Received task update:', data.action);
            // Updates are handled automatically by the API client
        });

        // Handle real-time file updates
        window.prsmAPI.onWebSocketMessage('file_update', (data) => {
            console.log('📁 Received file update:', data.action);
            // Updates are handled automatically by the API client
        });

        console.log('🔌 WebSocket event handlers initialized');
    };

    const sendMessage = async () => {
        const promptTextarea = document.getElementById('prompt-input');
        const responseArea = document.querySelector('.response-area');
        const sendButton = document.getElementById('send-message-btn');

        if (!promptTextarea || !responseArea) return;

        const message = promptTextarea.value.trim();
        if (!message) return;

        try {
            // Clear the input
            promptTextarea.value = '';
            
            // Try WebSocket streaming first if available
            if (window.prsmAPI.connected && window.prsmAPI.sendMessageWithStreaming) {
                const response = await window.prsmAPI.sendMessageWithStreaming(message);
                
                if (response.streaming) {
                    // WebSocket streaming initiated - UI updates handled by WebSocket events
                    console.log('📡 Message sent via WebSocket streaming');
                    return;
                }
            }
            
            // Fallback to REST API
            sendButton.disabled = true;
            sendButton.textContent = 'Sending...';
            
            // Add user message to UI
            addMessageToUI('user', message);
            
            // Send to API
            const response = await window.prsmAPI.sendMessage(message);
            
            if (response.success && response.ai_response) {
                // Add AI response to UI
                addMessageToUI('assistant', response.ai_response.content, response.ai_response);
                
                // Update context display
                updateContextDisplay(response.conversation_status);
            } else {
                addMessageToUI('assistant', 'Sorry, there was an error processing your message.');
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToUI('assistant', 'Error: Could not send message. Using mock mode.');
        } finally {
            // Re-enable send button
            sendButton.disabled = false;
            sendButton.textContent = 'Send (CMD+Enter)';
        }
    };

    const addMessageToUI = (role, content, metadata = {}) => {
        const responseArea = document.querySelector('.response-area');
        if (!responseArea) return;

        // Clear placeholder text if this is the first real message
        if (responseArea.innerHTML.includes('AI responses will appear here...')) {
            responseArea.innerHTML = '';
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="message-role">${role === 'user' ? 'You' : 'PRSM'}</span>
                <span class="message-time">${timestamp}</span>
                ${metadata.model_used ? `<span class="message-model">${metadata.model_used}</span>` : ''}
            </div>
            <div class="message-content">${content}</div>
        `;

        responseArea.appendChild(messageDiv);
        responseArea.scrollTop = responseArea.scrollHeight;
    };

    const updateContextDisplay = (conversationStatus) => {
        if (!conversationStatus) return;

        const progressBar = document.querySelector('.context-progress-bar progress');
        const valueSpan = document.querySelector('.context-progress-bar .value');

        if (progressBar && valueSpan) {
            progressBar.value = conversationStatus.context_used;
            progressBar.max = conversationStatus.context_limit;
            valueSpan.textContent = `${conversationStatus.context_used} / ${conversationStatus.context_limit}`;
        }
    };

    const loadConversationHistory = async () => {
        try {
            const response = await window.prsmAPI.listConversations();
            if (response.success && response.conversations) {
                updateConversationHistory(response.conversations);
            }
        } catch (error) {
            console.warn('Could not load conversation history:', error);
        }
    };

    const updateConversationHistory = (conversations) => {
        const historyList = document.querySelector('.history-list');
        if (!historyList) return;

        historyList.innerHTML = '';
        
        conversations.forEach(conv => {
            const li = document.createElement('li');
            li.innerHTML = `<a href="#" data-conversation-id="${conv.conversation_id}">${conv.title}</a>`;
            historyList.appendChild(li);
        });

        console.log(`📋 Updated conversation history with ${conversations.length} conversations`);
    };

    const loadTokenomicsData = async () => {
        try {
            const response = await window.prsmAPI.getTokenomics();
            if (response.success && response.tokenomics) {
                updateTokenomicsDisplay(response.tokenomics);
            }
        } catch (error) {
            console.warn('Could not load tokenomics data:', error);
        }
    };

    const updateTokenomicsDisplay = (tokenomics) => {
        // Update the tokenomics section with real data
        const tokenomicsContent = document.querySelector('#tokenomics-content .placeholder-box');
        if (!tokenomicsContent) return;

        const { balance, staking, earnings } = tokenomics;
        
        tokenomicsContent.innerHTML = `
            <p>Monitor your FTNS token balance, earnings, and staking.</p>
            <p><strong>Balance:</strong> ${balance.total} FTNS (Available: ${balance.available})</p>
            <p><strong>Staked:</strong> ${staking.staked_amount} FTNS (APY: ${staking.apy}%)</p>
            <p><strong>Earnings Source:</strong> ${earnings.current_status} (Total: ${earnings.total_earned} FTNS)</p>
            <button>Stake Tokens</button> <button>View Transactions</button>
        `;

        console.log('💰 Updated tokenomics display');
    };

    const loadTasksData = async () => {
        try {
            const response = await window.prsmAPI.getTasks();
            if (response.success && response.tasks) {
                updateTasksDisplay(response.tasks, response.statistics);
            }
        } catch (error) {
            console.warn('Could not load tasks data:', error);
        }
    };

    const updateTasksDisplay = (tasks, statistics) => {
        const tasksContent = document.querySelector('#tasks-content .placeholder-box ul');
        if (!tasksContent) return;

        tasksContent.innerHTML = '';
        
        tasks.forEach(task => {
            const li = document.createElement('li');
            li.innerHTML = `
                ${task.title} 
                <button class="inline-btn" onclick="markTaskDone('${task.task_id}')">
                    ${task.status === 'completed' ? 'Completed' : 'Mark Done'}
                </button>
            `;
            tasksContent.appendChild(li);
        });

        console.log(`📋 Updated tasks display with ${tasks.length} tasks`);
    };

    const loadFilesData = async () => {
        try {
            const response = await window.prsmAPI.listFiles();
            if (response.success && response.files) {
                updateFilesDisplay(response.files);
            }
        } catch (error) {
            console.warn('Could not load files data:', error);
        }
    };

    const updateFilesDisplay = (files) => {
        const filesContent = document.querySelector('#my-files-content .placeholder-box ul');
        if (!filesContent) return;

        filesContent.innerHTML = '';
        
        files.forEach(file => {
            const li = document.createElement('li');
            const sizeKB = Math.round(file.size / 1024);
            li.innerHTML = `
                ${file.filename} (${sizeKB} KB)
                <button class="inline-btn">Share</button> 
                <span class="file-status">(${file.privacy})</span>
            `;
            filesContent.appendChild(li);
        });

        console.log(`📁 Updated files display with ${files.length} files`);
    };

    // File upload handling
    const setupFileUpload = () => {
        const uploadComputer = document.getElementById('upload-computer');
        
        if (uploadComputer) {
            uploadComputer.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Create hidden file input
                const fileInput = document.createElement('input');
                fileInput.type = 'file';
                fileInput.multiple = true;
                fileInput.accept = '*/*';
                
                fileInput.onchange = async (event) => {
                    const files = event.target.files;
                    for (let file of files) {
                        try {
                            console.log(`📤 Uploading file: ${file.name}`);
                            const response = await window.prsmAPI.uploadFile(file);
                            
                            if (response.success) {
                                console.log(`✅ File uploaded: ${file.name}`);
                                // Refresh files display
                                loadFilesData();
                            }
                        } catch (error) {
                            console.error(`❌ Upload failed for ${file.name}:`, error);
                        }
                    }
                };
                
                fileInput.click();
            });
        }
    };

    // Settings save handling
    const setupSettingsSave = () => {
        if (saveSettingsBtn) {
            // Remove existing listener and add new one
            saveSettingsBtn.removeEventListener('click', handleSettingsSave);
            saveSettingsBtn.addEventListener('click', handleSettingsSave);
        }
    };

    const handleSettingsSave = async () => {
        console.log("💾 Saving API Settings via PRSM API...");
        
        try {
            // Collect API key data
            const apiKeys = {};
            
            const openaiKey = document.getElementById('openai-key')?.value;
            const anthropicKey = document.getElementById('anthropic-key')?.value;
            const geminiKey = document.getElementById('google-gemini-key')?.value;
            
            if (openaiKey) {
                apiKeys.openai = {
                    key: openaiKey,
                    mode_mapping: document.getElementById('openai-mode-map')?.value || 'dynamic'
                };
            }
            
            if (anthropicKey) {
                apiKeys.anthropic = {
                    key: anthropicKey,
                    mode_mapping: document.getElementById('anthropic-mode-map')?.value || 'dynamic'
                };
            }
            
            if (geminiKey) {
                apiKeys.google_gemini = {
                    key: geminiKey,
                    mode_mapping: document.getElementById('gemini-mode-map')?.value || 'dynamic'
                };
            }

            const settingsData = {
                api_keys: apiKeys,
                ui_preferences: {
                    theme: document.body.classList.contains('light-theme') ? 'light' : 'dark',
                    left_panel_collapsed: leftPanel?.classList.contains('collapsed') || false
                }
            };

            const response = await window.prsmAPI.saveSettings(settingsData);
            
            if (response.success) {
                resetUnsavedChangesState();
                alert(`✅ Settings saved! Configured ${response.api_keys_configured} API keys.`);
            } else {
                alert("❌ Failed to save settings");
            }
            
        } catch (error) {
            console.error("Settings save error:", error);
            alert("⚠️ Settings saved locally (API unavailable)");
            resetUnsavedChangesState();
        }
    };

    // Start API integration
    initializeAPIIntegration();
    
    // Setup additional handlers
    setupFileUpload();
    setupSettingsSave();
    setupIntegrations();
    
    console.log('🎯 PRSM UI fully initialized with API integration');
});

// === INTEGRATION LAYER FUNCTIONALITY ===

// Platform connection states
const platformStates = {
    github: { connected: false, status: 'not_connected' },
    huggingface: { connected: false, status: 'not_connected' },
    ollama: { connected: false, status: 'available' }
};

// Initialize integration functionality
function setupIntegrations() {
    console.log('🔌 Setting up integrations functionality...');
    
    // Initialize integration elements after DOM is ready
    setTimeout(() => {
        initializeIntegrationElements();
        loadInitialIntegrationData();
    }, 100);
}

function initializeIntegrationElements() {
    // Platform connection elements
    integrationElements = {
        github: {
            card: document.querySelector('[data-platform="github"]'),
            status: document.getElementById('github-status'),
            connectBtn: document.getElementById('github-connect-btn'),
            disconnectBtn: document.getElementById('github-disconnect-btn')
        },
        huggingface: {
            card: document.querySelector('[data-platform="huggingface"]'),
            status: document.getElementById('hf-status'),
            connectBtn: document.getElementById('hf-connect-btn'),
            disconnectBtn: document.getElementById('hf-disconnect-btn')
        },
        ollama: {
            card: document.querySelector('[data-platform="ollama"]'),
            status: document.getElementById('ollama-status'),
            connectBtn: document.getElementById('ollama-connect-btn'),
            disconnectBtn: document.getElementById('ollama-disconnect-btn')
        }
    };

    // Search elements
    integrationElements.search = {
        input: document.getElementById('content-search-input'),
        typeSelect: document.getElementById('content-type-select'),
        platformSelect: document.getElementById('platform-select'),
        button: document.getElementById('search-content-btn'),
        results: document.getElementById('search-results')
    };

    // Import history elements
    integrationElements.imports = {
        list: document.getElementById('import-history-list'),
        stats: document.getElementById('import-stats'),
        refreshBtn: document.querySelector('.refresh-btn')
    };

    // Health dashboard elements
    integrationElements.health = {
        overall: document.getElementById('overall-health'),
        connectors: document.getElementById('connector-health'),
        activeImports: document.getElementById('active-imports'),
        checkBtn: document.querySelector('.health-check-btn')
    };

    console.log('✅ Integration elements initialized');
}

async function loadInitialIntegrationData() {
    try {
        // Load system health
        await checkSystemHealth();
        
        // Load import history
        await refreshImportHistory();
        
        // Check connector health
        await updateConnectorHealth();
        
        console.log('✅ Initial integration data loaded');
    } catch (error) {
        console.warn('⚠️ Failed to load initial integration data:', error);
    }
}

// === PLATFORM CONNECTION FUNCTIONS ===

async function connectPlatform(platform) {
    console.log(`🔌 Connecting to ${platform}...`);
    
    try {
        let credentials = {};
        
        if (platform === 'github') {
            // For demo purposes, simulate OAuth flow
            credentials = {
                oauth_credentials: { access_token: 'demo_github_token' }
            };
        } else if (platform === 'huggingface') {
            // For demo purposes, use a placeholder API key
            credentials = {
                api_key: 'demo_hf_token'
            };
        } else if (platform === 'ollama') {
            // Local connection - no credentials needed
            credentials = {};
        }
        
        const response = await window.prsmAPI.registerConnector(platform, credentials);
        
        if (response.message) {
            updatePlatformStatus(platform, 'connected');
            window.prsmAPI.showNotification(
                'Platform Connected',
                `Successfully connected to ${platform}`,
                'info'
            );
        } else {
            throw new Error('Registration failed');
        }
        
    } catch (error) {
        console.error(`Failed to connect to ${platform}:`, error);
        updatePlatformStatus(platform, 'error');
        window.prsmAPI.showNotification(
            'Connection Failed',
            `Failed to connect to ${platform}: ${error.message}`,
            'error'
        );
    }
}

async function disconnectPlatform(platform) {
    console.log(`❌ Disconnecting from ${platform}...`);
    
    // For demo purposes, just update the UI
    updatePlatformStatus(platform, 'not_connected');
    
    window.prsmAPI.showNotification(
        'Platform Disconnected',
        `Disconnected from ${platform}`,
        'info'
    );
}

function updatePlatformStatus(platform, status) {
    const elements = integrationElements[platform];
    if (!elements) return;
    
    platformStates[platform] = { 
        connected: status === 'connected',
        status: status 
    };
    
    const statusElement = elements.status;
    const connectBtn = elements.connectBtn;
    const disconnectBtn = elements.disconnectBtn;
    
    if (statusElement) {
        statusElement.textContent = getStatusText(status);
        statusElement.className = `connection-status ${status}`;
    }
    
    if (connectBtn && disconnectBtn) {
        if (status === 'connected') {
            connectBtn.classList.add('hidden');
            disconnectBtn.classList.remove('hidden');
        } else {
            connectBtn.classList.remove('hidden');
            disconnectBtn.classList.add('hidden');
        }
    }
}

function getStatusText(status) {
    const statusTexts = {
        'connected': 'Connected',
        'not_connected': 'Not Connected',
        'error': 'Connection Error',
        'available': 'Available'
    };
    return statusTexts[status] || status;
}

// === CONTENT SEARCH FUNCTIONS ===

async function searchContent() {
    const searchElements = integrationElements.search;
    if (!searchElements) return;
    
    const query = searchElements.input.value.trim();
    if (!query) {
        window.prsmAPI.showNotification('Search Error', 'Please enter a search query', 'error');
        return;
    }
    
    const contentType = searchElements.typeSelect.value;
    const selectedPlatform = searchElements.platformSelect.value;
    const platforms = selectedPlatform ? [selectedPlatform] : null;
    
    console.log(`🔍 Searching for "${query}" (${contentType})`);
    
    // Show loading state
    searchElements.button.disabled = true;
    searchElements.button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
    
    try {
        const results = await window.prsmAPI.searchContent(query, platforms, contentType, 10);
        displaySearchResults(results);
        
        // Show results container
        searchElements.results.classList.remove('hidden');
        
    } catch (error) {
        console.error('Search failed:', error);
        window.prsmAPI.showNotification('Search Failed', error.message, 'error');
    } finally {
        // Reset button state
        searchElements.button.disabled = false;
        searchElements.button.innerHTML = '<i class="fas fa-search"></i> Search';
    }
}

function displaySearchResults(results) {
    const resultsContainer = integrationElements.search.results;
    if (!resultsContainer) return;
    
    if (!results || results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <p>No results found. Try adjusting your search terms or filters.</p>
            </div>
        `;
        return;
    }
    
    const resultsHTML = results.map(result => `
        <div class="search-result-item">
            <div class="result-info">
                <div class="result-title">${result.display_name}</div>
                <div class="result-description">${result.description || 'No description available'}</div>
                <div class="result-metadata">
                    <span>${result.platform}</span>
                    <span>${result.owner_id}</span>
                    ${result.metadata?.license ? `<span>License: ${result.metadata.license}</span>` : ''}
                    ${result.metadata?.stars ? `<span>⭐ ${result.metadata.stars}</span>` : ''}
                    ${result.metadata?.downloads ? `<span>📥 ${result.metadata.downloads}</span>` : ''}
                </div>
            </div>
            <button class="import-btn" onclick="startImport('${result.platform}', '${result.external_id}', '${result.display_name}')">
                Import
            </button>
        </div>
    `).join('');
    
    resultsContainer.innerHTML = resultsHTML;
}

// === IMPORT FUNCTIONS ===

async function startImport(platform, externalId, displayName) {
    console.log(`📥 Starting import of ${externalId} from ${platform}`);
    
    const importData = {
        source: {
            platform: platform,
            external_id: externalId,
            display_name: displayName,
            owner_id: externalId.split('/')[0] || 'unknown',
            url: `https://${platform === 'github' ? 'github.com' : 'huggingface.co'}/${externalId}`
        },
        import_type: integrationElements.search.typeSelect.value,
        security_scan_required: true,
        license_check_required: true,
        auto_reward_creator: true
    };
    
    try {
        const response = await window.prsmAPI.submitImportRequest(importData);
        
        if (response.request_id) {
            window.prsmAPI.showNotification(
                'Import Started',
                `Import request submitted for ${displayName}`,
                'info'
            );
            
            // Refresh import history
            setTimeout(() => refreshImportHistory(), 1000);
        }
        
    } catch (error) {
        console.error('Import failed:', error);
        window.prsmAPI.showNotification('Import Failed', error.message, 'error');
    }
}

// === IMPORT HISTORY FUNCTIONS ===

async function refreshImportHistory() {
    console.log('🔄 Refreshing import history...');
    
    const importsList = integrationElements.imports?.list;
    const importStats = integrationElements.imports?.stats;
    
    if (!importsList) return;
    
    try {
        const history = await window.prsmAPI.getImportHistory(20, 0);
        
        if (history.length === 0) {
            importsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <p>No imports yet. Start by searching and importing content from connected platforms.</p>
                </div>
            `;
        } else {
            const historyHTML = history.map(item => `
                <div class="import-item">
                    <div class="import-info">
                        <div class="import-title">Import Request</div>
                        <div class="import-details">
                            <span>ID: ${item.request_id.substring(0, 8)}...</span>
                            <span>${new Date(item.created_at).toLocaleDateString()}</span>
                            <span>Duration: ${item.import_duration ? item.import_duration.toFixed(1) + 's' : 'N/A'}</span>
                        </div>
                    </div>
                    <div class="import-status ${item.status}">${item.status.toUpperCase()}</div>
                </div>
            `).join('');
            
            importsList.innerHTML = historyHTML;
        }
        
        if (importStats) {
            importStats.textContent = `${history.length} imports total`;
        }
        
    } catch (error) {
        console.error('Failed to load import history:', error);
        if (importsList) {
            importsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Failed to load import history. Please try again.</p>
                </div>
            `;
        }
    }
}

// === HEALTH MONITORING FUNCTIONS ===

async function checkSystemHealth() {
    console.log('💓 Checking system health...');
    
    const healthElements = integrationElements.health;
    if (!healthElements) return;
    
    try {
        const health = await window.prsmAPI.getIntegrationHealth();
        
        // Update overall health
        if (healthElements.overall) {
            const statusElement = healthElements.overall.querySelector('.health-status');
            const labelElement = healthElements.overall.querySelector('.health-label');
            
            if (statusElement) {
                statusElement.textContent = health.overall_status;
                statusElement.className = `health-status ${health.overall_status.toLowerCase()}`;
            }
        }
        
        // Update connectors health
        if (healthElements.connectors) {
            const statusElement = healthElements.connectors.querySelector('.health-status');
            if (statusElement) {
                statusElement.textContent = `${health.connectors.healthy}/${health.connectors.total}`;
            }
        }
        
        // Update active imports
        if (healthElements.activeImports) {
            const statusElement = healthElements.activeImports.querySelector('.health-status');
            if (statusElement) {
                statusElement.textContent = health.imports.active.toString();
            }
        }
        
    } catch (error) {
        console.error('Health check failed:', error);
        
        // Show unknown status for all indicators
        ['overall', 'connectors', 'activeImports'].forEach(key => {
            const element = healthElements[key];
            if (element) {
                const statusElement = element.querySelector('.health-status');
                if (statusElement) {
                    statusElement.textContent = 'Unknown';
                    statusElement.className = 'health-status unknown';
                }
            }
        });
    }
}

async function updateConnectorHealth() {
    console.log('🔌 Updating connector health...');
    
    try {
        const health = await window.prsmAPI.getConnectorsHealth();
        
        // Update platform statuses based on health data
        Object.keys(health).forEach(platform => {
            const healthData = health[platform];
            if (healthData.status === 'healthy') {
                updatePlatformStatus(platform, 'connected');
            } else {
                updatePlatformStatus(platform, 'error');
            }
        });
        
    } catch (error) {
        console.warn('Failed to update connector health:', error);
    }
}

// --- Simple Tab Navigation ---
function initializeTabNavigation() {
    const tabsContainer = document.getElementById('tabs-container');
    const leftBtn = document.getElementById('scroll-left');
    const rightBtn = document.getElementById('scroll-right');
    
    if (!tabsContainer || !leftBtn || !rightBtn) {
        return;
    }
    
    let scrollPosition = 0;
    const scrollAmount = 200; // pixels to scroll
    
    // Scroll left
    leftBtn.addEventListener('click', () => {
        scrollPosition = Math.min(scrollPosition + scrollAmount, 0);
        tabsContainer.style.transform = `translateX(${scrollPosition}px)`;
    });
    
    // Scroll right
    rightBtn.addEventListener('click', () => {
        const containerWidth = tabsContainer.parentElement.offsetWidth;
        const contentWidth = tabsContainer.scrollWidth;
        const maxScroll = containerWidth - contentWidth;
        
        scrollPosition = Math.max(scrollPosition - scrollAmount, maxScroll);
        tabsContainer.style.transform = `translateX(${scrollPosition}px)`;
    });
    
    // Tab clicking functionality
    const tabs = tabsContainer.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            // Add active class to clicked tab
            tab.classList.add('active');
            
            // Switch content sections
            const targetId = tab.getAttribute('data-target');
            if (targetId) {
                // Hide all content sections
                const allSections = document.querySelectorAll('.content-section');
                allSections.forEach(section => {
                    section.classList.remove('active');
                });
                
                // Show target content section
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.add('active');
                }
            }
        });
    });
}

// Initialize tab navigation
initializeTabNavigation();

// --- Cloud Storage Integration Functions ---
function connectCloudService(serviceName) {
    console.log(`Connecting to ${serviceName}...`);
    
    // In a real implementation, this would:
    // 1. Open OAuth flow for the service
    // 2. Handle authentication
    // 3. Store access tokens
    // 4. Update UI to show connected state
    
    // For demo purposes, simulate connection after delay
    const button = event.target;
    const card = button.closest('.cloud-service-card');
    const statusElement = card.querySelector('.connection-status');
    
    // Show loading state
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
    button.disabled = true;
    
    setTimeout(() => {
        // Update connection status
        statusElement.textContent = 'Connected';
        statusElement.className = 'connection-status connected';
        
        // Update button to disconnect
        button.innerHTML = '<i class="fas fa-unlink"></i> Disconnect';
        button.className = 'disconnect-btn';
        button.onclick = () => disconnectCloudService(serviceName);
        button.disabled = false;
        
        // Show success message
        showNotification(`Successfully connected to ${serviceName}!`, 'success');
        
    }, 2000);
}

function disconnectCloudService(serviceName) {
    console.log(`Disconnecting from ${serviceName}...`);
    
    const button = event.target;
    const card = button.closest('.cloud-service-card');
    const statusElement = card.querySelector('.connection-status');
    
    // Show loading state
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Disconnecting...';
    button.disabled = true;
    
    setTimeout(() => {
        // Update connection status
        statusElement.textContent = 'Not Connected';
        statusElement.className = 'connection-status disconnected';
        
        // Update button to connect
        button.innerHTML = '<i class="fas fa-plug"></i> Connect';
        button.className = 'connect-btn';
        button.onclick = () => connectCloudService(serviceName);
        button.disabled = false;
        
        // Show success message
        showNotification(`Disconnected from ${serviceName}`, 'info');
        
    }, 1500);
}

function uploadFile() {
    console.log('Opening file upload dialog...');
    
    // Create a file input element
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.accept = '*/*';
    
    fileInput.onchange = function(event) {
        const files = event.target.files;
        if (files.length > 0) {
            console.log(`Selected ${files.length} file(s):`, files);
            
            // In a real implementation, this would upload the files
            // For demo, just show a notification
            const fileNames = Array.from(files).map(f => f.name).join(', ');
            showNotification(`Would upload: ${fileNames}`, 'info');
        }
    };
    
    // Trigger file selection
    fileInput.click();
}

function createFolder() {
    console.log('Creating new folder...');
    
    const folderName = prompt('Enter folder name:');
    if (folderName && folderName.trim()) {
        console.log(`Creating folder: ${folderName}`);
        showNotification(`Would create folder: ${folderName}`, 'info');
    }
}

function showNotification(message, type = 'info') {
    // Simple notification system
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '12px 16px',
        backgroundColor: type === 'success' ? '#27ae60' : type === 'error' ? '#e74c3c' : '#3498db',
        color: 'white',
        borderRadius: '6px',
        zIndex: '10000',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        transform: 'translateX(100%)',
        transition: 'transform 0.3s ease'
    });
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Make functions available globally for onclick handlers
window.connectPlatform = connectPlatform;
window.disconnectPlatform = disconnectPlatform;
window.connectCloudService = connectCloudService;
window.disconnectCloudService = disconnectCloudService;
window.uploadFile = uploadFile;
window.createFolder = createFolder;

// --- Task Management Functions ---
function acceptAISuggestion(suggestionType) {
    console.log(`Accepting AI suggestion: ${suggestionType}`);
    
    const suggestionItem = event.target.closest('.suggestion-item');
    const button = event.target;
    
    // Show loading state
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    button.disabled = true;
    
    setTimeout(() => {
        // Update button to show accepted
        button.innerHTML = '<i class="fas fa-check"></i> Accepted';
        button.style.background = '#27ae60';
        button.style.color = 'white';
        button.disabled = true;
        
        // Show notification
        showNotification(`AI suggestion applied: ${suggestionType}`, 'success');
        
        // In a real implementation, this would actually execute the suggestion
        if (suggestionType === 'analysis-template') {
            showNotification('Created structured analysis template for research papers', 'info');
        } else if (suggestionType === 'prioritize-tasks') {
            showNotification('Tasks have been re-prioritized based on deadlines', 'info');
        }
        
    }, 1500);
}

function generateTasks() {
    console.log('Generating smart tasks with AI...');
    
    const button = event.target;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    button.disabled = true;
    
    setTimeout(() => {
        button.innerHTML = '<i class="fas fa-magic"></i> Generate Smart Tasks';
        button.disabled = false;
        
        showNotification('Generated 3 new AI-optimized tasks based on your workflow', 'success');
        
        // In a real implementation, this would add new tasks to the task list
        console.log('Would add new AI-generated tasks to the task list');
        
    }, 2500);
}

function optimizeWorkflow() {
    console.log('Optimizing workflow with AI...');
    
    const button = event.target;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Optimizing...';
    button.disabled = true;
    
    setTimeout(() => {
        button.innerHTML = '<i class="fas fa-route"></i> Optimize Workflow';
        button.disabled = false;
        
        showNotification('Workflow optimized! Task dependencies and timelines updated', 'success');
        
        // In a real implementation, this would reorder and optimize the task workflow
        console.log('Would optimize task workflow and dependencies');
        
    }, 3000);
}

function createTask(taskType) {
    console.log(`Creating new ${taskType} task...`);
    
    const taskTemplates = {
        'research': 'Research Task: Literature review and analysis',
        'analysis': 'Data Analysis: Process and visualize dataset',
        'meeting': 'Team Meeting: Schedule sync with stakeholders',
        'review': 'Document Review: Review and provide feedback'
    };
    
    const taskTitle = taskTemplates[taskType] || 'New Task';
    showNotification(`Created new task: ${taskTitle}`, 'success');
    
    // In a real implementation, this would add the task to the task list
    console.log(`Would create new ${taskType} task with title: ${taskTitle}`);
}

function useTemplate(templateType) {
    console.log(`Using template: ${templateType}`);
    
    const templateInfo = {
        'research-workflow': { name: 'Research Workflow', tasks: 8 },
        'data-analysis': { name: 'Data Analysis Pipeline', tasks: 6 },
        'ml-experiment': { name: 'ML Experiment', tasks: 10 }
    };
    
    const template = templateInfo[templateType];
    if (template) {
        showNotification(`Created ${template.tasks} tasks from ${template.name} template`, 'success');
        
        // In a real implementation, this would create all tasks from the template
        console.log(`Would create ${template.tasks} tasks from ${template.name} template`);
    }
}

function showTaskCreator() {
    console.log('Opening task creator...');
    
    // Simple task creator simulation
    const taskTitle = prompt('Enter task title:');
    if (taskTitle && taskTitle.trim()) {
        showNotification(`Created new task: ${taskTitle}`, 'success');
        console.log(`Would create new task: ${taskTitle}`);
    }
}

// Task category filtering
function initializeTaskCategories() {
    const categoryTabs = document.querySelectorAll('.category-tab');
    
    categoryTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            categoryTabs.forEach(t => t.classList.remove('active'));
            // Add active class to clicked tab
            tab.classList.add('active');
            
            const category = tab.getAttribute('data-category');
            console.log(`Filtering tasks by category: ${category}`);
            
            // In a real implementation, this would filter the task list
            showNotification(`Filtered tasks by: ${category}`, 'info');
        });
    });
}

// Task completion handling
function initializeTaskCheckboxes() {
    const taskCheckboxes = document.querySelectorAll('.task-checkbox input[type="checkbox"]');
    
    taskCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', (event) => {
            const taskItem = event.target.closest('.task-item');
            const taskTitle = taskItem.querySelector('.task-title').textContent;
            
            if (event.target.checked) {
                taskItem.classList.add('completed');
                showNotification(`Task completed: ${taskTitle}`, 'success');
            } else {
                taskItem.classList.remove('completed');
                showNotification(`Task reopened: ${taskTitle}`, 'info');
            }
        });
    });
}

// Task filtering and sorting
function initializeTaskControls() {
    const filterSelect = document.querySelector('.task-filter-select');
    const sortSelect = document.querySelector('.task-sort-select');
    
    if (filterSelect) {
        filterSelect.addEventListener('change', (event) => {
            const filterValue = event.target.value;
            console.log(`Filtering tasks by: ${filterValue}`);
            showNotification(`Filtered tasks: ${filterValue}`, 'info');
        });
    }
    
    if (sortSelect) {
        sortSelect.addEventListener('change', (event) => {
            const sortValue = event.target.value;
            console.log(`Sorting tasks by: ${sortValue}`);
            showNotification(`Tasks sorted by: ${sortValue}`, 'info');
        });
    }
}

// Initialize task management when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Add delay to ensure task elements are loaded
    setTimeout(() => {
        initializeTaskCategories();
        initializeTaskCheckboxes();
        initializeTaskControls();
    }, 500);
});

// ============================================
// ANALYTICS DASHBOARD FUNCTIONALITY
// ============================================

// Analytics chart timeframe handling
function initializeAnalyticsCharts() {
    const timeframeSelects = document.querySelectorAll('.chart-timeframe');
    
    timeframeSelects.forEach(select => {
        select.addEventListener('change', (event) => {
            const timeframe = event.target.value;
            updateChartData(timeframe);
            showNotification(`Chart updated to show ${timeframe}`, 'info');
        });
    });
}

// Update chart data based on timeframe
function updateChartData(timeframe) {
    const chartBars = document.querySelectorAll('.chart-bar');
    
    // Mock data for different timeframes
    const mockData = {
        '24h': [60, 80, 45, 90, 75, 65, 85],
        '7d': [70, 85, 60, 95, 80, 70, 90],
        '30d': [55, 75, 40, 85, 70, 60, 80],
        '90d': [65, 90, 55, 100, 85, 75, 95]
    };
    
    const data = mockData[timeframe] || mockData['7d'];
    
    chartBars.forEach((bar, index) => {
        if (data[index]) {
            bar.style.height = `${data[index]}%`;
        }
    });
}

// Cost optimization insights
function applyOptimization(type) {
    let message = '';
    
    switch(type) {
        case 'model-switch':
            message = 'AI model optimization settings have been updated. Estimated savings: ₦ 380/month';
            break;
        case 'batch-requests':
            message = 'Request batching configuration enabled. Review your API calls for optimization opportunities.';
            break;
        default:
            message = 'Optimization applied successfully.';
    }
    
    showNotification(message, 'success');
}

// Export functionality
function exportAnalyticsData(format) {
    let message = '';
    
    switch(format) {
        case 'pdf':
            message = 'Weekly usage report PDF is being generated...';
            break;
        case 'csv':
            message = 'Raw analytics data is being exported to CSV...';
            break;
        case 'insights':
            message = 'Cost optimization report is being prepared...';
            break;
        default:
            message = 'Export started...';
    }
    
    showNotification(message, 'info');
    
    // Simulate export processing
    setTimeout(() => {
        showNotification('Export completed successfully!', 'success');
    }, 2000);
}

// Scheduled reports management
function scheduleNewReport() {
    showNotification('Report scheduling interface would open here', 'info');
}

function editScheduledReport(reportId) {
    showNotification(`Editing report: ${reportId}`, 'info');
}

function disableScheduledReport(reportId) {
    showNotification(`Report disabled: ${reportId}`, 'info');
}

// Analytics dashboard initialization
function initializeAnalyticsDashboard() {
    initializeAnalyticsCharts();
    
    // Add click handlers for optimization insights
    const insightButtons = document.querySelectorAll('.insight-action-btn');
    insightButtons.forEach(button => {
        button.addEventListener('click', (event) => {
            const insightItem = event.target.closest('.insight-item');
            const insightText = insightItem.querySelector('strong').textContent;
            
            if (insightText.includes('smaller models')) {
                applyOptimization('model-switch');
            } else if (insightText.includes('Batch')) {
                applyOptimization('batch-requests');
            } else {
                applyOptimization('default');
            }
        });
    });
    
    // Add click handlers for export buttons
    const exportButtons = document.querySelectorAll('.export-btn');
    exportButtons.forEach(button => {
        button.addEventListener('click', (event) => {
            const exportCard = event.target.closest('.export-card');
            const exportTitle = exportCard.querySelector('h6').textContent;
            
            if (exportTitle.includes('PDF')) {
                exportAnalyticsData('pdf');
            } else if (exportTitle.includes('CSV')) {
                exportAnalyticsData('csv');
            } else if (exportTitle.includes('Optimization')) {
                exportAnalyticsData('insights');
            }
        });
    });
    
    // Add click handlers for scheduled reports
    const addReportBtn = document.querySelector('.add-report-btn');
    if (addReportBtn) {
        addReportBtn.addEventListener('click', scheduleNewReport);
    }
    
    const reportActionBtns = document.querySelectorAll('.report-action-btn');
    reportActionBtns.forEach(button => {
        button.addEventListener('click', (event) => {
            const action = event.target.textContent.toLowerCase();
            const reportItem = event.target.closest('.scheduled-report-item');
            const reportName = reportItem.querySelector('.report-name').textContent;
            
            if (action === 'edit') {
                editScheduledReport(reportName);
            } else if (action === 'disable') {
                disableScheduledReport(reportName);
            }
        });
    });
}

// Initialize analytics when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Add delay to ensure analytics elements are loaded
    setTimeout(() => {
        initializeAnalyticsDashboard();
    }, 600);
});

// Make task functions available globally
window.acceptAISuggestion = acceptAISuggestion;
window.generateTasks = generateTasks;
window.optimizeWorkflow = optimizeWorkflow;
window.createTask = createTask;
window.useTemplate = useTemplate;
window.showTaskCreator = showTaskCreator;
window.searchContent = searchContent;
window.startImport = startImport;
window.refreshImportHistory = refreshImportHistory;
window.checkSystemHealth = checkSystemHealth;

// Make analytics functions available globally
window.exportAnalyticsData = exportAnalyticsData;
window.applyOptimization = applyOptimization;
window.scheduleNewReport = scheduleNewReport;
window.updateChartData = updateChartData;

// ============================================
// INFORMATION SPACE FUNCTIONALITY
// ============================================

// Research subject presets
const researchPresets = {
    'apm': {
        name: 'Atomically Precise Manufacturing',
        nodes: [
            { id: 'apm-core', label: 'Atomically Precise Manufacturing', type: 'central-node', icon: 'fas fa-atom', x: 50, y: 50 },
            { id: 'molecular-assembly', label: 'Molecular Assembly', type: 'high-opportunity', icon: 'fas fa-puzzle-piece', x: 25, y: 30 },
            { id: 'scanning-probe', label: 'Scanning Probe Techniques', type: 'medium-opportunity', icon: 'fas fa-microscope', x: 75, y: 25 },
            { id: 'quantum-control', label: 'Quantum Control Systems', type: 'high-complexity', icon: 'fas fa-atom', x: 20, y: 70 },
            { id: 'ai-materials', label: 'AI-Driven Materials Discovery', type: 'emerging-field', icon: 'fas fa-brain', x: 80, y: 75 },
            { id: 'surface-chemistry', label: 'Surface Chemistry', type: 'established-field', icon: 'fas fa-layer-group', x: 45, y: 15 },
            { id: 'computational-modeling', label: 'Computational Modeling', type: 'medium-opportunity', icon: 'fas fa-calculator', x: 15, y: 50 }
        ]
    },
    'quantum': {
        name: 'Quantum Computing',
        nodes: [
            { id: 'quantum-core', label: 'Quantum Computing', type: 'central-node', icon: 'fas fa-microchip', x: 50, y: 50 },
            { id: 'quantum-algorithms', label: 'Quantum Algorithms', type: 'high-opportunity', icon: 'fas fa-code', x: 30, y: 25 },
            { id: 'quantum-hardware', label: 'Quantum Hardware', type: 'high-complexity', icon: 'fas fa-server', x: 70, y: 30 },
            { id: 'quantum-error-correction', label: 'Quantum Error Correction', type: 'high-complexity', icon: 'fas fa-shield-alt', x: 25, y: 75 },
            { id: 'quantum-networking', label: 'Quantum Networking', type: 'emerging-field', icon: 'fas fa-network-wired', x: 75, y: 70 },
            { id: 'quantum-software', label: 'Quantum Software', type: 'medium-opportunity', icon: 'fas fa-laptop-code', x: 15, y: 45 }
        ]
    },
    'fusion': {
        name: 'Nuclear Fusion',
        nodes: [
            { id: 'fusion-core', label: 'Nuclear Fusion', type: 'central-node', icon: 'fas fa-sun', x: 50, y: 50 },
            { id: 'plasma-physics', label: 'Plasma Physics', type: 'established-field', icon: 'fas fa-fire', x: 30, y: 30 },
            { id: 'magnetic-confinement', label: 'Magnetic Confinement', type: 'high-complexity', icon: 'fas fa-magnet', x: 70, y: 25 },
            { id: 'inertial-confinement', label: 'Inertial Confinement', type: 'high-opportunity', icon: 'fas fa-compress-arrows-alt', x: 25, y: 70 },
            { id: 'tritium-breeding', label: 'Tritium Breeding', type: 'medium-opportunity', icon: 'fas fa-atom', x: 75, y: 75 },
            { id: 'fusion-materials', label: 'Fusion Materials', type: 'high-complexity', icon: 'fas fa-industry', x: 15, y: 50 }
        ]
    },
    'longevity': {
        name: 'Longevity Research',
        nodes: [
            { id: 'longevity-core', label: 'Longevity Research', type: 'central-node', icon: 'fas fa-dna', x: 50, y: 50 },
            { id: 'cellular-reprogramming', label: 'Cellular Reprogramming', type: 'high-opportunity', icon: 'fas fa-redo', x: 25, y: 25 },
            { id: 'senescence', label: 'Cellular Senescence', type: 'established-field', icon: 'fas fa-clock', x: 75, y: 30 },
            { id: 'stem-cells', label: 'Stem Cell Therapy', type: 'medium-opportunity', icon: 'fas fa-seedling', x: 20, y: 75 },
            { id: 'genetic-engineering', label: 'Genetic Engineering', type: 'emerging-field', icon: 'fas fa-cut', x: 80, y: 70 },
            { id: 'biomarkers', label: 'Aging Biomarkers', type: 'medium-opportunity', icon: 'fas fa-chart-line', x: 15, y: 45 }
        ]
    }
};

// Load subject preset
function loadSubjectPreset(presetId) {
    const preset = researchPresets[presetId];
    if (!preset) return;
    
    const subjectInput = document.getElementById('research-subject-input');
    if (subjectInput) {
        subjectInput.value = preset.name;
    }
    
    updateVisualization(preset);
    showNotification(`Loaded preset: ${preset.name}`, 'info');
}

// Update visualization with new data
function updateVisualization(preset) {
    const canvas = document.getElementById('info-space-canvas');
    if (!canvas) return;
    
    // Clear existing nodes
    const existingNodes = canvas.querySelectorAll('.research-node');
    existingNodes.forEach(node => node.remove());
    
    // Add new nodes
    preset.nodes.forEach(nodeData => {
        createResearchNode(nodeData, canvas);
    });
    
    // Update connections
    updateConnectionLines(preset.nodes);
}

// Create research node element
function createResearchNode(nodeData, container) {
    const node = document.createElement('div');
    node.className = `research-node ${nodeData.type}`;
    node.setAttribute('data-domain', nodeData.id);
    node.style.left = `${nodeData.x}%`;
    node.style.top = `${nodeData.y}%`;
    
    node.innerHTML = `
        <div class="node-content">
            <div class="node-icon">
                <i class="${nodeData.icon}"></i>
            </div>
            <div class="node-label">${nodeData.label}</div>
            <div class="node-complexity">${getComplexityLabel(nodeData.type)}</div>
        </div>
    `;
    
    node.addEventListener('click', () => {
        showNodeDetails(nodeData);
    });
    
    container.querySelector('.space-network').appendChild(node);
}

// Get complexity label for node type
function getComplexityLabel(type) {
    const labels = {
        'central-node': 'Core Subject',
        'high-opportunity': 'High Opportunity',
        'medium-opportunity': 'Medium Opportunity',
        'high-complexity': 'High Complexity',
        'emerging-field': 'Emerging Field',
        'established-field': 'Established Field'
    };
    return labels[type] || 'Unknown';
}

// Update connection lines between nodes
function updateConnectionLines(nodes) {
    const svg = document.querySelector('.connection-lines');
    if (!svg) return;
    
    // Clear existing lines
    svg.innerHTML = '';
    
    // Create connections (simplified - in real app would be data-driven)
    const connections = [
        { from: 0, to: 1, strength: 'strong' },
        { from: 0, to: 2, strength: 'medium' },
        { from: 0, to: 3, strength: 'weak' },
        { from: 0, to: 4, strength: 'medium' },
        { from: 0, to: 5, strength: 'strong' },
        { from: 0, to: 6, strength: 'medium' },
        { from: 1, to: 5, strength: 'weak' },
        { from: 2, to: 5, strength: 'weak' },
        { from: 4, to: 6, strength: 'medium' }
    ];
    
    connections.forEach(conn => {
        if (nodes[conn.from] && nodes[conn.to]) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('class', `connection-line ${conn.strength}`);
            line.setAttribute('x1', `${nodes[conn.from].x}%`);
            line.setAttribute('y1', `${nodes[conn.from].y}%`);
            line.setAttribute('x2', `${nodes[conn.to].x}%`);
            line.setAttribute('y2', `${nodes[conn.to].y}%`);
            svg.appendChild(line);
        }
    });
}

// Show node details
function showNodeDetails(nodeData) {
    showNotification(`Exploring: ${nodeData.label}`, 'info');
    // In a real implementation, this would open a detailed view or sidebar
}

// Analyze subject with NWTN
function analyzeSubjectWithNWTN() {
    const subjectInput = document.getElementById('research-subject-input');
    const subject = subjectInput ? subjectInput.value : '';
    
    if (!subject.trim()) {
        showNotification('Please enter a research subject to analyze', 'warning');
        return;
    }
    
    showNotification(`Analyzing "${subject}" with NWTN AI...`, 'info');
    
    // Simulate analysis processing
    setTimeout(() => {
        showNotification('NWTN analysis complete! Information space updated.', 'success');
        // In real implementation, would update visualization with AI-generated insights
    }, 3000);
}

// View mode change handler
function handleViewModeChange(mode) {
    const canvas = document.querySelector('.information-space-canvas');
    if (!canvas) return;
    
    // Remove existing mode classes
    canvas.classList.remove('opportunity-view', 'complexity-view', 'connections-view', 'timeline-view');
    
    // Add new mode class
    canvas.classList.add(`${mode}-view`);
    
    showNotification(`Switched to ${mode} view`, 'info');
}

// Focus area change handler
function handleFocusAreaChange(area) {
    const nodes = document.querySelectorAll('.research-node');
    
    nodes.forEach(node => {
        if (area === 'all') {
            node.style.opacity = '1';
        } else {
            // In real implementation, would filter based on domain categories
            const shouldShow = Math.random() > 0.3; // Mock filtering
            node.style.opacity = shouldShow ? '1' : '0.3';
        }
    });
    
    showNotification(`Focused on: ${area}`, 'info');
}

// Explore opportunity
function exploreOpportunity(opportunityId) {
    const opportunities = {
        'molecular-assembly': 'Molecular Assembly Automation research path would involve: 1) Advanced AFM positioning systems, 2) Machine learning for molecular recognition, 3) Automated feedback control systems',
        'quantum-control': 'Quantum-Enhanced Precision Control path: 1) Quantum sensing integration, 2) Coherent control protocols, 3) Error correction for manufacturing',
        'ai-molecular-tools': 'AI-Designed Molecular Tools path: 1) Generative models for molecular design, 2) Physics-informed neural networks, 3) High-throughput virtual screening'
    };
    
    const message = opportunities[opportunityId] || 'Research path details would be generated by NWTN AI';
    showNotification(message, 'info');
}

// Generate research path
function generateResearchPath() {
    const fromSelect = document.getElementById('path-from-select');
    const toSelect = document.getElementById('path-to-select');
    
    if (!fromSelect || !toSelect) return;
    
    const from = fromSelect.value;
    const to = toSelect.value;
    
    showNotification(`Generating research path from "${from}" to "${to}"...`, 'info');
    
    // Simulate path generation
    setTimeout(() => {
        showNotification('Research path generated successfully!', 'success');
        // In real implementation, would update the timeline with AI-generated path
    }, 2000);
}

// Recenter view
function recenterView() {
    const canvas = document.querySelector('.information-space-canvas');
    if (canvas) {
        canvas.scrollTo({
            left: canvas.scrollWidth / 2 - canvas.clientWidth / 2,
            top: canvas.scrollHeight / 2 - canvas.clientHeight / 2,
            behavior: 'smooth'
        });
    }
    showNotification('View recentered', 'info');
}

// Toggle fullscreen
function toggleFullscreen() {
    const canvas = document.querySelector('.information-space-canvas');
    if (!canvas) return;
    
    if (!document.fullscreenElement) {
        canvas.requestFullscreen().then(() => {
            showNotification('Entered fullscreen mode', 'info');
        }).catch(() => {
            showNotification('Fullscreen not supported', 'warning');
        });
    } else {
        document.exitFullscreen().then(() => {
            showNotification('Exited fullscreen mode', 'info');
        });
    }
}

// Initialize Information Space
function initializeInformationSpace() {
    // Load default preset
    loadSubjectPreset('apm');
    
    // Set up event listeners
    const analyzeBtn = document.getElementById('analyze-subject-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeSubjectWithNWTN);
    }
    
    const viewModeSelect = document.getElementById('view-mode-select');
    if (viewModeSelect) {
        viewModeSelect.addEventListener('change', (e) => {
            handleViewModeChange(e.target.value);
        });
    }
    
    const focusAreaSelect = document.getElementById('focus-area-select');
    if (focusAreaSelect) {
        focusAreaSelect.addEventListener('change', (e) => {
            handleFocusAreaChange(e.target.value);
        });
    }
    
    const recenterBtn = document.getElementById('recenter-view-btn');
    if (recenterBtn) {
        recenterBtn.addEventListener('click', recenterView);
    }
    
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', toggleFullscreen);
    }
    
    const generatePathBtn = document.getElementById('generate-path-btn');
    if (generatePathBtn) {
        generatePathBtn.addEventListener('click', generateResearchPath);
    }
}

// Initialize Information Space when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        initializeInformationSpace();
    }, 700);
});

// Make Information Space functions available globally
window.loadSubjectPreset = loadSubjectPreset;
window.exploreOpportunity = exploreOpportunity;
window.generateResearchPath = generateResearchPath;
window.analyzeSubjectWithNWTN = analyzeSubjectWithNWTN;

// ============================================
// COLLABORATION FUNCTIONALITY
// ============================================

// Collaboration Tab Management
function switchCollabTab(tabName) {
    // Remove active class from all tabs and content
    document.querySelectorAll('.collab-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.collab-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Add active class to selected tab and content
    const selectedTab = document.querySelector(`[data-tab="${tabName}"]`);
    const selectedContent = document.getElementById(`${tabName}-tab-content`);
    
    if (selectedTab && selectedContent) {
        selectedTab.classList.add('active');
        selectedContent.classList.add('active');
    }
}

// Team Switching
function switchTeam(teamId) {
    console.log(`Switching to team: ${teamId}`);
    
    // Update channels based on team
    const channelsData = {
        'quantum-research': [
            { name: 'general', icon: 'hashtag', unread: 3 },
            { name: 'quantum-theory', icon: 'atom', unread: 0 },
            { name: 'experiments', icon: 'flask', unread: 1 },
            { name: 'publications', icon: 'file-alt', unread: 0 }
        ],
        'ai-ethics': [
            { name: 'general', icon: 'hashtag', unread: 2 },
            { name: 'policy-review', icon: 'gavel', unread: 0 },
            { name: 'community-feedback', icon: 'users', unread: 3 },
            { name: 'documentation', icon: 'file-alt', unread: 1 }
        ],
        'apm-project': [
            { name: 'general', icon: 'hashtag', unread: 3 },
            { name: 'research', icon: 'flask', unread: 0 },
            { name: 'publications', icon: 'file-alt', unread: 1 },
            { name: 'code-review', icon: 'code', unread: 0 }
        ]
    };
    
    const channelsList = document.querySelector('.channels-list');
    if (channelsList && channelsData[teamId]) {
        channelsList.innerHTML = channelsData[teamId].map(channel => `
            <div class="channel-item ${channel.name === 'general' ? 'active' : ''}" data-channel="${channel.name}">
                <i class="fas fa-${channel.icon}"></i>
                <span>${channel.name}</span>
                ${channel.unread > 0 ? `<span class="unread-count">${channel.unread}</span>` : ''}
            </div>
        `).join('');
        
        // Add click handlers to new channels
        channelsList.querySelectorAll('.channel-item').forEach(item => {
            item.addEventListener('click', () => switchChannel(item.dataset.channel));
        });
    }
}

// Channel Switching
function switchChannel(channelName) {
    // Update active channel
    document.querySelectorAll('.channel-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const selectedChannel = document.querySelector(`[data-channel="${channelName}"]`);
    if (selectedChannel) {
        selectedChannel.classList.add('active');
        
        // Remove unread count
        const unreadCount = selectedChannel.querySelector('.unread-count');
        if (unreadCount) {
            unreadCount.remove();
        }
    }
    
    // Update chat header
    const channelInfo = document.querySelector('.channel-info h5');
    if (channelInfo) {
        channelInfo.innerHTML = `<i class="fas fa-hashtag"></i> ${channelName}`;
    }
    
    // Load channel messages (mock)
    loadChannelMessages(channelName);
}

// Load Channel Messages
function loadChannelMessages(channelName) {
    const messagesData = {
        'general': [
            {
                user: 'Dr. Sarah Chen',
                avatar: 'SC',
                time: 'Today at 2:30 PM',
                content: 'Just finished reviewing the latest APM simulation results. The precision improvements are impressive! 📊',
                reactions: ['👍 3', '🚀 2']
            },
            {
                user: 'Alex Rodriguez',
                avatar: 'AR',
                time: 'Today at 2:35 PM',
                content: 'Agreed! Should we schedule a team meeting to discuss the next phase? I\'ve been working on the quantum control algorithms.'
            }
        ],
        'research': [
            {
                user: 'Dr. Maria Thompson',
                avatar: 'MT',
                time: 'Today at 1:15 PM',
                content: 'I\'ve uploaded the latest ethics review documents to the shared folder. Please review by Friday.',
                reactions: ['✅ 4']
            }
        ],
        'publications': [
            {
                user: 'Dr. Sarah Chen',
                avatar: 'SC',
                time: 'Today at 11:00 AM',
                content: 'The peer review feedback is mostly positive. We need to address the methodology section concerns.',
                reactions: ['📝 2']
            }
        ]
    };
    
    const chatMessages = document.querySelector('.chat-messages');
    const messages = messagesData[channelName] || messagesData['general'];
    
    if (chatMessages) {
        chatMessages.innerHTML = messages.map(msg => `
            <div class="message-group">
                <div class="message">
                    <div class="message-header">
                        <div class="user-avatar">
                            <img src="data:image/svg+xml;base64,${generateAvatarSVG(msg.avatar)}" alt="${msg.avatar}">
                        </div>
                        <span class="username">${msg.user}</span>
                        <span class="timestamp">${msg.time}</span>
                    </div>
                    <div class="message-content">${msg.content}</div>
                    ${msg.reactions ? `<div class="message-reactions">
                        ${msg.reactions.map(reaction => `<span class="reaction">${reaction}</span>`).join('')}
                    </div>` : ''}
                </div>
            </div>
        `).join('');
    }
}

// Generate Avatar SVG
function generateAvatarSVG(initials) {
    const colors = ['#4F76DF', '#FF787C', '#9CA3AF', '#6B7380'];
    const color = colors[initials.charCodeAt(0) % colors.length];
    
    const svg = `<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="16" cy="16" r="16" fill="${color}"/>
        <text x="16" y="21" text-anchor="middle" fill="white" font-family="Inter" font-size="12" font-weight="600">${initials}</text>
    </svg>`;
    
    return btoa(svg);
}

// Send Message
function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();
    
    if (message) {
        // Add message to chat
        const chatMessages = document.querySelector('.chat-messages');
        const newMessage = document.createElement('div');
        newMessage.className = 'message-group';
        newMessage.innerHTML = `
            <div class="message">
                <div class="message-header">
                    <div class="user-avatar">
                        <img src="data:image/svg+xml;base64,${generateAvatarSVG('YOU')}" alt="YOU">
                    </div>
                    <span class="username">You</span>
                    <span class="timestamp">Just now</span>
                </div>
                <div class="message-content">${message}</div>
            </div>
        `;
        
        chatMessages.appendChild(newMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Clear input
        chatInput.value = '';
    }
}

// Channel Management
function createChannel() {
    const channelName = prompt('Enter channel name:');
    if (channelName) {
        console.log(`Creating channel: ${channelName}`);
        // In a real app, this would create the channel via API
    }
}

// Task Management
function createTask() {
    console.log('Creating new task...');
    // In a real app, this would open a task creation modal
    alert('Task creation feature would open here in a real application.');
}

// Kanban Management
function addKanbanColumn() {
    const columnName = prompt('Enter column name:');
    if (columnName) {
        console.log(`Adding Kanban column: ${columnName}`);
        // In a real app, this would add the column to the board
    }
}

function kanbanSettings() {
    console.log('Opening Kanban settings...');
    // In a real app, this would open settings modal
    alert('Kanban settings would open here in a real application.');
}

// File Management
function uploadFile() {
    console.log('Uploading file...');
    // In a real app, this would open file upload dialog
    alert('File upload dialog would open here in a real application.');
}

function createFolder() {
    const folderName = prompt('Enter folder name:');
    if (folderName) {
        console.log(`Creating folder: ${folderName}`);
        // In a real app, this would create the folder
    }
}

// Team Management
function inviteTeamMember() {
    const email = prompt('Enter team member email:');
    if (email) {
        console.log(`Inviting team member: ${email}`);
        // In a real app, this would send an invitation
        alert(`Invitation would be sent to ${email} in a real application.`);
    }
}

function createProject() {
    const projectName = prompt('Enter project name:');
    if (projectName) {
        console.log(`Creating project: ${projectName}`);
        // In a real app, this would create the project
        alert(`Project "${projectName}" would be created in a real application.`);
    }
}

// Quick Actions
function scheduleTeamMeeting() {
    console.log('Scheduling team meeting...');
    alert('Meeting scheduling interface would open here in a real application.');
}

function shareScreen() {
    console.log('Starting screen share...');
    alert('Screen sharing would start here in a real application.');
}

function createPoll() {
    console.log('Creating poll...');
    alert('Poll creation interface would open here in a real application.');
}

function sendAnnouncement() {
    const announcement = prompt('Enter announcement message:');
    if (announcement) {
        console.log(`Sending announcement: ${announcement}`);
        alert(`Announcement would be sent to all team members in a real application.`);
    }
}

// Initialize Collaboration Features
function initializeCollaborationFeatures() {
    // Add enter key handler for chat input
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Add click handlers for channel items
    document.querySelectorAll('.channel-item').forEach(item => {
        item.addEventListener('click', () => {
            switchChannel(item.dataset.channel);
        });
    });
    
    // Add click handlers for collaboration tabs
    document.querySelectorAll('.collab-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchCollabTab(tab.dataset.tab);
        });
    });
    
    // Initialize with general channel messages
    loadChannelMessages('general');
    
    // Set up task checkbox handlers
    document.querySelectorAll('.task-checkbox input').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const taskItem = e.target.closest('.task-item');
            if (taskItem) {
                taskItem.classList.toggle('completed', e.target.checked);
            }
        });
    });
    
    // Set up file view toggle
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const view = btn.dataset.view;
            const filesContainer = document.querySelector('.files-grid, .files-list');
            if (filesContainer) {
                // Remove both classes first
                filesContainer.classList.remove('files-grid', 'files-list');
                // Add the appropriate class based on view
                filesContainer.classList.add(view === 'list' ? 'files-list' : 'files-grid');
            }
        });
    });
    
    console.log('Collaboration features initialized');
}

// Make collaboration functions available globally
window.switchCollabTab = switchCollabTab;
window.switchTeam = switchTeam;
window.switchChannel = switchChannel;
window.sendMessage = sendMessage;
window.createChannel = createChannel;
window.createTask = createTask;
window.addKanbanColumn = addKanbanColumn;
window.kanbanSettings = kanbanSettings;
window.uploadFile = uploadFile;
window.createFolder = createFolder;
window.inviteTeamMember = inviteTeamMember;
window.createProject = createProject;
window.scheduleTeamMeeting = scheduleTeamMeeting;
window.shareScreen = shareScreen;
window.createPoll = createPoll;
window.sendAnnouncement = sendAnnouncement;

// Initialize collaboration features when collaboration content is ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        initializeCollaborationFeatures();
        initializeGovernanceFeatures();
    }, 800);
});

// ===============================================
// GOVERNANCE FUNCTIONALITY
// ===============================================

function initializeGovernanceFeatures() {
    // Initialize governance navigation
    initializeGovernanceNavigation();
    
    // Initialize voting features
    initializeVotingSystem();
    
    // Initialize treasury management
    initializeTreasuryFeatures();
    
    // Initialize proposal creation
    initializeProposalCreation();
    
    // Initialize analytics interactions
    initializeGovernanceAnalytics();
    
    // Initialize action center
    initializeActionCenter();
    
    console.log('Governance features initialized successfully');
}

function initializeGovernanceNavigation() {
    const navButtons = document.querySelectorAll('.governance-nav-btn');
    const tabContents = document.querySelectorAll('.governance-tab-content');
    
    navButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all buttons and tabs
            navButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Show corresponding tab content
            const targetTab = button.getAttribute('data-tab');
            const targetContent = document.getElementById(targetTab);
            
            if (targetContent) {
                targetContent.classList.add('active');
            }
            
            // Update analytics based on tab switch
            updateGovernanceAnalytics(targetTab);
        });
    });
}

function initializeVotingSystem() {
    // Initialize proposal voting
    initializeProposalVoting();
    
    // Initialize quadratic voting
    initializeQuadraticVoting();
    
    // Initialize voting power display
    updateVotingPowerDisplay();
}

function initializeProposalVoting() {
    const voteButtons = document.querySelectorAll('.vote-btn');
    
    voteButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            
            const proposalCard = button.closest('.proposal-card');
            const proposalId = proposalCard.getAttribute('data-proposal-id');
            const voteType = button.classList.contains('yes') ? 'yes' : 'no';
            
            // Show voting confirmation
            showVotingConfirmation(proposalId, voteType, button);
        });
    });
}

function showVotingConfirmation(proposalId, voteType, button) {
    const votingPower = getCurrentVotingPower();
    const confirmation = confirm(
        `Confirm your ${voteType.toUpperCase()} vote?\n\n` +
        `Proposal ID: ${proposalId}\n` +
        `Your Voting Power: ${votingPower} FTNS\n` +
        `This action cannot be undone.`
    );
    
    if (confirmation) {
        castVote(proposalId, voteType, votingPower, button);
    }
}

function castVote(proposalId, voteType, votingPower, button) {
    // Simulate API call
    const proposalCard = button.closest('.proposal-card');
    const votingProgress = proposalCard.querySelector('.voting-progress');
    
    // Update button states
    const allVoteButtons = proposalCard.querySelectorAll('.vote-btn');
    allVoteButtons.forEach(btn => {
        btn.disabled = true;
        btn.textContent = btn === button ? 'Voted!' : 'Disabled';
    });
    
    // Update voting bars (simulate vote counting)
    updateVotingBars(proposalCard, voteType, votingPower);
    
    // Show success message
    showNotification(`Vote cast successfully! Your ${voteType.toUpperCase()} vote has been recorded.`, 'success');
    
    // Update governance analytics
    updateVoteParticipationMetrics();
}

function updateVotingBars(proposalCard, voteType, votingPower) {
    const votingBar = proposalCard.querySelector('.voting-bar');
    const yesBar = votingBar.querySelector('.vote-yes');
    const noBar = votingBar.querySelector('.vote-no');
    const votingStats = proposalCard.querySelector('.voting-stats');
    
    // Get current percentages (simulated)
    let yesPercentage = parseFloat(yesBar.style.width) || 45;
    let noPercentage = parseFloat(noBar.style.width) || 35;
    
    // Add vote influence based on voting power (simplified)
    const influence = Math.min(votingPower / 1000 * 5, 10); // Max 10% influence
    
    if (voteType === 'yes') {
        yesPercentage += influence;
    } else {
        noPercentage += influence;
    }
    
    // Normalize to ensure total doesn't exceed 100%
    const total = yesPercentage + noPercentage;
    if (total > 80) {
        const factor = 80 / total;
        yesPercentage *= factor;
        noPercentage *= factor;
    }
    
    // Update visual bars
    yesBar.style.width = `${yesPercentage}%`;
    noBar.style.width = `${noPercentage}%`;
    
    // Update stats text
    if (votingStats) {
        votingStats.innerHTML = `
            <span>Yes: ${yesPercentage.toFixed(1)}%</span>
            <span>No: ${noPercentage.toFixed(1)}%</span>
        `;
    }
}

function initializeQuadraticVoting() {
    const allocationSliders = document.querySelectorAll('.allocation-slider input[type="range"]');
    
    allocationSliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            updateQuadraticVotingDisplay(slider);
        });
        
        // Initialize display
        updateQuadraticVotingDisplay(slider);
    });
}

function updateQuadraticVotingDisplay(slider) {
    const allocationDisplay = slider.closest('.quadratic-voting').querySelector('.allocation-display');
    const tokensAllocated = parseInt(slider.value);
    
    // Calculate quadratic voting power
    const votingPower = Math.sqrt(tokensAllocated);
    
    // Update display
    const tokensSpan = allocationDisplay.querySelector('.tokens-allocated');
    const powerSpan = allocationDisplay.querySelector('.voting-power');
    
    if (tokensSpan) tokensSpan.textContent = `${tokensAllocated} FTNS`;
    if (powerSpan) powerSpan.textContent = `${votingPower.toFixed(2)} Voting Power`;
    
    // Update slider background color based on allocation
    const percentage = (tokensAllocated / slider.max) * 100;
    slider.style.background = `linear-gradient(to right, var(--primary-color) 0%, var(--primary-color) ${percentage}%, var(--border-color) ${percentage}%, var(--border-color) 100%)`;
}

function getCurrentVotingPower() {
    // Simulate getting user's current voting power
    const userStake = 2500; // Example stake
    return Math.sqrt(userStake).toFixed(2);
}

function updateVotingPowerDisplay() {
    const votingPowerElements = document.querySelectorAll('.voting-power .power-amount');
    const currentPower = getCurrentVotingPower();
    
    votingPowerElements.forEach(element => {
        element.textContent = currentPower;
    });
}

function initializeTreasuryFeatures() {
    // Initialize fund allocation visualization
    initializeFundAllocation();
    
    // Initialize funding proposal interactions
    initializeFundingProposals();
    
    // Update treasury displays
    updateTreasuryDisplays();
}

function initializeFundAllocation() {
    const allocationBars = document.querySelectorAll('.allocation-fill');
    
    // Animate allocation bars on load
    setTimeout(() => {
        allocationBars.forEach(bar => {
            const targetWidth = bar.getAttribute('data-percentage') || '0';
            bar.style.width = `${targetWidth}%`;
        });
    }, 500);
}

function initializeFundingProposals() {
    const fundingCards = document.querySelectorAll('.funding-proposal-card');
    
    fundingCards.forEach(card => {
        const fundingBar = card.querySelector('.funding-fill');
        const targetPercentage = parseFloat(card.getAttribute('data-funding-progress')) || 0;
        
        // Animate funding progress
        setTimeout(() => {
            fundingBar.style.width = `${targetPercentage}%`;
        }, 300);
        
        // Add click handler for funding details
        card.addEventListener('click', () => {
            showFundingProposalDetails(card);
        });
    });
}

function showFundingProposalDetails(card) {
    const proposalTitle = card.querySelector('.proposal-title').textContent;
    const fundingAmount = card.querySelector('.funding-amount').textContent;
    const fundingPurpose = card.querySelector('.funding-purpose').textContent;
    
    // Create modal content (simplified)
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content">
            <h3>${proposalTitle}</h3>
            <p><strong>Requested Amount:</strong> ${fundingAmount}</p>
            <p><strong>Purpose:</strong> ${fundingPurpose}</p>
            <div class="modal-actions">
                <button class="btn btn-primary" onclick="supportFundingProposal()">Support Proposal</button>
                <button class="btn btn-secondary" onclick="closeModal()">Close</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
}

function supportFundingProposal() {
    showNotification('Funding proposal support recorded! Your vote has been added.', 'success');
    closeModal();
}

function closeModal() {
    const modal = document.querySelector('.modal-overlay');
    if (modal) {
        modal.remove();
    }
}

function updateTreasuryDisplays() {
    // Simulate real-time treasury updates
    const treasuryAmounts = document.querySelectorAll('.treasury-amount');
    
    treasuryAmounts.forEach(amount => {
        // Add subtle animation to show data is live
        amount.style.transition = 'color 0.3s ease';
        amount.addEventListener('mouseenter', () => {
            amount.style.color = 'var(--accent-color)';
        });
        amount.addEventListener('mouseleave', () => {
            amount.style.color = 'var(--primary-color)';
        });
    });
}

function initializeProposalCreation() {
    const createProposalBtn = document.querySelector('[data-action="create-proposal"]');
    
    if (createProposalBtn) {
        createProposalBtn.addEventListener('click', () => {
            showProposalCreationForm();
        });
    }
}

function showProposalCreationForm() {
    const formHTML = `
        <div class="modal-overlay">
            <div class="modal-content proposal-form">
                <h3><i class="fas fa-plus-circle"></i> Create New Proposal</h3>
                <form id="proposal-form">
                    <div class="form-group">
                        <label for="proposal-title">Proposal Title</label>
                        <input type="text" id="proposal-title" required placeholder="Enter proposal title...">
                    </div>
                    <div class="form-group">
                        <label for="proposal-description">Description</label>
                        <textarea id="proposal-description" rows="4" required placeholder="Describe your proposal in detail..."></textarea>
                    </div>
                    <div class="form-group">
                        <label for="proposal-type">Proposal Type</label>
                        <select id="proposal-type" required>
                            <option value="">Select type...</option>
                            <option value="governance">Governance Change</option>
                            <option value="funding">Funding Request</option>
                            <option value="technical">Technical Improvement</option>
                            <option value="community">Community Initiative</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="funding-amount">Funding Amount (if applicable)</label>
                        <input type="number" id="funding-amount" placeholder="0" min="0" step="100">
                        <small>Amount in FTNS tokens</small>
                    </div>
                    <div class="form-group">
                        <label for="voting-duration">Voting Duration</label>
                        <select id="voting-duration" required>
                            <option value="7">7 Days</option>
                            <option value="14">14 Days</option>
                            <option value="30">30 Days</option>
                        </select>
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-rocket"></i> Submit Proposal
                        </button>
                        <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                    </div>
                </form>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', formHTML);
    
    // Handle form submission
    document.getElementById('proposal-form').addEventListener('submit', handleProposalSubmission);
}

function handleProposalSubmission(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const proposalData = {
        title: formData.get('proposal-title') || document.getElementById('proposal-title').value,
        description: formData.get('proposal-description') || document.getElementById('proposal-description').value,
        type: formData.get('proposal-type') || document.getElementById('proposal-type').value,
        fundingAmount: formData.get('funding-amount') || document.getElementById('funding-amount').value,
        votingDuration: formData.get('voting-duration') || document.getElementById('voting-duration').value
    };
    
    // Validate required fields
    if (!proposalData.title || !proposalData.description || !proposalData.type) {
        showNotification('Please fill in all required fields.', 'error');
        return;
    }
    
    // Simulate proposal submission
    submitProposal(proposalData);
}

function submitProposal(proposalData) {
    // Show loading state
    const submitBtn = document.querySelector('#proposal-form button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
    submitBtn.disabled = true;
    
    // Simulate API call
    setTimeout(() => {
        // Success
        showNotification('Proposal submitted successfully! It will appear in the proposals list shortly.', 'success');
        closeModal();
        
        // Add proposal to the list (simplified)
        addProposalToList(proposalData);
        
        // Update governance metrics
        updateGovernanceMetrics();
    }, 2000);
}

function addProposalToList(proposalData) {
    const proposalsList = document.querySelector('.featured-proposals') || document.querySelector('#proposals-content');
    
    if (!proposalsList) return;
    
    const proposalHTML = `
        <div class="proposal-card" data-proposal-id="prop_${Date.now()}">
            <div class="proposal-header">
                <div>
                    <h4 class="proposal-title">${proposalData.title}</h4>
                    <div class="proposal-meta">
                        <span>Submitted by: You</span> • 
                        <span>Just now</span> • 
                        <span>Type: ${proposalData.type}</span>
                    </div>
                </div>
                <span class="proposal-status active">Active</span>
            </div>
            <p class="proposal-description">${proposalData.description}</p>
            ${proposalData.fundingAmount ? `<p><strong>Funding Requested:</strong> ${proposalData.fundingAmount} FTNS</p>` : ''}
            <div class="proposal-voting">
                <div class="voting-progress">
                    <div class="voting-bar">
                        <div class="vote-yes" style="width: 0%"></div>
                        <div class="vote-no" style="width: 0%"></div>
                    </div>
                    <div class="voting-stats">
                        <span>Yes: 0%</span>
                        <span>No: 0%</span>
                    </div>
                </div>
                <div class="voting-power">
                    <span class="power-amount">${getCurrentVotingPower()}</span>
                    <span class="power-label">Your Power</span>
                </div>
                <div class="voting-buttons">
                    <button class="vote-btn yes">Yes</button>
                    <button class="vote-btn no">No</button>
                </div>
            </div>
        </div>
    `;
    
    proposalsList.insertAdjacentHTML('afterbegin', proposalHTML);
    
    // Reinitialize voting for new proposal
    initializeProposalVoting();
}

function initializeGovernanceAnalytics() {
    // Initialize health indicators
    initializeHealthIndicators();
    
    // Initialize concentration monitoring
    initializeConcentrationMonitoring();
    
    // Update analytics displays
    updateAnalyticsDisplays();
}

function initializeHealthIndicators() {
    const healthIndicators = document.querySelectorAll('.health-indicator');
    
    healthIndicators.forEach(indicator => {
        const score = indicator.querySelector('.health-score');
        const value = parseFloat(score.textContent);
        
        // Add color class based on value
        if (value >= 85) {
            score.classList.add('excellent');
        } else if (value >= 70) {
            score.classList.add('good');
        } else if (value >= 50) {
            score.classList.add('warning');
        } else {
            score.classList.add('critical');
        }
        
        // Add hover effects
        indicator.addEventListener('mouseenter', () => {
            indicator.style.transform = 'translateY(-2px)';
        });
        
        indicator.addEventListener('mouseleave', () => {
            indicator.style.transform = 'translateY(0)';
        });
    });
}

function initializeConcentrationMonitoring() {
    const concentrationCards = document.querySelectorAll('.concentration-card');
    
    concentrationCards.forEach(card => {
        const score = card.querySelector('.concentration-score');
        const value = parseFloat(score.textContent);
        
        // Add color class based on concentration level
        if (value <= 0.3) {
            score.classList.add('safe');
        } else if (value <= 0.5) {
            score.classList.add('moderate');
        } else if (value <= 0.7) {
            score.classList.add('high');
        } else {
            score.classList.add('critical');
        }
    });
}

function updateAnalyticsDisplays() {
    // Simulate real-time updates
    const metricValues = document.querySelectorAll('.metric-value');
    
    metricValues.forEach(value => {
        value.addEventListener('mouseenter', () => {
            value.style.fontWeight = '600';
            value.style.color = 'var(--primary-color)';
        });
        
        value.addEventListener('mouseleave', () => {
            value.style.fontWeight = '500';
            value.style.color = 'var(--text-primary)';
        });
    });
}

function initializeActionCenter() {
    const actionButtons = document.querySelectorAll('.action-btn');
    
    actionButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            
            const action = button.getAttribute('data-action') || button.textContent.toLowerCase().trim();
            handleGovernanceAction(action, button);
        });
    });
}

function handleGovernanceAction(action, button) {
    const originalText = button.innerHTML;
    
    switch (action) {
        case 'create-proposal':
        case 'create proposal':
            showProposalCreationForm();
            break;
            
        case 'treasury-access':
        case 'access treasury':
            showTreasuryAccess();
            break;
            
        case 'delegate-vote':
        case 'delegate voting power':
            showDelegationForm();
            break;
            
        case 'governance-settings':
        case 'governance settings':
            showGovernanceSettings();
            break;
            
        case 'export-data':
        case 'export governance data':
            exportGovernanceData(button);
            break;
            
        default:
            showNotification(`Action "${action}" is not yet implemented.`, 'info');
    }
}

function showTreasuryAccess() {
    const accessHTML = `
        <div class="modal-overlay">
            <div class="modal-content">
                <h3><i class="fas fa-university"></i> Treasury Access</h3>
                <p>Current treasury balance: <strong>1,234,567 FTNS</strong></p>
                <div class="treasury-actions">
                    <button class="btn btn-primary" onclick="showFundingRequestForm()">Request Funding</button>
                    <button class="btn btn-secondary" onclick="showTreasuryHistory()">View History</button>
                    <button class="btn btn-secondary" onclick="closeModal()">Close</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', accessHTML);
}

function showDelegationForm() {
    const delegationHTML = `
        <div class="modal-overlay">
            <div class="modal-content">
                <h3><i class="fas fa-user-friends"></i> Delegate Voting Power</h3>
                <form id="delegation-form">
                    <div class="form-group">
                        <label for="delegate-address">Delegate Address</label>
                        <input type="text" id="delegate-address" placeholder="Enter delegate's wallet address..." required>
                    </div>
                    <div class="form-group">
                        <label for="delegation-amount">Amount to Delegate</label>
                        <input type="number" id="delegation-amount" placeholder="Enter FTNS amount..." min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="delegation-duration">Delegation Duration</label>
                        <select id="delegation-duration" required>
                            <option value="30">30 Days</option>
                            <option value="90">90 Days</option>
                            <option value="180">180 Days</option>
                            <option value="365">1 Year</option>
                        </select>
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="btn btn-primary">Delegate</button>
                        <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                    </div>
                </form>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', delegationHTML);
    
    document.getElementById('delegation-form').addEventListener('submit', (e) => {
        e.preventDefault();
        showNotification('Voting power delegated successfully!', 'success');
        closeModal();
    });
}

function showGovernanceSettings() {
    const settingsHTML = `
        <div class="modal-overlay">
            <div class="modal-content">
                <h3><i class="fas fa-cog"></i> Governance Settings</h3>
                <div class="settings-group">
                    <h4>Notification Preferences</h4>
                    <label><input type="checkbox" checked> New proposals</label>
                    <label><input type="checkbox" checked> Voting reminders</label>
                    <label><input type="checkbox"> Treasury updates</label>
                </div>
                <div class="settings-group">
                    <h4>Voting Preferences</h4>
                    <label><input type="checkbox"> Auto-delegate when offline</label>
                    <label><input type="checkbox" checked> Require confirmation for votes</label>
                </div>
                <div class="form-actions">
                    <button class="btn btn-primary" onclick="saveGovernanceSettings()">Save Settings</button>
                    <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', settingsHTML);
}

function saveGovernanceSettings() {
    showNotification('Governance settings saved successfully!', 'success');
    closeModal();
}

function exportGovernanceData(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Exporting...';
    button.disabled = true;
    
    setTimeout(() => {
        // Simulate data export
        const data = {
            proposals: 'governance_proposals.json',
            votes: 'voting_history.json',
            treasury: 'treasury_transactions.json',
            analytics: 'governance_analytics.json'
        };
        
        showNotification('Governance data exported successfully! Check your downloads folder.', 'success');
        
        button.innerHTML = originalText;
        button.disabled = false;
    }, 2000);
}

// Update functions for metrics and analytics

function updateGovernanceAnalytics(activeTab) {
    // Update analytics based on current tab
    console.log(`Updating analytics for tab: ${activeTab}`);
    
    // Simulate analytics updates
    const metricValues = document.querySelectorAll('.metric-value');
    metricValues.forEach(value => {
        // Add subtle flash to show data update
        value.style.transition = 'background-color 0.3s ease';
        value.style.backgroundColor = 'rgba(0, 123, 255, 0.1)';
        setTimeout(() => {
            value.style.backgroundColor = 'transparent';
        }, 300);
    });
}

function updateVoteParticipationMetrics() {
    // Update participation metrics after voting
    const participationMetric = document.querySelector('[data-metric="participation"] .metric-value');
    if (participationMetric) {
        const currentValue = parseFloat(participationMetric.textContent);
        participationMetric.textContent = `${(currentValue + 0.1).toFixed(1)}%`;
    }
}

function updateGovernanceMetrics() {
    // Update overall governance metrics
    const proposalCountMetric = document.querySelector('[data-metric="proposals"] .stat-value');
    if (proposalCountMetric) {
        const currentCount = parseInt(proposalCountMetric.textContent);
        proposalCountMetric.textContent = currentCount + 1;
    }
}

// Utility function for notifications
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    // Add styles with PRSM design language
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--card-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-left: 4px solid ${type === 'success' ? 'var(--prsm-success)' : type === 'error' ? 'var(--prsm-error)' : 'var(--prsm-primary)'};
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        z-index: 10000;
        max-width: 400px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.9rem;
        font-family: var(--font-main);
        font-weight: 500;
        animation: slideIn 0.3s ease;
        opacity: 1;
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Add CSS animations for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification-close {
        background: none;
        border: none;
        font-size: 1.2rem;
        cursor: pointer;
        color: var(--text-secondary);
        padding: 0;
        margin-left: auto;
    }
    
    .notification-close:hover {
        color: var(--text-primary);
    }
    
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    
    .modal-content {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 24px;
        max-width: 500px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .modal-content h3 {
        margin-top: 0;
        margin-bottom: 20px;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .form-group {
        margin-bottom: 16px;
    }
    
    .form-group label {
        display: block;
        margin-bottom: 4px;
        font-weight: 500;
        color: var(--text-primary);
    }
    
    .form-group input,
    .form-group select,
    .form-group textarea {
        width: 100%;
        padding: 8px 12px;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        background: var(--panel-bg);
        color: var(--text-primary);
        font-size: 0.9rem;
        box-sizing: border-box;
    }
    
    .form-group small {
        color: var(--text-secondary);
        font-size: 0.8rem;
    }
    
    .form-actions {
        display: flex;
        gap: 12px;
        justify-content: flex-end;
        margin-top: 20px;
    }
    
    .btn {
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .btn-primary {
        background: var(--primary-color);
        color: white;
    }
    
    .btn-primary:hover {
        background: var(--primary-color-dark);
    }
    
    .btn-secondary {
        background: var(--card-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    
    .btn-secondary:hover {
        background: var(--border-color);
    }
    
    .settings-group {
        margin-bottom: 20px;
    }
    
    .settings-group h4 {
        margin-bottom: 12px;
        color: var(--text-primary);
    }
    
    .settings-group label {
        display: block;
        margin-bottom: 8px;
        cursor: pointer;
        color: var(--text-primary);
    }
    
    .settings-group input[type="checkbox"] {
        margin-right: 8px;
    }
    
    .treasury-actions {
        display: flex;
        gap: 12px;
        justify-content: center;
        margin-top: 20px;
    }
`;

document.head.appendChild(notificationStyles);

    // === Tokenomics Tab Functionality ===
    
    // Tokenomics Tab Navigation
    function setupTokenomicsNavigation() {
        const tokenomicsNavTabs = document.querySelectorAll('.tokenomics-nav-tab');
        const tokenomicsTabContents = document.querySelectorAll('.tokenomics-tab-content');
        
        if (tokenomicsNavTabs.length === 0) return;
        
        tokenomicsNavTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.dataset.tab;
                
                // Update active tab
                tokenomicsNavTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Update active content
                tokenomicsTabContents.forEach(content => {
                    content.classList.remove('active');
                    if (content.id === targetTab + '-content') {
                        content.classList.add('active');
                    }
                });
            });
        });
        
        // Set first tab as active by default
        if (tokenomicsNavTabs[0]) {
            tokenomicsNavTabs[0].click();
        }
    }
    
    // Budget Allocation Sliders
    function setupBudgetSliders() {
        const sliders = document.querySelectorAll('.slider-input');
        const totalBudget = 50000; // Example total FTNS budget
        
        sliders.forEach(slider => {
            slider.addEventListener('input', (e) => {
                const value = e.target.value;
                const category = e.target.dataset.category;
                const valueDisplay = e.target.parentElement.querySelector('.slider-value');
                const allocatedAmount = Math.round((value / 100) * totalBudget);
                
                if (valueDisplay) {
                    valueDisplay.textContent = `${value}% (${allocatedAmount.toLocaleString()} FTNS)`;
                }
                
                // Update donut chart visualization
                updateAllocationChart();
            });
            
            // Initialize slider values
            slider.dispatchEvent(new Event('input'));
        });
    }
    
    // Update Allocation Donut Chart
    function updateAllocationChart() {
        const sliders = document.querySelectorAll('.slider-input');
        const donutChart = document.querySelector('.donut-chart');
        const chartTotal = document.querySelector('.chart-total');
        const legend = document.querySelector('.allocation-legend');
        
        if (!donutChart || !sliders.length) return;
        
        const categories = [];
        const colors = ['#ffffff', '#cccccc', '#999999', '#666666', '#333333'];
        let total = 0;
        
        sliders.forEach((slider, index) => {
            const value = parseInt(slider.value);
            const category = slider.dataset.category || `Category ${index + 1}`;
            categories.push({
                name: category,
                value: value,
                color: colors[index % colors.length]
            });
            total += value;
        });
        
        // Update chart
        let cumulativePercentage = 0;
        let segments = '';
        
        categories.forEach(category => {
            const percentage = category.value;
            const startAngle = (cumulativePercentage / 100) * 360;
            const endAngle = ((cumulativePercentage + percentage) / 100) * 360;
            
            const x1 = 50 + 40 * Math.cos((startAngle * Math.PI) / 180);
            const y1 = 50 + 40 * Math.sin((startAngle * Math.PI) / 180);
            const x2 = 50 + 40 * Math.cos((endAngle * Math.PI) / 180);
            const y2 = 50 + 40 * Math.sin((endAngle * Math.PI) / 180);
            
            const largeArcFlag = percentage > 50 ? 1 : 0;
            
            segments += `
                <path d="M 50,50 L ${x1},${y1} A 40,40 0 ${largeArcFlag},1 ${x2},${y2} Z"
                      fill="${category.color}" stroke="var(--bg-primary)" stroke-width="1"/>
            `;
            
            cumulativePercentage += percentage;
        });
        
        donutChart.innerHTML = `
            <svg viewBox="0 0 100 100">
                ${segments}
                <circle cx="50" cy="50" r="25" fill="var(--bg-primary)"/>
            </svg>
        `;
        
        // Update center text
        if (chartTotal) {
            chartTotal.textContent = `${total}%`;
        }
        
        // Update legend
        if (legend) {
            legend.innerHTML = categories.map(category => `
                <div class="legend-item">
                    <div class="legend-color" style="background-color: ${category.color}"></div>
                    <span class="legend-text">${category.name}: ${category.value}%</span>
                </div>
            `).join('');
        }
    }
    
    // Budget Scenario Planning
    function setupScenarioPlanning() {
        const scenarioBtns = document.querySelectorAll('.scenario-btn');
        const forecastChart = document.querySelector('.forecast-chart');
        
        scenarioBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Update active scenario
                scenarioBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const scenario = btn.dataset.scenario;
                updateForecastChart(scenario);
            });
        });
        
        // Set first scenario as active by default
        if (scenarioBtns[0]) {
            scenarioBtns[0].click();
        }
    }
    
    // Update Forecast Chart based on scenario
    function updateForecastChart(scenario) {
        const forecastChart = document.querySelector('.forecast-chart');
        if (!forecastChart) return;
        
        const scenarios = {
            conservative: 'Conservative: 5% monthly growth, 18 months runway',
            optimistic: 'Optimistic: 15% monthly growth, 12 months runway',
            aggressive: 'Aggressive: 25% monthly growth, 8 months runway'
        };
        
        forecastChart.textContent = scenarios[scenario] || 'Forecast visualization placeholder';
    }
    
    // Budget Comparison Progress Bars
    function setupBudgetComparison() {
        const comparisonItems = document.querySelectorAll('.comparison-item');
        
        comparisonItems.forEach(item => {
            const progressBar = item.querySelector('.comparison-progress');
            const budgetValue = item.querySelector('.budget-value');
            const actualValue = item.querySelector('.actual-value');
            
            if (progressBar && budgetValue && actualValue) {
                const budget = parseFloat(budgetValue.textContent.replace(/[^\d.]/g, ''));
                const actual = parseFloat(actualValue.textContent.replace(/[^\d.]/g, ''));
                const percentage = Math.min((actual / budget) * 100, 100);
                
                progressBar.style.width = `${percentage}%`;
                
                // Add color coding
                if (percentage > 90) {
                    progressBar.style.backgroundColor = '#ef4444'; // Red for over budget
                } else if (percentage > 70) {
                    progressBar.style.backgroundColor = '#f59e0b'; // Orange for approaching budget
                } else {
                    progressBar.style.backgroundColor = 'var(--accent-primary)'; // Normal color
                }
            }
        });
    }
    
    // Expense Filtering
    function setupExpenseFiltering() {
        const filterSelects = document.querySelectorAll('.filter-select');
        const expenseItems = document.querySelectorAll('.expense-item');
        
        filterSelects.forEach(select => {
            select.addEventListener('change', () => {
                filterExpenses();
            });
        });
        
        function filterExpenses() {
            const categoryFilter = document.querySelector('.filter-select[data-filter="category"]')?.value || 'all';
            const timeFilter = document.querySelector('.filter-select[data-filter="time"]')?.value || 'all';
            
            expenseItems.forEach(item => {
                let show = true;
                
                // Apply category filter
                if (categoryFilter !== 'all') {
                    const itemCategory = item.dataset.category;
                    if (itemCategory !== categoryFilter) {
                        show = false;
                    }
                }
                
                // Apply time filter
                if (timeFilter !== 'all') {
                    const itemDate = new Date(item.dataset.date);
                    const now = new Date();
                    const daysDiff = Math.floor((now - itemDate) / (1000 * 60 * 60 * 24));
                    
                    if (timeFilter === 'week' && daysDiff > 7) show = false;
                    if (timeFilter === 'month' && daysDiff > 30) show = false;
                    if (timeFilter === 'quarter' && daysDiff > 90) show = false;
                }
                
                item.style.display = show ? 'flex' : 'none';
            });
        }
    }
    
    // Budget Alert Management
    function setupBudgetAlerts() {
        const alertItems = document.querySelectorAll('.alert-item');
        
        alertItems.forEach(alert => {
            alert.addEventListener('click', () => {
                // Mark alert as read or expand details
                alert.style.opacity = '0.7';
                
                // Could implement alert dismissal or detailed view
                showNotification('Alert acknowledged', 'info');
            });
        });
    }
    
    // Cost Savings Tracking
    function setupCostSavings() {
        const savingsItems = document.querySelectorAll('.savings-item');
        
        savingsItems.forEach(item => {
            item.addEventListener('click', () => {
                const title = item.querySelector('.savings-title')?.textContent;
                const amount = item.querySelector('.savings-amount')?.textContent;
                
                showNotification(`${title}: ${amount} potential savings`, 'success');
            });
        });
    }
    
    // Budget Export Functionality
    function setupBudgetExport() {
        const exportBtn = document.querySelector('.tokenomics-actions .btn[data-action="export"]');
        
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                // Simulate budget data export
                const budgetData = {
                    overview: {
                        availableFTNS: 45000,
                        allocatedFTNS: 35000,
                        monthlyBurn: 5000,
                        monthsRunway: 9
                    },
                    allocation: {
                        research: 40,
                        development: 30,
                        marketing: 15,
                        operations: 10,
                        reserve: 5
                    },
                    exportDate: new Date().toISOString()
                };
                
                // Create and download JSON file
                const dataStr = JSON.stringify(budgetData, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `prsm_budget_${new Date().toISOString().split('T')[0]}.json`;
                link.click();
                
                URL.revokeObjectURL(url);
                showNotification('Budget data exported successfully', 'success');
            });
        }
    }
    
    // Budget Creation Modal
    function setupBudgetCreation() {
        const createBtn = document.querySelector('.tokenomics-actions .btn[data-action="buy-ftns"]');
        
        if (createBtn) {
            createBtn.addEventListener('click', () => {
                showFTNSPurchaseModal();
            });
        }
    }
    
    function showFTNSPurchaseModal() {
        const modal = createModal('Purchase FTNS Tokens', `
            <div class="ftns-purchase-form">
                <div class="form-group">
                    <label for="ftns-amount">FTNS Amount</label>
                    <input type="number" id="ftns-amount" placeholder="1000" min="100">
                    <small>Minimum purchase: 100 FTNS</small>
                </div>
                <div class="form-group">
                    <label for="payment-method">Payment Method</label>
                    <select id="payment-method">
                        <option value="credit-card">Credit Card</option>
                        <option value="crypto">Cryptocurrency</option>
                        <option value="bank-transfer">Bank Transfer</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="purchase-purpose">Purpose (Optional)</label>
                    <select id="purchase-purpose">
                        <option value="">Select purpose...</option>
                        <option value="research">Research & Development</option>
                        <option value="staking">Token Staking</option>
                        <option value="marketplace">Marketplace Transactions</option>
                        <option value="governance">Governance Participation</option>
                    </select>
                </div>
                <div class="purchase-summary">
                    <div class="summary-line">
                        <span>FTNS Rate:</span>
                        <span>$2.00 USD per FTNS</span>
                    </div>
                    <div class="summary-line total">
                        <span>Total Cost:</span>
                        <span id="total-cost">$2,000.00 USD</span>
                    </div>
                </div>
            </div>
        `, [
            {
                text: 'Purchase FTNS',
                class: 'primary',
                action: () => {
                    const amount = document.getElementById('ftns-amount').value;
                    const paymentMethod = document.getElementById('payment-method').value;
                    const purpose = document.getElementById('purchase-purpose').value;
                    
                    if (amount && paymentMethod) {
                        // Simulate FTNS purchase
                        showNotification(`Successfully purchased ${amount} FTNS tokens`, 'success');
                        closeModal();
                    } else {
                        showNotification('Please fill in required fields', 'error');
                    }
                }
            },
            {
                text: 'Cancel',
                class: 'secondary',
                action: closeModal
            }
        ]);
        
        // Add real-time cost calculation
        const amountInput = document.getElementById('ftns-amount');
        const totalCostSpan = document.getElementById('total-cost');
        if (amountInput && totalCostSpan) {
            amountInput.addEventListener('input', () => {
                const amount = parseFloat(amountInput.value) || 0;
                const cost = amount * 2.00; // $2.00 per FTNS
                totalCostSpan.textContent = `$${cost.toLocaleString('en-US', {minimumFractionDigits: 2})} USD`;
            });
        }
    }
    
    // Initialize Tokenomics Management functionality
    function initializeTokenomicsManagement() {
        // Set up tokenomics tab functionality when tokenomics nav button is clicked
        const tokenomicsNavBtn = document.querySelector('[data-target="tokenomics-content"]');
        if (tokenomicsNavBtn) {
            tokenomicsNavBtn.addEventListener('click', () => {
                // Delay initialization to ensure DOM is ready
                setTimeout(() => {
                    setupTokenomicsNavigation();
                    setupBudgetSliders();
                    setupScenarioPlanning();
                    setupBudgetComparison();
                    setupExpenseFiltering();
                    setupBudgetAlerts();
                    setupCostSavings();
                    setupBudgetExport();
                    setupBudgetCreation();
                }, 100);
            });
        }
    }
    
    // Initialize tokenomics management when DOM is loaded
    initializeTokenomicsManagement();

// ============================================================================
// MARKETPLACE ASSET TYPE SWITCHING FUNCTIONALITY
// ============================================================================

function initializeMarketplaceAssetSwitching() {
    /**
     * Initialize marketplace asset type navigation and filtering
     * Enables interactive switching between different marketplace asset types
     */
    const assetTypeCards = document.querySelectorAll('.asset-type-card');
    const assetCards = document.querySelectorAll('.asset-card');
    
    console.log(`Found ${assetTypeCards.length} asset type cards`);
    if (assetTypeCards.length === 0) return;
    
    // Initialize with the active asset type card
    const activeCard = document.querySelector('.asset-type-card.active');
    if (activeCard) {
        const activeType = activeCard.getAttribute('data-type');
        updateMarketplaceStats(activeType, activeCard);
    }
    
    // Add click handlers to asset type cards
    assetTypeCards.forEach(card => {
        console.log('Adding click handler to:', card.getAttribute('data-type'));
        card.addEventListener('click', () => {
            const selectedType = card.getAttribute('data-type');
            console.log('Asset type card clicked:', selectedType);
            
            // Special handling for data work vs other asset types
            if (selectedType === 'data_work') {
                // This will be handled by the data work toggle function
                return;
            } else {
                // If we're currently in data work hub, return to main marketplace
                const dataWorkHub = document.getElementById('data-work-hub');
                if (dataWorkHub && dataWorkHub.style.display !== 'none') {
                    showMainMarketplace();
                }
            }
            
            // Update active state
            assetTypeCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            
            // Filter asset cards based on selected type
            filterAssetCards(selectedType);
            
            // Update marketplace stats based on selection
            updateMarketplaceStats(selectedType, card);
            
            // Show assets for the selected type
            showAssetsByType(selectedType);
            
            // Smooth scroll to assets if needed
            const assetsSection = document.querySelector('.marketplace-assets');
            if (assetsSection) {
                assetsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
        
        // Add hover effect
        card.addEventListener('mouseenter', () => {
            if (!card.classList.contains('active')) {
                card.style.transform = 'translateY(-2px)';
            }
        });
        
        card.addEventListener('mouseleave', () => {
            if (!card.classList.contains('active')) {
                card.style.transform = 'translateY(0)';
            }
        });
    });
    
    // Initialize with default selection (AI Models)
    const defaultCard = document.querySelector('.asset-type-card.active');
    if (defaultCard) {
        const defaultType = defaultCard.getAttribute('data-type');
        filterAssetCards(defaultType);
    }
}

function filterAssetCards(selectedType) {
    /**
     * Filter and display asset cards based on selected type
     * @param {string} selectedType - The selected asset type to filter by
     */
    const assetCards = document.querySelectorAll('.asset-card');
    const allAssetsContainer = document.querySelector('.marketplace-assets');
    
    if (!assetCards.length || !allAssetsContainer) return;
    
    // Show loading state
    showMarketplaceLoading(true);
    
    setTimeout(() => {
        assetCards.forEach(card => {
            const cardType = card.getAttribute('data-asset-type');
            
            if (!cardType || cardType === selectedType || selectedType === 'all') {
                card.style.display = 'block';
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                
                // Animate in
                setTimeout(() => {
                    card.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, Math.random() * 100);
            } else {
                card.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
                card.style.opacity = '0';
                card.style.transform = 'translateY(-20px)';
                
                setTimeout(() => {
                    card.style.display = 'none';
                }, 200);
            }
        });
        
        // Hide loading state
        setTimeout(() => {
            showMarketplaceLoading(false);
        }, 300);
    }, 100);
}

function updateMarketplaceStats(selectedType, selectedCard) {
    /**
     * Update marketplace statistics display based on selected asset type
     * @param {string} selectedType - The selected asset type
     * @param {Element} selectedCard - The selected asset type card element
     */
    const statsElement = document.querySelector('.marketplace-stats');
    const headerStatsElement = document.querySelector('.section-header .stats');
    const discoveryTitleElement = document.getElementById('discovery-title');
    
    if (!selectedCard) return;
    
    const assetCount = selectedCard.querySelector('.asset-count')?.textContent || '0';
    const assetName = selectedCard.querySelector('h6')?.textContent || 'Assets';
    
    // Update discovery section title based on selected asset type
    if (discoveryTitleElement) {
        const discoveryTitles = {
            'ai_model': 'AI Model Discovery',
            'dataset': 'Dataset Discovery',
            'agent_workflow': 'AI Agent Discovery',
            'mcp_tool': 'MCP Tool Discovery',
            'compute_resource': 'Compute Resource Discovery',
            'knowledge_resource': 'Knowledge Resource Discovery',
            'evaluation_service': 'Evaluation Service Discovery',
            'training_service': 'Training Service Discovery',
            'safety_tool': 'Safety Tool Discovery',
            'data_work': 'Data Work Discovery'
        };
        
        const newTitle = discoveryTitles[selectedType] || 'Asset Discovery';
        discoveryTitleElement.textContent = newTitle;
        console.log('✅ Updated discovery title to:', newTitle, 'for type:', selectedType);
    }
    
    // Update header stats
    if (headerStatsElement) {
        if (selectedType === 'all' || !selectedType) {
            headerStatsElement.textContent = '8,847 Total Assets • 9 Asset Types';
        } else {
            headerStatsElement.textContent = `${assetCount} ${assetName} • Active Category`;
        }
    }
    
    // Update detailed stats if available
    if (statsElement) {
        const statCards = statsElement.querySelectorAll('.stat-card');
        statCards.forEach(statCard => {
            const statTitle = statCard.querySelector('.stat-title');
            const statValue = statCard.querySelector('.stat-value');
            
            if (statTitle && statValue) {
                // Update stats based on selected type
                if (statTitle.textContent.includes('Assets')) {
                    statValue.textContent = assetCount;
                } else if (statTitle.textContent.includes('Category')) {
                    statValue.textContent = assetName;
                }
            }
        });
    }
    
    // Update results info message
    updateResultsInfo(selectedType);
}

function updateResultsInfo(selectedType) {
    /**
     * Update the "Showing X of Y items" message based on selected asset type
     */
    const resultsInfoElement = document.querySelector('.results-info');
    if (!resultsInfoElement) return;
    
    // Define reasonable total counts and appropriate terms for each category
    const categoryInfo = {
        'ai_model': { total: 2847, term: 'models' },
        'dataset': { total: 1523, term: 'datasets' },
        'agent_workflow': { total: 892, term: 'workflows' },
        'mcp_tool': { total: 634, term: 'tools' },
        'compute_resource': { total: 387, term: 'resources' },
        'knowledge_resource': { total: 756, term: 'resources' },
        'evaluation_service': { total: 295, term: 'services' },
        'training_service': { total: 178, term: 'services' },
        'safety_tool': { total: 423, term: 'tools' },
        'data_work': { total: 2847, term: 'jobs' },
        'api_integration': { total: 1156, term: 'integrations' },
        'monitoring_analytics': { total: 892, term: 'tools' }
    };
    
    const info = categoryInfo[selectedType] || { total: 500, term: 'items' };
    const shownCount = 6; // We always show 6 simulated options
    
    resultsInfoElement.textContent = `Showing ${shownCount} of ${info.total.toLocaleString()} ${info.term}`;
    
    // Also update the "Load More" button text to match the category
    const loadMoreBtn = document.getElementById('load-more-models');
    if (loadMoreBtn) {
        const capitalizedTerm = info.term.charAt(0).toUpperCase() + info.term.slice(1);
        loadMoreBtn.textContent = `Load More ${capitalizedTerm}`;
    }
    
    console.log(`Updated results info: Showing ${shownCount} of ${info.total.toLocaleString()} ${info.term}`);
}

function showMarketplaceLoading(show) {
    /**
     * Show or hide marketplace loading state
     * @param {boolean} show - Whether to show loading state
     */
    let loadingElement = document.querySelector('.marketplace-loading');
    
    if (show && !loadingElement) {
        loadingElement = document.createElement('div');
        loadingElement.className = 'marketplace-loading';
        loadingElement.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-spinner fa-spin"></i>
                <span>Loading assets...</span>
            </div>
        `;
        
        const assetsContainer = document.querySelector('.marketplace-assets');
        if (assetsContainer) {
            assetsContainer.style.position = 'relative';
            assetsContainer.appendChild(loadingElement);
        }
    } else if (!show && loadingElement) {
        loadingElement.remove();
    }
}

function showAssetsByType(selectedType) {
    /**
     * Show asset examples based on selected type
     * @param {string} selectedType - The selected asset type
     */
    const modelCards = document.querySelectorAll('.model-card');
    const categoryFilter = document.getElementById('category-filter');
    
    // Hide all model cards first
    modelCards.forEach(card => {
        card.style.display = 'none';
    });
    
    // Show cards matching the selected type
    const matchingCards = document.querySelectorAll(`.model-card[data-type="${selectedType}"]`);
    matchingCards.forEach(card => {
        card.style.display = 'block';
    });
    
    // Update category filter options
    if (categoryFilter) {
        // Hide all optgroups first
        const optgroups = categoryFilter.querySelectorAll('optgroup');
        optgroups.forEach(group => {
            group.style.display = 'none';
        });
        
        // Show the relevant optgroup
        const categoryMapping = {
            'ai_model': 'ai-model-categories',
            'dataset': 'dataset-categories', 
            'agent_workflow': 'agent-categories',
            'mcp_tool': 'tool-categories',
            'compute_resource': 'compute-categories',
            'knowledge_resource': 'knowledge-categories',
            'evaluation_service': 'evaluation-categories',
            'training_service': 'training-categories',
            'safety_tool': 'safety-categories',
            'data_work': 'data-work-categories'
        };
        
        const targetGroupId = categoryMapping[selectedType];
        if (targetGroupId) {
            const targetGroup = document.getElementById(targetGroupId);
            if (targetGroup) {
                targetGroup.style.display = 'block';
            }
        }
    }
    
    console.log('✅ Showing assets for type:', selectedType, 'Found', matchingCards.length, 'matching cards');
}

function initializeAssetCardInteractions() {
    /**
     * Initialize interactions for individual asset cards
     */
    const assetCards = document.querySelectorAll('.asset-card');
    
    assetCards.forEach(card => {
        // Add hover effects
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-4px)';
            card.style.boxShadow = '0 8px 25px rgba(255, 255, 255, 0.1)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = '0 2px 10px rgba(255, 255, 255, 0.05)';
        });
        
        // Add click handlers for asset cards
        card.addEventListener('click', (e) => {
            // Don't trigger if clicking on buttons
            if (e.target.closest('.asset-actions')) return;
            
            const assetName = card.querySelector('.asset-title')?.textContent || 'Unknown Asset';
            const assetType = card.getAttribute('data-asset-type') || 'unknown';
            
            showAssetDetails(assetName, assetType, card);
        });
        
        // Add click handlers for asset action buttons
        const actionButtons = card.querySelectorAll('.asset-actions .action-btn');
        actionButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const action = button.getAttribute('data-action') || button.textContent.toLowerCase();
                const assetName = card.querySelector('.asset-title')?.textContent || 'Unknown Asset';
                
                handleAssetAction(action, assetName, card);
            });
        });
    });
}

function showAssetDetails(assetName, assetType, cardElement) {
    /**
     * Show detailed view of a marketplace asset
     * @param {string} assetName - Name of the asset
     * @param {string} assetType - Type of the asset
     * @param {Element} cardElement - The asset card element
     */
    console.log(`Opening details for ${assetType}: ${assetName}`);
    
    // Create modal or navigate to detail view
    const modal = document.createElement('div');
    modal.className = 'asset-detail-modal';
    modal.innerHTML = `
        <div class="modal-overlay" onclick="this.parentElement.remove()"></div>
        <div class="modal-content">
            <div class="modal-header">
                <h3><i class="fas fa-info-circle"></i> ${assetName}</h3>
                <button class="modal-close" onclick="this.closest('.asset-detail-modal').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div class="asset-detail-info">
                    <p><strong>Type:</strong> ${assetType.replace('_', ' ').toUpperCase()}</p>
                    <p><strong>Status:</strong> Available</p>
                    <p><strong>Category:</strong> ${getAssetCategory(assetType)}</p>
                    <p>This is a detailed view of the selected marketplace asset. In a production environment, this would show comprehensive information including specifications, pricing, reviews, and usage examples.</p>
                </div>
                <div class="asset-detail-actions">
                    <button class="btn btn-primary" onclick="handleAssetAction('view', '${assetName}')">
                        <i class="fas fa-eye"></i> View Details
                    </button>
                    <button class="btn btn-secondary" onclick="handleAssetAction('download', '${assetName}')">
                        <i class="fas fa-download"></i> Download
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Animate in
    setTimeout(() => {
        modal.style.opacity = '1';
        modal.querySelector('.modal-content').style.transform = 'scale(1)';
    }, 10);
}

function handleAssetAction(action, assetName, cardElement) {
    /**
     * Handle actions performed on marketplace assets
     * @param {string} action - The action to perform
     * @param {string} assetName - Name of the asset
     * @param {Element} cardElement - The asset card element
     */
    console.log(`Performing action "${action}" on asset: ${assetName}`);
    
    // Show action feedback
    showActionFeedback(action, assetName);
    
    switch (action) {
        case 'download':
        case 'get':
            handleAssetDownload(assetName);
            break;
        case 'view':
        case 'details':
            handleAssetView(assetName);
            break;
        case 'favorite':
        case 'bookmark':
            handleAssetFavorite(assetName, cardElement);
            break;
        case 'share':
            handleAssetShare(assetName);
            break;
        default:
            console.log(`Unknown action: ${action}`);
    }
}

function getAssetCategory(assetType) {
    /**
     * Get human-readable category name for asset type
     * @param {string} assetType - The asset type identifier
     * @returns {string} - Human-readable category name
     */
    const categories = {
        'ai_model': 'Artificial Intelligence Models',
        'dataset': 'Data & Datasets',
        'agent_workflow': 'AI Agents & Workflows',
        'mcp_tool': 'Model Context Protocol Tools',
        'compute_resource': 'Computational Resources',
        'knowledge_resource': 'Knowledge Resources',
        'evaluation_service': 'Evaluation Services',
        'training_service': 'Training Services',
        'safety_tool': 'Safety & Governance Tools'
    };
    
    return categories[assetType] || 'Unknown Category';
}

function showActionFeedback(action, assetName) {
    /**
     * Show user feedback for performed actions
     * @param {string} action - The performed action
     * @param {string} assetName - Name of the asset
     */
    const feedback = document.createElement('div');
    feedback.className = 'action-feedback';
    feedback.innerHTML = `
        <div class="feedback-content">
            <i class="fas fa-check-circle"></i>
            <span>Action "${action}" performed on "${assetName}"</span>
        </div>
    `;
    
    document.body.appendChild(feedback);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        feedback.remove();
    }, 3000);
}

function handleAssetDownload(assetName) {
    console.log(`Downloading asset: ${assetName}`);
    // Implement actual download logic
}

function handleAssetView(assetName) {
    console.log(`Viewing asset details: ${assetName}`);
    // Implement detailed view logic
}

function handleAssetFavorite(assetName, cardElement) {
    console.log(`Favoriting asset: ${assetName}`);
    
    // Toggle favorite state
    const favoriteBtn = cardElement.querySelector('.action-btn[data-action="favorite"]');
    if (favoriteBtn) {
        favoriteBtn.classList.toggle('favorited');
        const icon = favoriteBtn.querySelector('i');
        if (icon) {
            icon.className = favoriteBtn.classList.contains('favorited') 
                ? 'fas fa-heart' 
                : 'far fa-heart';
        }
    }
}

function handleAssetShare(assetName) {
    console.log(`Sharing asset: ${assetName}`);
    
    if (navigator.share) {
        navigator.share({
            title: `PRSM Marketplace - ${assetName}`,
            text: `Check out this asset on PRSM Marketplace: ${assetName}`,
            url: window.location.href
        });
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(window.location.href).then(() => {
            showActionFeedback('copied link', assetName);
        });
    }
}

// Initialize marketplace functionality when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Add delay to ensure marketplace elements are loaded
    setTimeout(() => {
        console.log('🛒 Initializing marketplace functionality...');
        try {
            initializeMarketplaceAssetSwitching();
            initializeAssetCardInteractions();
            initializeModelLabFunctionality();
            initializeDataWorkFunctionality();
            console.log('✅ Marketplace functionality initialized successfully.');
        } catch (error) {
            console.error('❌ Error initializing marketplace:', error);
        }
    }, 300);
});

// ========================================
// MODEL LAB FUNCTIONALITY
// ========================================

function initializeModelLabFunctionality() {
    // Initialize workflow step navigation
    initializeWorkflowSteps();
    
    // Initialize model selection
    initializeModelSelection();
    
    // Initialize upload functionality
    initializeDatasetUpload();
    
    // Initialize accuracy slider
    initializeAccuracySlider();
    
    // Initialize marketplace sharing toggle
    initializeMarketplaceSharing();
    
    // Initialize distillation controls
    initializeDistillationControls();
}

// Workflow step navigation
function initializeWorkflowSteps() {
    const steps = document.querySelectorAll('.workflow-step');
    const nextBtn = document.getElementById('next-step');
    const prevBtn = document.getElementById('prev-step');
    const startBtn = document.getElementById('start-distillation');
    const progressFill = document.querySelector('.workflow-controls .progress-fill');
    const currentStepText = document.querySelector('.current-step');
    
    let currentStep = 0;
    const totalSteps = steps.length;
    
    function updateStepDisplay() {
        // Update step visibility
        steps.forEach((step, index) => {
            step.classList.toggle('active', index === currentStep);
        });
        
        // Update progress bar
        const progressPercent = ((currentStep + 1) / totalSteps) * 100;
        if (progressFill) {
            progressFill.style.width = `${progressPercent}%`;
        }
        
        // Update step text
        if (currentStepText) {
            currentStepText.textContent = `Step ${currentStep + 1} of ${totalSteps}`;
        }
        
        // Update button states
        if (prevBtn) {
            prevBtn.disabled = currentStep === 0;
        }
        
        if (nextBtn && startBtn) {
            if (currentStep === totalSteps - 1) {
                nextBtn.style.display = 'none';
                startBtn.style.display = 'inline-flex';
            } else {
                nextBtn.style.display = 'inline-flex';
                startBtn.style.display = 'none';
            }
        }
    }
    
    // Next button handler
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (currentStep < totalSteps - 1) {
                currentStep++;
                updateStepDisplay();
            }
        });
    }
    
    // Previous button handler
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentStep > 0) {
                currentStep--;
                updateStepDisplay();
            }
        });
    }
    
    // Start distillation button handler
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            startDistillationProcess();
        });
    }
    
    // Initialize display
    updateStepDisplay();
}

// Model selection functionality
function initializeModelSelection() {
    const modelCards = document.querySelectorAll('.model-card');
    
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            // Remove selected class from all cards
            modelCards.forEach(c => c.classList.remove('selected'));
            
            // Add selected class to clicked card
            card.classList.add('selected');
            
            // Update the button text
            const selectButton = card.querySelector('.btn');
            if (selectButton) {
                selectButton.textContent = 'Selected';
                selectButton.classList.remove('secondary');
                selectButton.classList.add('primary');
            }
            
            // Reset other buttons
            modelCards.forEach(otherCard => {
                if (otherCard !== card) {
                    const otherButton = otherCard.querySelector('.btn');
                    if (otherButton) {
                        otherButton.textContent = 'Select';
                        otherButton.classList.remove('primary');
                        otherButton.classList.add('secondary');
                    }
                }
            });
        });
    });
}

// Dataset upload functionality
function initializeDatasetUpload() {
    const uploadOptions = document.querySelectorAll('.upload-option');
    const uploadDropzone = document.querySelector('.upload-dropzone');
    const chooseFilesBtn = uploadDropzone?.querySelector('.btn');
    
    // Upload option switching
    uploadOptions.forEach(option => {
        option.addEventListener('click', () => {
            uploadOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
            
            // You could show/hide different upload interfaces here
            // based on the selected option (local, cloud, database)
        });
    });
    
    // File selection
    if (chooseFilesBtn) {
        chooseFilesBtn.addEventListener('click', () => {
            // Create a hidden file input
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.multiple = true;
            fileInput.accept = '.csv,.json,.jsonl,.txt,.zip';
            
            fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                handleFileSelection(files);
            });
            
            fileInput.click();
        });
    }
    
    // Drag and drop functionality
    if (uploadDropzone) {
        uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDropzone.style.borderColor = 'var(--accent-primary)';
            uploadDropzone.style.background = 'var(--bg-tertiary)';
        });
        
        uploadDropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadDropzone.style.borderColor = 'var(--border-color)';
            uploadDropzone.style.background = 'var(--bg-primary)';
        });
        
        uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDropzone.style.borderColor = 'var(--border-color)';
            uploadDropzone.style.background = 'var(--bg-primary)';
            
            const files = Array.from(e.dataTransfer.files);
            handleFileSelection(files);
        });
    }
}

function handleFileSelection(files) {
    const fileList = document.querySelector('.file-list');
    if (!fileList) return;
    
    files.forEach(file => {
        // Create file item element
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        // Determine file icon based on extension
        const extension = file.name.split('.').pop().toLowerCase();
        let iconClass = 'fas fa-file';
        
        switch (extension) {
            case 'csv':
                iconClass = 'fas fa-file-csv';
                break;
            case 'json':
            case 'jsonl':
                iconClass = 'fas fa-file-code';
                break;
            case 'txt':
                iconClass = 'fas fa-file-alt';
                break;
            case 'zip':
                iconClass = 'fas fa-file-archive';
                break;
        }
        
        fileItem.innerHTML = `
            <div class="file-icon">
                <i class="${iconClass}"></i>
            </div>
            <div class="file-info">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)} • Processing...</span>
            </div>
            <div class="file-actions">
                <button class="btn-icon" title="Preview">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn-icon" title="Remove" onclick="this.closest('.file-item').remove()">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        fileList.appendChild(fileItem);
        
        // Simulate processing delay
        setTimeout(() => {
            const sizeSpan = fileItem.querySelector('.file-size');
            if (sizeSpan) {
                // Simulate record count based on file size
                const estimatedRecords = Math.floor(file.size / 100);
                sizeSpan.textContent = `${formatFileSize(file.size)} • ~${estimatedRecords.toLocaleString()} records`;
            }
        }, 1000);
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Accuracy slider functionality
function initializeAccuracySlider() {
    const slider = document.querySelector('.accuracy-slider .slider');
    const valueDisplay = document.querySelector('.accuracy-value');
    
    if (slider && valueDisplay) {
        slider.addEventListener('input', (e) => {
            valueDisplay.textContent = `${e.target.value}%`;
        });
    }
}

// Marketplace sharing toggle
function initializeMarketplaceSharing() {
    const shareCheckbox = document.getElementById('share-marketplace');
    const sharingDetails = document.querySelector('.sharing-details');
    
    if (shareCheckbox && sharingDetails) {
        shareCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                sharingDetails.style.display = 'block';
            } else {
                sharingDetails.style.display = 'none';
            }
        });
    }
}

// Distillation process controls
function initializeDistillationControls() {
    // Recent model actions
    const modelActions = document.querySelectorAll('.model-actions .btn-icon');
    
    modelActions.forEach(action => {
        action.addEventListener('click', (e) => {
            const title = action.getAttribute('title');
            const modelName = action.closest('.model-item').querySelector('h6').textContent;
            
            switch (title) {
                case 'Download':
                    simulateDownload(modelName);
                    break;
                case 'Share':
                    simulateShare(modelName);
                    break;
                case 'Deploy':
                    simulateDeploy(modelName);
                    break;
            }
        });
    });
}

function startDistillationProcess() {
    const workflowContainer = document.querySelector('.distillation-workflow');
    const progressContainer = document.querySelector('.distillation-progress');
    
    if (workflowContainer && progressContainer) {
        // Hide workflow and show progress
        workflowContainer.style.display = 'none';
        progressContainer.style.display = 'block';
        
        // Simulate progress updates
        simulateDistillationProgress();
    }
}

function simulateDistillationProgress() {
    const stages = document.querySelectorAll('.stage');
    const progressBars = document.querySelectorAll('.stage-progress .progress-fill');
    const statusTexts = document.querySelectorAll('.stage-progress span');
    const detailStats = document.querySelectorAll('.detail-stats .value');
    const timeRemaining = document.querySelector('.time-remaining span');
    
    let currentStage = 1; // Start with stage 2 (Data Processing is complete)
    
    function updateProgress() {
        // Update current stage progress
        if (currentStage < stages.length && progressBars[currentStage]) {
            const currentProgress = parseInt(progressBars[currentStage].style.width) || 0;
            const newProgress = Math.min(currentProgress + Math.random() * 10, 100);
            
            progressBars[currentStage].style.width = `${newProgress}%`;
            statusTexts[currentStage].textContent = `${Math.floor(newProgress)}% complete`;
            
            // Update detail stats with realistic values
            if (detailStats.length >= 4) {
                detailStats[0].textContent = (0.1 - newProgress * 0.001).toFixed(4); // Training Loss
                detailStats[1].textContent = `${(90 + newProgress * 0.05).toFixed(1)}%`; // Validation Accuracy
                detailStats[2].textContent = `${Math.floor(1200 - newProgress * 4)} MB`; // Model Size
                detailStats[3].textContent = `${Math.floor(80 + newProgress * 0.1)}%`; // Compression Ratio
            }
            
            // Update time remaining
            if (timeRemaining) {
                const remaining = Math.floor(30 - (newProgress / 100) * 30);
                timeRemaining.textContent = `${remaining} minutes remaining`;
            }
            
            // Complete current stage and move to next
            if (newProgress >= 100) {
                stages[currentStage].classList.add('active');
                statusTexts[currentStage].textContent = 'Complete';
                currentStage++;
                
                if (currentStage >= stages.length) {
                    // All stages complete
                    setTimeout(() => {
                        completeDistillation();
                    }, 2000);
                    return;
                }
            }
        }
        
        // Continue updating
        setTimeout(updateProgress, 1000 + Math.random() * 2000);
    }
    
    updateProgress();
}

function completeDistillation() {
    const progressContainer = document.querySelector('.distillation-progress');
    const recentModels = document.querySelector('.recent-models .models-list');
    
    if (progressContainer) {
        progressContainer.innerHTML = `
            <div class="progress-header">
                <h5><i class="fas fa-check-circle" style="color: #10b981;"></i> Distillation Complete!</h5>
                <p>Your specialized model has been successfully created and is ready for use.</p>
            </div>
            <div style="text-align: center; padding: 32px;">
                <div style="background: var(--bg-tertiary); border-radius: 12px; padding: 24px; margin-bottom: 24px;">
                    <h6 style="margin: 0 0 16px 0;">Model Performance Summary</h6>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px;">
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: 700; color: var(--accent-primary);">96.2%</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">ACCURACY</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: 700; color: var(--accent-primary);">423 MB</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">MODEL SIZE</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: 700; color: var(--accent-primary);">89%</div>
                            <div style="font-size: 12px; color: var(--text-secondary);">COMPRESSION</div>
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 12px; justify-content: center;">
                    <button class="btn primary" onclick="simulateDownload('Loan Risk Assessor v2.0')">
                        <i class="fas fa-download"></i> Download Model
                    </button>
                    <button class="btn secondary" onclick="location.reload()">
                        <i class="fas fa-plus"></i> Create Another Model
                    </button>
                </div>
            </div>
        `;
    }
    
    // Add new model to recent models list
    if (recentModels) {
        const newModel = document.createElement('div');
        newModel.className = 'model-item';
        newModel.style.border = '2px solid var(--accent-primary)';
        newModel.innerHTML = `
            <div class="model-icon">
                <i class="fas fa-certificate"></i>
            </div>
            <div class="model-info">
                <h6>Loan Risk Assessor v2.0 <span style="background: var(--accent-primary); color: var(--bg-primary); padding: 2px 8px; border-radius: 4px; font-size: 10px; margin-left: 8px;">NEW</span></h6>
                <p>Financial risk assessment model</p>
                <div class="model-meta">
                    <span>Created just now</span>
                    <span>•</span>
                    <span>96.2% accuracy</span>
                    <span>•</span>
                    <span>423 MB</span>
                </div>
            </div>
            <div class="model-actions">
                <button class="btn-icon" title="Download">
                    <i class="fas fa-download"></i>
                </button>
                <button class="btn-icon" title="Share">
                    <i class="fas fa-share"></i>
                </button>
                <button class="btn-icon" title="Deploy">
                    <i class="fas fa-rocket"></i>
                </button>
            </div>
        `;
        
        recentModels.insertBefore(newModel, recentModels.firstChild);
    }
}

// Utility functions for model actions
function simulateDownload(modelName) {
    showNotification(`Starting download of ${modelName}...`, 'success');
    
    // Simulate download progress
    setTimeout(() => {
        showNotification(`${modelName} download complete!`, 'success');
    }, 3000);
}

function simulateShare(modelName) {
    if (navigator.share) {
        navigator.share({
            title: `PRSM Model Lab - ${modelName}`,
            text: `Check out this custom model I created with PRSM Model Lab: ${modelName}`,
            url: window.location.href
        });
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(window.location.href).then(() => {
            showNotification(`Shareable link for ${modelName} copied to clipboard!`, 'success');
        });
    }
}

function simulateDeploy(modelName) {
    showNotification(`Deploying ${modelName} to cloud infrastructure...`, 'info');
    
    setTimeout(() => {
        showNotification(`${modelName} successfully deployed! Endpoint: https://api.prsm.dev/models/${modelName.toLowerCase().replace(/\s+/g, '-')}`, 'success');
    }, 5000);
}

// ===============================
// FTNS Budget & Cost Management
// ===============================

class FTNSBudgetManager {
    constructor() {
        this.currentBudget = 5000; // Default budget
        this.usedFTNS = 1500; // Current usage
        this.totalFTNSBalance = 42750; // User's total FTNS balance
        this.ftnsPrice = 0.0234; // Starting price in USD
        this.priceVariation = 0.001; // Price variation range
        
        this.initializeElements();
        this.setupEventListeners();
        this.startPriceUpdates();
        this.updateDisplay();
    }
    
    initializeElements() {
        this.ftnsProgress = document.getElementById('ftns-progress');
        this.ftnsDisplay = document.getElementById('ftns-display');
        this.ftnsCost = document.getElementById('ftns-cost');
        this.budgetModal = document.getElementById('budget-modal');
        this.adjustBudgetBtn = document.getElementById('adjust-budget-btn');
        this.customBudgetInput = document.getElementById('custom-budget-input');
        this.currentFtnsPrice = document.getElementById('current-ftns-price');
        this.budgetUsdValue = document.getElementById('budget-usd-value');
        this.totalFtnsBalanceElement = document.getElementById('total-ftns-balance');
        this.sessionFtnsUsed = document.getElementById('session-ftns-used');
        this.remainingAfterBudget = document.getElementById('remaining-after-budget');
        
        // Debug log to check if elements are found
        console.log('FTNS Elements initialized:', {
            ftnsProgress: !!this.ftnsProgress,
            ftnsDisplay: !!this.ftnsDisplay,
            ftnsCost: !!this.ftnsCost,
            budgetModal: !!this.budgetModal,
            adjustBudgetBtn: !!this.adjustBudgetBtn,
            totalFtnsBalanceElement: !!this.totalFtnsBalanceElement
        });
    }
    
    setupEventListeners() {
        // Budget adjustment button
        if (this.adjustBudgetBtn) {
            console.log('Adding click listener to budget button');
            this.adjustBudgetBtn.addEventListener('click', () => {
                console.log('Budget button clicked!');
                this.openBudgetModal();
            });
        } else {
            console.error('Budget adjustment button not found!');
        }
        
        // Modal close buttons
        const closeBudgetModal = document.getElementById('close-budget-modal');
        const cancelBudget = document.getElementById('cancel-budget');
        if (closeBudgetModal) closeBudgetModal.addEventListener('click', () => this.closeBudgetModal());
        if (cancelBudget) cancelBudget.addEventListener('click', () => this.closeBudgetModal());
        
        // Save budget button
        const saveBudget = document.getElementById('save-budget');
        if (saveBudget) saveBudget.addEventListener('click', () => this.saveBudget());
        
        // Preset budget buttons
        const presetButtons = document.querySelectorAll('.preset-btn');
        presetButtons.forEach(btn => {
            btn.addEventListener('click', () => this.selectPreset(btn));
        });
        
        // Custom input updates
        if (this.customBudgetInput) {
            this.customBudgetInput.addEventListener('input', () => this.updateCustomBudget());
        }
        
        // Close modal when clicking outside
        if (this.budgetModal) {
            this.budgetModal.addEventListener('click', (e) => {
                if (e.target === this.budgetModal) {
                    this.closeBudgetModal();
                }
            });
        }
    }
    
    openBudgetModal() {
        console.log('openBudgetModal called, modal element:', this.budgetModal);
        if (this.budgetModal) {
            console.log('Setting modal display to flex');
            this.budgetModal.style.display = 'flex';
            this.updateModalDisplay();
            
            // Set current budget in custom input
            if (this.customBudgetInput) {
                this.customBudgetInput.value = this.currentBudget;
            }
            
            // Set active preset button
            this.updateActivePreset();
            console.log('Modal should now be visible');
        } else {
            console.error('Budget modal element not found!');
        }
    }
    
    closeBudgetModal() {
        if (this.budgetModal) {
            this.budgetModal.style.display = 'none';
        }
    }
    
    selectPreset(button) {
        // Remove active class from all presets
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Update custom input
        const amount = parseInt(button.dataset.amount);
        if (this.customBudgetInput) {
            this.customBudgetInput.value = amount;
        }
        
        this.updateModalDisplay();
    }
    
    updateActivePreset() {
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
            if (parseInt(btn.dataset.amount) === this.currentBudget) {
                btn.classList.add('active');
            }
        });
    }
    
    updateCustomBudget() {
        // Remove active class from all presets if custom value doesn't match any
        const customValue = parseInt(this.customBudgetInput.value);
        let matchesPreset = false;
        
        document.querySelectorAll('.preset-btn').forEach(btn => {
            if (parseInt(btn.dataset.amount) === customValue) {
                btn.classList.add('active');
                matchesPreset = true;
            } else {
                btn.classList.remove('active');
            }
        });
        
        this.updateModalDisplay();
    }
    
    updateModalDisplay() {
        const budgetAmount = parseInt(this.customBudgetInput?.value) || this.currentBudget;
        
        // Update total FTNS balance
        if (this.totalFtnsBalanceElement) {
            this.totalFtnsBalanceElement.textContent = `${this.totalFTNSBalance.toLocaleString()} FTNS`;
        }
        
        // Update session FTNS used
        if (this.sessionFtnsUsed) {
            this.sessionFtnsUsed.textContent = `${this.usedFTNS.toLocaleString()} FTNS`;
        }
        
        // Calculate and update remaining balance after budget
        if (this.remainingAfterBudget) {
            const remainingBalance = this.totalFTNSBalance - budgetAmount;
            this.remainingAfterBudget.textContent = `${remainingBalance.toLocaleString()} FTNS`;
            
            // Add visual warning if budget exceeds balance
            if (remainingBalance < 0) {
                this.remainingAfterBudget.style.color = '#f87171';
                this.remainingAfterBudget.textContent = `${remainingBalance.toLocaleString()} FTNS (Insufficient Balance!)`;
            } else if (remainingBalance < 1000) {
                this.remainingAfterBudget.style.color = '#fbbf24';
            } else {
                this.remainingAfterBudget.style.color = 'var(--text-primary)';
            }
        }
        
        // Update price display
        if (this.currentFtnsPrice) {
            this.currentFtnsPrice.textContent = `$${this.ftnsPrice.toFixed(4)}`;
        }
        
        // Update USD value
        if (this.budgetUsdValue) {
            const usdValue = (budgetAmount * this.ftnsPrice).toFixed(2);
            this.budgetUsdValue.textContent = `$${usdValue}`;
        }
    }
    
    saveBudget() {
        const newBudget = parseInt(this.customBudgetInput?.value) || this.currentBudget;
        
        if (newBudget < 100) {
            alert('Budget must be at least 100 FTNS');
            return;
        }
        
        if (newBudget > this.totalFTNSBalance) {
            alert(`Budget cannot exceed your total balance of ${this.totalFTNSBalance.toLocaleString()} FTNS`);
            return;
        }
        
        if (newBudget > 100000) {
            alert('Budget cannot exceed 100,000 FTNS');
            return;
        }
        
        this.currentBudget = newBudget;
        this.updateDisplay();
        this.closeBudgetModal();
        
        // Show confirmation
        const remainingBalance = this.totalFTNSBalance - newBudget;
        showNotification(`FTNS budget updated to ${newBudget.toLocaleString()} FTNS. Remaining balance: ${remainingBalance.toLocaleString()} FTNS`, 'success');
    }
    
    updateDisplay() {
        // Update progress bar
        if (this.ftnsProgress) {
            this.ftnsProgress.max = this.currentBudget;
            this.ftnsProgress.value = this.usedFTNS;
        }
        
        // Update display text
        if (this.ftnsDisplay) {
            this.ftnsDisplay.textContent = `${this.usedFTNS.toLocaleString()} / ${this.currentBudget.toLocaleString()}`;
        }
        
        // Update cost
        this.updateCostDisplay();
    }
    
    updateCostDisplay() {
        if (this.ftnsCost) {
            const totalCost = (this.usedFTNS * this.ftnsPrice).toFixed(2);
            this.ftnsCost.textContent = `Cost: $${totalCost} USD`;
        }
    }
    
    simulateFTNSUsage(amount) {
        this.usedFTNS += amount;
        
        // Also reduce total balance as FTNS are spent
        this.totalFTNSBalance -= amount;
        
        if (this.usedFTNS > this.currentBudget) {
            this.usedFTNS = this.currentBudget;
            showNotification('FTNS budget limit reached!', 'warning');
        }
        
        // Check if user is running low on total balance
        if (this.totalFTNSBalance < 5000) {
            showNotification('Warning: Your FTNS balance is running low. Consider purchasing more FTNS.', 'warning');
        }
        
        this.updateDisplay();
    }
    
    startPriceUpdates() {
        // Simulate real-time FTNS price fluctuations
        setInterval(() => {
            // Create realistic price movement (small random walk)
            const change = (Math.random() - 0.5) * this.priceVariation * 2;
            this.ftnsPrice += change;
            
            // Keep price within reasonable bounds
            this.ftnsPrice = Math.max(0.01, Math.min(0.1, this.ftnsPrice));
            
            this.updateCostDisplay();
            
            // Update modal if open
            if (this.budgetModal && this.budgetModal.style.display === 'flex') {
                this.updateModalDisplay();
            }
        }, 5000); // Update every 5 seconds
    }
    
    // Public methods for integration with conversation features
    addFTNSUsage(tokens) {
        // Convert tokens to FTNS usage (approximately 1:1 ratio)
        this.simulateFTNSUsage(tokens);
    }
    
    getCurrentPrice() {
        return this.ftnsPrice;
    }
    
    getBudgetStatus() {
        return {
            used: this.usedFTNS,
            budget: this.currentBudget,
            remaining: this.currentBudget - this.usedFTNS,
            cost: this.usedFTNS * this.ftnsPrice,
            totalBalance: this.totalFTNSBalance,
            balanceAfterBudget: this.totalFTNSBalance - this.currentBudget,
            currentPrice: this.ftnsPrice
        };
    }
}

// Initialize FTNS Budget Manager when DOM is loaded
let ftnsManager;

document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for other initializations to complete
    setTimeout(() => {
        console.log('Initializing FTNS Budget Manager...');
        
        // Check if elements exist before initializing
        const adjustBtn = document.getElementById('adjust-budget-btn');
        const modal = document.getElementById('budget-modal');
        console.log('Pre-check elements:', {
            adjustBtn: !!adjustBtn,
            modal: !!modal
        });
        
        ftnsManager = new FTNSBudgetManager();
        console.log('FTNS Budget Manager initialized');
        
        // Simulate periodic FTNS usage for demo purposes
        setInterval(() => {
            if (Math.random() > 0.7) { // 30% chance every 10 seconds
                const usage = Math.floor(Math.random() * 50) + 10; // 10-59 FTNS
                ftnsManager.simulateFTNSUsage(usage);
            }
        }, 10000);
    }, 2000); // Increased wait time to 2 seconds
});

// Fallback initialization for budget modal (in case class initialization fails)
window.addEventListener('load', function() {
    setTimeout(() => {
        console.log('Fallback initialization for budget modal');
        
        const adjustBtn = document.getElementById('adjust-budget-btn');
        const modal = document.getElementById('budget-modal');
        
        if (adjustBtn && modal) {
            console.log('Setting up fallback budget modal');
            
            adjustBtn.addEventListener('click', function() {
                console.log('Fallback: Budget button clicked');
                modal.style.display = 'flex';
            });
            
            // Close button
            const closeBtn = document.getElementById('close-budget-modal');
            if (closeBtn) {
                closeBtn.addEventListener('click', function() {
                    modal.style.display = 'none';
                });
            }
            
            // Cancel button  
            const cancelBtn = document.getElementById('cancel-budget');
            if (cancelBtn) {
                cancelBtn.addEventListener('click', function() {
                    modal.style.display = 'none';
                });
            }
            
            // Click outside to close
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.style.display = 'none';
                }
            });
            
            console.log('Fallback budget modal setup complete');
        } else {
            console.error('Fallback: Could not find budget elements', {
                adjustBtn: !!adjustBtn,
                modal: !!modal
            });
        }
    }, 3000);
});

// ===============================
// FTNS Selling Functionality
// ===============================

class FTNSSellManager {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.updateDisplays();
    }
    
    initializeElements() {
        this.sellBalanceDisplay = document.getElementById('sell-balance-display');
        this.sellRateDisplay = document.getElementById('sell-rate-display');
        this.ftnsSellAmount = document.getElementById('ftns-sell-amount');
        this.sellCurrencySelect = document.getElementById('sell-currency-select');
        this.estimatedPayout = document.getElementById('estimated-payout');
        this.feeBreakdown = document.getElementById('fee-breakdown');
        this.netPayout = document.getElementById('net-payout');
        this.sellBtn = document.getElementById('sell-ftns-btn');
        
        // Crypto exchange rates (mock data)
        this.exchangeRates = {
            usd: 1,
            btc: 0.000028,  // Example: 1 USD = 0.000028 BTC
            eth: 0.0004,    // Example: 1 USD = 0.0004 ETH
            usdc: 0.999     // Example: 1 USD = 0.999 USDC
        };
        
        // Fee structures
        this.fees = {
            usd: { percentage: 0.025, fixed: 0.50 },    // 2.5% + $0.50
            btc: { percentage: 0.015, fixed: 0 },       // 1.5%
            eth: { percentage: 0.015, fixed: 0 },       // 1.5% 
            usdc: { percentage: 0.015, fixed: 0 }       // 1.5%
        };
    }
    
    setupEventListeners() {
        if (this.ftnsSellAmount) {
            this.ftnsSellAmount.addEventListener('input', () => this.calculatePayout());
        }
        
        if (this.sellCurrencySelect) {
            this.sellCurrencySelect.addEventListener('change', () => this.calculatePayout());
        }
        
        if (this.sellBtn) {
            this.sellBtn.addEventListener('click', () => this.processSell());
        }
        
        // Payout option selection
        document.querySelectorAll('.payout-option').forEach(btn => {
            btn.addEventListener('click', (e) => this.selectPayoutOption(e.target));
        });
    }
    
    updateDisplays() {
        // Update balance display
        if (this.sellBalanceDisplay && ftnsManager) {
            const balance = ftnsManager.totalFTNSBalance;
            this.sellBalanceDisplay.textContent = `${balance.toLocaleString()} FTNS`;
            
            // Update max attribute on input
            if (this.ftnsSellAmount) {
                this.ftnsSellAmount.max = balance;
            }
        }
        
        // Update rate display
        if (this.sellRateDisplay && ftnsManager) {
            const rate = ftnsManager.ftnsPrice;
            this.sellRateDisplay.textContent = `1 FTNS = $${rate.toFixed(4)} USD`;
        }
    }
    
    calculatePayout() {
        const ftnsAmount = parseFloat(this.ftnsSellAmount?.value) || 0;
        const currency = this.sellCurrencySelect?.value || 'usd';
        
        if (ftnsAmount <= 0) {
            this.resetCalculator();
            return;
        }
        
        // Check if amount exceeds balance
        const maxBalance = ftnsManager?.totalFTNSBalance || 42750;
        if (ftnsAmount > maxBalance) {
            this.ftnsSellAmount.value = maxBalance;
            return this.calculatePayout();
        }
        
        // Get current FTNS price
        const ftnsPrice = ftnsManager?.ftnsPrice || 0.0234;
        
        // Calculate gross payout in USD
        const grossUSD = ftnsAmount * ftnsPrice;
        
        // Calculate fees
        const feeStructure = this.fees[currency];
        const percentageFee = grossUSD * feeStructure.percentage;
        const totalFeeUSD = percentageFee + feeStructure.fixed;
        
        // Calculate net payout in USD
        const netUSD = grossUSD - totalFeeUSD;
        
        // Convert to target currency
        const exchangeRate = this.exchangeRates[currency];
        const grossPayout = grossUSD * exchangeRate;
        const netPayout = netUSD * exchangeRate;
        const totalFee = totalFeeUSD * exchangeRate;
        
        // Update displays
        this.updatePayoutDisplays(currency, grossPayout, totalFee, netPayout);
        
        // Enable/disable sell button
        if (this.sellBtn) {
            this.sellBtn.disabled = ftnsAmount <= 0 || ftnsAmount > maxBalance;
        }
    }
    
    updatePayoutDisplays(currency, gross, fee, net) {
        const symbols = {
            usd: '$',
            btc: '₿',
            eth: 'Ξ',
            usdc: '$'
        };
        
        const decimals = {
            usd: 2,
            btc: 8,
            eth: 6,
            usdc: 2
        };
        
        const symbol = symbols[currency];
        const decimal = decimals[currency];
        
        if (this.estimatedPayout) {
            this.estimatedPayout.textContent = `${symbol}${gross.toFixed(decimal)}`;
        }
        
        if (this.feeBreakdown) {
            this.feeBreakdown.textContent = `Fee: ${symbol}${fee.toFixed(decimal)}`;
        }
        
        if (this.netPayout) {
            this.netPayout.textContent = `Net: ${symbol}${net.toFixed(decimal)}`;
        }
    }
    
    resetCalculator() {
        if (this.estimatedPayout) this.estimatedPayout.textContent = '$0.00';
        if (this.feeBreakdown) this.feeBreakdown.textContent = 'Fee: $0.00';
        if (this.netPayout) this.netPayout.textContent = 'Net: $0.00';
        if (this.sellBtn) this.sellBtn.disabled = true;
    }
    
    selectPayoutOption(button) {
        // Remove active class from siblings
        const siblings = button.parentElement.querySelectorAll('.payout-option');
        siblings.forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        button.classList.add('active');
    }
    
    processSell() {
        const ftnsAmount = parseFloat(this.ftnsSellAmount?.value) || 0;
        const currency = this.sellCurrencySelect?.value || 'usd';
        
        if (ftnsAmount <= 0) {
            alert('Please enter a valid FTNS amount to sell');
            return;
        }
        
        const maxBalance = ftnsManager?.totalFTNSBalance || 42750;
        if (ftnsAmount > maxBalance) {
            alert(`You cannot sell more than your available balance of ${maxBalance.toLocaleString()} FTNS`);
            return;
        }
        
        // Check if payout method is selected
        const activePayoutOption = document.querySelector('.payout-option.active');
        if (!activePayoutOption) {
            alert('Please select a payout method');
            return;
        }
        
        // Calculate final amounts
        const ftnsPrice = ftnsManager?.ftnsPrice || 0.0234;
        const grossUSD = ftnsAmount * ftnsPrice;
        const feeStructure = this.fees[currency];
        const totalFeeUSD = (grossUSD * feeStructure.percentage) + feeStructure.fixed;
        const netUSD = grossUSD - totalFeeUSD;
        
        const exchangeRate = this.exchangeRates[currency];
        const netPayout = netUSD * exchangeRate;
        
        const symbols = { usd: '$', btc: '₿', eth: 'Ξ', usdc: '$' };
        const decimals = { usd: 2, btc: 8, eth: 6, usdc: 2 };
        
        const symbol = symbols[currency];
        const decimal = decimals[currency];
        
        // Confirmation dialog
        const confirmed = confirm(
            `Confirm FTNS Sale:\n\n` +
            `Amount: ${ftnsAmount.toLocaleString()} FTNS\n` +
            `Payout: ${symbol}${netPayout.toFixed(decimal)} ${currency.toUpperCase()}\n` +
            `Method: ${activePayoutOption.textContent}\n\n` +
            `Proceed with sale?`
        );
        
        if (confirmed) {
            this.executeSell(ftnsAmount, netPayout, currency, activePayoutOption.textContent);
        }
    }
    
    executeSell(ftnsAmount, netPayout, currency, payoutMethod) {
        // Update FTNS balance
        if (ftnsManager) {
            ftnsManager.totalFTNSBalance -= ftnsAmount;
            ftnsManager.updateDisplay();
        }
        
        // Update sell display
        this.updateDisplays();
        
        // Reset form
        if (this.ftnsSellAmount) this.ftnsSellAmount.value = '';
        this.resetCalculator();
        
        // Remove active payout selection
        document.querySelectorAll('.payout-option.active').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show success notification
        const symbols = { usd: '$', btc: '₿', eth: 'Ξ', usdc: '$' };
        const decimals = { usd: 2, btc: 8, eth: 6, usdc: 2 };
        const symbol = symbols[currency];
        const decimal = decimals[currency];
        
        showNotification(
            `Successfully sold ${ftnsAmount.toLocaleString()} FTNS for ${symbol}${netPayout.toFixed(decimal)} via ${payoutMethod}. Processing time: 1-3 business days.`,
            'success'
        );
    }
    
    // Public method to update displays when FTNS price changes
    updateFromPriceChange() {
        this.updateDisplays();
        if (this.ftnsSellAmount?.value) {
            this.calculatePayout();
        }
    }
}

// Initialize FTNS Sell Manager
let ftnsSellManager;

document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        ftnsSellManager = new FTNSSellManager();
        console.log('FTNS Sell Manager initialized');
        
        // Update sell displays when FTNS price changes
        if (ftnsManager) {
            const originalUpdateCostDisplay = ftnsManager.updateCostDisplay;
            ftnsManager.updateCostDisplay = function() {
                originalUpdateCostDisplay.call(this);
                if (ftnsSellManager) {
                    ftnsSellManager.updateFromPriceChange();
                }
            };
        }
    }, 2500);
});

// ========================================
// DATA WORK HUB FUNCTIONALITY
// ========================================

function initializeDataWorkFunctionality() {
    /**
     * Initialize all data work related functionality
     */
    console.log('Initializing data work functionality...');
    initializeDataWorkToggle();
    initializeDataWorkButtons();
    initializeJobApplications();
    initializeWorkerTools();
    addOnboardingTrigger(); // Add the Get Started button
    console.log('Data work functionality initialized.');
}

function showMainMarketplace() {
    /**
     * Return to the main marketplace from the data work hub
     */
    const dataWorkHub = document.getElementById('data-work-hub');
    const otherSections = document.querySelectorAll('.marketplace-section:not(.data-work-hub)');
    const assetTypeCards = document.querySelectorAll('.asset-type-card');
    
    // Add fade out animation to data work hub
    if (dataWorkHub) {
        dataWorkHub.style.opacity = '0';
        dataWorkHub.style.transform = 'translateY(-10px)';
        
        setTimeout(() => {
            dataWorkHub.style.display = 'none';
            dataWorkHub.style.opacity = '1';
            dataWorkHub.style.transform = 'translateY(0)';
        }, 200);
    }
    
    // Show other marketplace sections with fade in
    otherSections.forEach(section => {
        section.style.display = 'block';
        section.style.opacity = '0';
        section.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
        }, 100);
    });
    
    // Reset active state - make AI Models active by default
    assetTypeCards.forEach(card => card.classList.remove('active'));
    const aiModelCard = document.querySelector('.asset-type-card[data-type="ai_model"]');
    if (aiModelCard) {
        aiModelCard.classList.add('active');
    }
    
    // Filter assets to show AI models
    filterAssetCards('ai_model');
    
    // Update discovery title
    if (aiModelCard) {
        updateMarketplaceStats('ai_model', aiModelCard);
    }
    
    console.log('Returned to main marketplace');
}

function showDataWorkHub() {
    /**
     * Navigate to the Data Work Hub from any section
     */
    console.log('Navigating to Data Work Hub...');
    
    // First, ensure we're on the marketplace tab
    const marketplaceTab = document.querySelector('[data-target="marketplace-content"]');
    if (marketplaceTab) {
        marketplaceTab.click();
    }
    
    // Wait a bit for tab switching, then show Data Work Hub
    setTimeout(() => {
        const dataWorkHub = document.getElementById('data-work-hub');
        const otherSections = document.querySelectorAll('.marketplace-section:not(.data-work-hub)');
        const assetTypeCards = document.querySelectorAll('.asset-type-card');
        const dataWorkCard = document.querySelector('.asset-type-card[data-type="data_work"]');
        
        console.log('Data Work Hub element:', dataWorkHub ? 'found' : 'not found');
        console.log('Other sections found:', otherSections.length);
        
        // Hide other marketplace sections with fade out
        otherSections.forEach(section => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(-10px)';
            
            setTimeout(() => {
                section.style.display = 'none';
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
            }, 200);
        });
        
        // Show data work hub with fade in
        if (dataWorkHub) {
            console.log('Showing Data Work Hub...');
            setTimeout(() => {
                dataWorkHub.style.display = 'block';
                dataWorkHub.style.opacity = '0';
                dataWorkHub.style.transform = 'translateY(10px)';
                
                setTimeout(() => {
                    dataWorkHub.style.opacity = '1';
                    dataWorkHub.style.transform = 'translateY(0)';
                    console.log('Data Work Hub should now be visible');
                }, 100);
            }, 100);
        } else {
            console.error('Data Work Hub element not found!');
        }
        
        // Update active state
        assetTypeCards.forEach(card => card.classList.remove('active'));
        if (dataWorkCard) {
            dataWorkCard.classList.add('active');
        }
        
        console.log('Data Work Hub navigation completed');
    }, 150);
}

function initializeDataWorkToggle() {
    /**
     * Handle showing/hiding data work hub when data work asset type is selected
     */
    const dataWorkCard = document.querySelector('.asset-type-card[data-type="data_work"]');
    const dataWorkHub = document.getElementById('data-work-hub');
    const assetTypeCards = document.querySelectorAll('.asset-type-card');
    const backToMarketplaceBtn = document.getElementById('back-to-marketplace-btn');
    
    console.log('Data work card:', dataWorkCard ? 'found' : 'not found');
    console.log('Data work hub:', dataWorkHub ? 'found' : 'not found');
    console.log('Back to marketplace button:', backToMarketplaceBtn ? 'found' : 'not found');
    
    if (!dataWorkCard || !dataWorkHub) return;
    
    // Add click handler for back to marketplace button
    if (backToMarketplaceBtn) {
        backToMarketplaceBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Back to marketplace clicked');
            showMainMarketplace();
        });
    }
    
    // Handle data work card click (shows filter/discovery)
    dataWorkCard.addEventListener('click', (e) => {
        e.stopPropagation();
        console.log('Data work card clicked - showing discovery with data work filter');
        
        // Ensure data work hub is hidden
        dataWorkHub.style.display = 'none';
        
        // Show other marketplace sections
        const otherSections = document.querySelectorAll('.marketplace-section:not(.data-work-hub)');
        otherSections.forEach(section => {
            section.style.display = 'block';
        });
        
        // Update active state
        assetTypeCards.forEach(card => card.classList.remove('active'));
        dataWorkCard.classList.add('active');
        
        // Update discovery title and show data work assets
        updateMarketplaceStats('data_work', dataWorkCard);
        showAssetsByType('data_work');
    });
    
    // Handle switching back to other asset types
    assetTypeCards.forEach(card => {
        if (card === dataWorkCard) return;
        
        card.addEventListener('click', () => {
            dataWorkHub.style.display = 'none';
            const otherSections = document.querySelectorAll('.marketplace-section:not(.data-work-hub)');
            otherSections.forEach(section => {
                section.style.display = 'block';
            });
        });
    });
}

function initializeDataWorkButtons() {
    /**
     * Initialize data work quick action buttons
     */
    // Use more specific selectors instead of :has() for better browser compatibility
    const quickActionBtns = document.querySelectorAll('.quick-action-btn');
    let postJobBtn, findWorkBtn;
    
    quickActionBtns.forEach(btn => {
        const icon = btn.querySelector('i');
        if (icon && icon.classList.contains('fa-briefcase')) {
            postJobBtn = btn;
        } else if (icon && icon.classList.contains('fa-user-friends')) {
            findWorkBtn = btn;
        }
    });
    
    if (postJobBtn) {
        postJobBtn.addEventListener('click', () => {
            showJobPostingModal();
        });
    }
    
    if (findWorkBtn) {
        findWorkBtn.addEventListener('click', () => {
            // Show filtered marketplace view with data work jobs
            const dataWorkCard = document.querySelector('.asset-type-card[data-type="data_work"]');
            if (dataWorkCard) {
                dataWorkCard.click();
            }
        });
    }
}

function initializeJobApplications() {
    /**
     * Handle job application buttons
     */
    const applyButtons = document.querySelectorAll('.apply-btn');
    
    applyButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const jobCard = e.target.closest('.job-card');
            const jobTitle = jobCard.querySelector('.job-title').textContent;
            const jobPayment = jobCard.querySelector('.job-payment').textContent;
            
            showJobApplicationModal(jobTitle, jobPayment);
        });
    });
}

function initializeWorkerTools() {
    /**
     * Initialize worker tool buttons
     */
    const toolButtons = document.querySelectorAll('.tool-btn');
    
    toolButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const toolCard = e.target.closest('.tool-card');
            const toolTitle = toolCard.querySelector('h6').textContent;
            
            switch(toolTitle) {
                case 'SMS Job Alerts':
                    showSMSAlertsModal();
                    break;
                case 'Auto Translation':
                    showTranslationModal();
                    break;
                case 'Currency Converter':
                    showCurrencyModal();
                    break;
                case 'Payment Protection':
                    showPaymentProtectionModal();
                    break;
            }
        });
    });
}

function showJobPostingModal() {
    /**
     * Show modal for posting a new data work job
     */
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content data-work-modal">
            <div class="modal-header">
                <h3><i class="fas fa-briefcase"></i> Post Data Work Job</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <form class="job-posting-form">
                    <div class="form-group">
                        <label for="job-title">Job Title</label>
                        <input type="text" id="job-title" placeholder="e.g., Medical Image Annotation" required>
                    </div>
                    <div class="form-group">
                        <label for="job-type">Job Type</label>
                        <select id="job-type" required>
                            <option value="image_annotation">Image Annotation</option>
                            <option value="text_labeling">Text Labeling</option>
                            <option value="audio_transcription">Audio Transcription</option>
                            <option value="data_cleaning">Data Cleaning</option>
                            <option value="content_moderation">Content Moderation</option>
                            <option value="survey_responses">Survey Responses</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="job-quantity">Quantity</label>
                            <input type="number" id="job-quantity" placeholder="1000" required>
                        </div>
                        <div class="form-group">
                            <label for="job-deadline">Deadline</label>
                            <input type="number" id="job-deadline" placeholder="5" required>
                            <small>Days from now</small>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="job-description">Description</label>
                        <textarea id="job-description" rows="4" placeholder="Detailed description of the work required..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="job-payment">Payment (FTNS)</label>
                        <input type="number" id="job-payment" placeholder="2500" required>
                        <div class="currency-preview">
                            <span class="currency-amount">$0.00 USD</span>
                            <span class="currency-amount">₦0 NGN</span>
                            <span class="currency-amount">₹0 INR</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="job-requirements">Requirements</label>
                        <textarea id="job-requirements" rows="3" placeholder="Special requirements, qualifications, or preferences..."></textarea>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal(this)">Cancel</button>
                <button class="btn btn-primary" onclick="submitJobPosting()">Post Job</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Add payment calculation
    const paymentInput = document.getElementById('job-payment');
    paymentInput.addEventListener('input', updateJobPaymentPreview);
    
    // Add modal close handlers
    addModalCloseHandlers(modal);
}

function showJobApplicationModal(jobTitle, jobPayment) {
    /**
     * Show modal for applying to a data work job
     */
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content data-work-modal">
            <div class="modal-header">
                <h3><i class="fas fa-user-edit"></i> Apply for Job</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div class="job-summary">
                    <h4>${jobTitle}</h4>
                    <div class="payment-info">
                        <span class="payment-amount">${jobPayment}</span>
                        <span class="payment-conversion">≈ $67.25 USD</span>
                    </div>
                </div>
                <form class="job-application-form">
                    <div class="form-group">
                        <label for="applicant-experience">Relevant Experience</label>
                        <textarea id="applicant-experience" rows="4" placeholder="Describe your relevant experience for this type of work..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="applicant-availability">Availability</label>
                        <select id="applicant-availability" required>
                            <option value="immediate">Available immediately</option>
                            <option value="1-day">Within 1 day</option>
                            <option value="2-days">Within 2 days</option>
                            <option value="1-week">Within 1 week</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="applicant-questions">Questions or Comments</label>
                        <textarea id="applicant-questions" rows="3" placeholder="Any questions about the job or additional information you'd like to provide..."></textarea>
                    </div>
                    <div class="form-group">
                        <label class="checkbox-group">
                            <input type="checkbox" id="terms-agreement" required>
                            <span>I agree to the PRSM Data Work Terms and Payment Protection policies</span>
                        </label>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal(this)">Cancel</button>
                <button class="btn btn-primary" onclick="submitJobApplication()">Submit Application</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    addModalCloseHandlers(modal);
}

function showSMSAlertsModal() {
    /**
     * Show modal for setting up SMS job alerts
     */
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content data-work-modal">
            <div class="modal-header">
                <h3><i class="fas fa-bell"></i> SMS Job Alerts</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <p>Get notified of new data work jobs that match your criteria via SMS.</p>
                <form class="sms-alerts-form">
                    <div class="form-group">
                        <label for="phone-number">Phone Number</label>
                        <input type="tel" id="phone-number" placeholder="+1-555-123-4567" required>
                    </div>
                    <div class="form-group">
                        <label>Job Types (select all that apply)</label>
                        <div class="checkbox-grid">
                            <label class="checkbox-option">
                                <input type="checkbox" value="image_annotation">
                                <span>Image Annotation</span>
                            </label>
                            <label class="checkbox-option">
                                <input type="checkbox" value="text_labeling">
                                <span>Text Labeling</span>
                            </label>
                            <label class="checkbox-option">
                                <input type="checkbox" value="audio_transcription">
                                <span>Audio Transcription</span>
                            </label>
                            <label class="checkbox-option">
                                <input type="checkbox" value="data_cleaning">
                                <span>Data Cleaning</span>
                            </label>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="min-payment">Minimum Payment (FTNS)</label>
                            <input type="number" id="min-payment" placeholder="1000">
                        </div>
                        <div class="form-group">
                            <label for="max-alerts">Max Alerts per Day</label>
                            <input type="number" id="max-alerts" placeholder="5" max="10">
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="quiet-hours">Quiet Hours</label>
                        <div class="time-range">
                            <input type="time" id="quiet-start" value="22:00">
                            <span>to</span>
                            <input type="time" id="quiet-end" value="06:00">
                        </div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal(this)">Cancel</button>
                <button class="btn btn-primary" onclick="saveSMSAlerts()">Save Alerts</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    addModalCloseHandlers(modal);
}

function showTranslationModal() {
    /**
     * Show modal for configuring auto-translation settings
     */
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content data-work-modal">
            <div class="modal-header">
                <h3><i class="fas fa-language"></i> Auto Translation Settings</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <p>Configure automatic translation of job postings to your preferred language.</p>
                <form class="translation-form">
                    <div class="form-group">
                        <label for="primary-language">Primary Language</label>
                        <select id="primary-language" required>
                            <option value="en">English</option>
                            <option value="es">Spanish</option>
                            <option value="fr">French</option>
                            <option value="de">German</option>
                            <option value="zh">Chinese (Mandarin)</option>
                            <option value="hi">Hindi</option>
                            <option value="ar">Arabic</option>
                            <option value="pt">Portuguese</option>
                            <option value="ru">Russian</option>
                            <option value="ja">Japanese</option>
                            <option value="yo">Yoruba</option>
                            <option value="sw">Swahili</option>
                            <option value="ha">Hausa</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="secondary-language">Secondary Language (Optional)</label>
                        <select id="secondary-language">
                            <option value="">None</option>
                            <option value="en">English</option>
                            <option value="es">Spanish</option>
                            <option value="fr">French</option>
                            <option value="de">German</option>
                            <option value="zh">Chinese (Mandarin)</option>
                            <option value="hi">Hindi</option>
                            <option value="ar">Arabic</option>
                            <option value="pt">Portuguese</option>
                            <option value="ru">Russian</option>
                            <option value="ja">Japanese</option>
                            <option value="yo">Yoruba</option>
                            <option value="sw">Swahili</option>
                            <option value="ha">Hausa</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="checkbox-group">
                            <input type="checkbox" id="auto-translate" checked>
                            <span>Automatically translate all job postings</span>
                        </label>
                        <label class="checkbox-group">
                            <input type="checkbox" id="show-original">
                            <span>Always show original text alongside translation</span>
                        </label>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal(this)">Cancel</button>
                <button class="btn btn-primary" onclick="saveTranslationSettings()">Save Settings</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    addModalCloseHandlers(modal);
}

function showCurrencyModal() {
    /**
     * Show modal for setting preferred currency
     */
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content data-work-modal">
            <div class="modal-header">
                <h3><i class="fas fa-exchange-alt"></i> Currency Preferences</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <p>Set your preferred currency for displaying job payments and earnings.</p>
                <form class="currency-form">
                    <div class="form-group">
                        <label for="primary-currency">Primary Currency</label>
                        <select id="primary-currency" required>
                            <option value="USD">🇺🇸 US Dollar (USD)</option>
                            <option value="NGN">🇳🇬 Nigerian Naira (NGN)</option>
                            <option value="INR">🇮🇳 Indian Rupee (INR)</option>
                            <option value="KES">🇰🇪 Kenyan Shilling (KES)</option>
                            <option value="GHS">🇬🇭 Ghanaian Cedi (GHS)</option>
                            <option value="ZAR">🇿🇦 South African Rand (ZAR)</option>
                            <option value="EGP">🇪🇬 Egyptian Pound (EGP)</option>
                            <option value="BRL">🇧🇷 Brazilian Real (BRL)</option>
                            <option value="PHP">🇵🇭 Philippine Peso (PHP)</option>
                            <option value="IDR">🇮🇩 Indonesian Rupiah (IDR)</option>
                            <option value="EUR">🇪🇺 Euro (EUR)</option>
                            <option value="GBP">🇬🇧 British Pound (GBP)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="location">Your Location</label>
                        <input type="text" id="location" placeholder="e.g., Lagos, Nigeria" required>
                        <small>Used to calculate local purchasing power and fair wages</small>
                    </div>
                    <div class="currency-preview">
                        <h6>Currency Preview</h6>
                        <div class="preview-item">
                            <span>Sample Job Payment: 2,500 FTNS</span>
                            <span class="converted-amount">≈ $48.00 USD</span>
                        </div>
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal(this)">Cancel</button>
                <button class="btn btn-primary" onclick="saveCurrencySettings()">Save Settings</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    addModalCloseHandlers(modal);
}

function showPaymentProtectionModal() {
    /**
     * Show information about payment protection
     */
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content data-work-modal">
            <div class="modal-header">
                <h3><i class="fas fa-shield-alt"></i> Payment Protection</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div class="protection-info">
                    <div class="protection-feature">
                        <i class="fas fa-lock"></i>
                        <div>
                            <h6>Escrow System</h6>
                            <p>Payments are held in secure FTNS escrow until work is completed and approved.</p>
                        </div>
                    </div>
                    <div class="protection-feature">
                        <i class="fas fa-gavel"></i>
                        <div>
                            <h6>Dispute Resolution</h6>
                            <p>PRSM governance system provides fair dispute resolution for payment issues.</p>
                        </div>
                    </div>
                    <div class="protection-feature">
                        <i class="fas fa-chart-line"></i>
                        <div>
                            <h6>Token Appreciation</h6>
                            <p>FTNS tokens may increase in value as PRSM adoption grows globally.</p>
                        </div>
                    </div>
                    <div class="protection-feature">
                        <i class="fas fa-globe"></i>
                        <div>
                            <h6>Global Access</h6>
                            <p>Convert FTNS to any major currency through integrated exchange systems.</p>
                        </div>
                    </div>
                </div>
                <div class="protection-stats">
                    <h6>Protection Statistics</h6>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <span class="stat-number">99.8%</span>
                            <span class="stat-label">Payment Success Rate</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">2.4 days</span>
                            <span class="stat-label">Average Resolution Time</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">$2.4M</span>
                            <span class="stat-label">Protected Payments</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-primary" onclick="closeModal(this)">Got It</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    addModalCloseHandlers(modal);
}

// Utility functions for data work
function updateJobPaymentPreview() {
    const paymentInput = document.getElementById('job-payment');
    const currencyPreviews = document.querySelectorAll('.currency-amount');
    
    if (!paymentInput || currencyPreviews.length === 0) return;
    
    const ftnsAmount = parseFloat(paymentInput.value) || 0;
    const ftnsRate = 0.0192; // $0.0192 per FTNS
    
    const usdAmount = ftnsAmount * ftnsRate;
    const ngnAmount = usdAmount * 817; // Approximate NGN rate
    const inrAmount = usdAmount * 84; // Approximate INR rate
    
    currencyPreviews[0].textContent = `$${usdAmount.toFixed(2)} USD`;
    currencyPreviews[1].textContent = `₦${ngnAmount.toLocaleString()} NGN`;
    currencyPreviews[2].textContent = `₹${inrAmount.toLocaleString()} INR`;
}

function addModalCloseHandlers(modal) {
    const closeBtn = modal.querySelector('.modal-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => closeModal(closeBtn));
    }
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal(modal);
        }
    });
}

function closeModal(element) {
    const modal = element.closest('.modal-overlay') || element;
    if (modal) {
        modal.remove();
    }
}

function submitJobPosting() {
    console.log('Job posting submitted');
    showActionFeedback('posted', 'New data work job');
    closeModal(document.querySelector('.modal-overlay'));
}

function submitJobApplication() {
    console.log('Job application submitted');
    showActionFeedback('applied', 'Job application');
    closeModal(document.querySelector('.modal-overlay'));
}

function saveSMSAlerts() {
    console.log('SMS alerts saved');
    showActionFeedback('configured', 'SMS job alerts');
    closeModal(document.querySelector('.modal-overlay'));
}

function saveTranslationSettings() {
    console.log('Translation settings saved');
    showActionFeedback('configured', 'Auto-translation');
    closeModal(document.querySelector('.modal-overlay'));
}

function saveCurrencySettings() {
    console.log('Currency settings saved');
    showActionFeedback('configured', 'Currency preferences');
    closeModal(document.querySelector('.modal-overlay'));
}

// === USER ONBOARDING FUNCTIONALITY ===

// Global onboarding state
let onboardingCurrentStep = 1;
let onboardingData = {
    language: {},
    location: {},
    work: {},
    notifications: {}
};

// Initialize onboarding functionality
function initializeOnboarding() {
    const onboardingModal = document.getElementById('user-onboarding-modal');
    const nextBtn = document.getElementById('onboarding-next-btn');
    const backBtn = document.getElementById('onboarding-back-btn');
    const finishBtn = document.getElementById('onboarding-finish-btn');
    
    if (nextBtn) {
        nextBtn.addEventListener('click', handleOnboardingNext);
    }
    
    if (backBtn) {
        backBtn.addEventListener('click', handleOnboardingBack);
    }
    
    if (finishBtn) {
        finishBtn.addEventListener('click', handleOnboardingFinish);
    }
    
    // Check if user needs onboarding on data work access
    checkOnboardingStatus();
}

// Check if user needs onboarding
function checkOnboardingStatus() {
    // Always add the appropriate button (onboarding trigger or get started)
    addOnboardingTrigger();
}

// Add onboarding trigger button to data work card
function addOnboardingTrigger() {
    const dataWorkCard = document.querySelector('[data-type="data_work"]');
    if (dataWorkCard) {
        // Remove any existing buttons first
        const existingTrigger = dataWorkCard.querySelector('.trigger-onboarding-btn');
        const existingGetStarted = dataWorkCard.querySelector('.get-started-btn');
        if (existingTrigger) existingTrigger.remove();
        if (existingGetStarted) existingGetStarted.remove();
        
        // Always show "Get Started" button that navigates to Data Work Hub
        // The onboarding can be triggered from within the Data Work Hub if needed
        const getStartedBtn = document.createElement('button');
        getStartedBtn.className = 'get-started-btn';
        getStartedBtn.innerHTML = '<i class="fas fa-arrow-right"></i> Get Started';
        getStartedBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            console.log('Get Started button clicked - navigating to Data Work Hub');
            showDataWorkHub();
        });
        dataWorkCard.appendChild(getStartedBtn);
        
        console.log('Get Started button added to Data Work card');
    }
}

// Start the onboarding process
function startOnboarding() {
    const modal = document.getElementById('user-onboarding-modal');
    if (modal) {
        modal.style.display = 'flex';
        onboardingCurrentStep = 1;
        updateOnboardingStep();
        resetOnboardingData();
    }
}

// Reset onboarding data
function resetOnboardingData() {
    onboardingData = {
        language: {},
        location: {},
        work: {},
        notifications: {}
    };
}

// Handle next button click
function handleOnboardingNext() {
    if (validateCurrentStep()) {
        saveCurrentStepData();
        if (onboardingCurrentStep < 4) {
            onboardingCurrentStep++;
            updateOnboardingStep();
        }
    }
}

// Handle back button click
function handleOnboardingBack() {
    if (onboardingCurrentStep > 1) {
        onboardingCurrentStep--;
        updateOnboardingStep();
    }
}

// Handle finish button click
function handleOnboardingFinish() {
    if (validateCurrentStep()) {
        saveCurrentStepData();
        completeOnboarding();
    }
}

// Update the onboarding step display
function updateOnboardingStep() {
    // Hide all steps
    for (let i = 1; i <= 4; i++) {
        const step = document.getElementById(`onboarding-step-${i}`);
        if (step) {
            step.style.display = i === onboardingCurrentStep ? 'block' : 'none';
        }
        
        // Update progress indicators
        const indicator = document.getElementById(`step-${i}-indicator`);
        if (indicator) {
            indicator.classList.toggle('active', i === onboardingCurrentStep);
            indicator.classList.toggle('completed', i < onboardingCurrentStep);
        }
    }
    
    // Update progress lines
    const progressLines = document.querySelectorAll('.progress-line');
    progressLines.forEach((line, index) => {
        line.classList.toggle('completed', index < onboardingCurrentStep - 1);
    });
    
    // Update button visibility
    const nextBtn = document.getElementById('onboarding-next-btn');
    const backBtn = document.getElementById('onboarding-back-btn');
    const finishBtn = document.getElementById('onboarding-finish-btn');
    
    if (nextBtn) {
        nextBtn.style.display = onboardingCurrentStep < 4 ? 'block' : 'none';
    }
    
    if (backBtn) {
        backBtn.style.display = onboardingCurrentStep > 1 ? 'block' : 'none';
    }
    
    if (finishBtn) {
        finishBtn.style.display = onboardingCurrentStep === 4 ? 'block' : 'none';
    }
}

// Validate current step
function validateCurrentStep() {
    const currentStepElement = document.getElementById(`onboarding-step-${onboardingCurrentStep}`);
    if (!currentStepElement) return false;
    
    const requiredFields = currentStepElement.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.style.borderColor = 'var(--prsm-error)';
            isValid = false;
        } else {
            field.style.borderColor = 'var(--border-color)';
        }
    });
    
    // Special validation for work categories (step 3)
    if (onboardingCurrentStep === 3) {
        const workCategories = currentStepElement.querySelectorAll('.work-categories input[type="checkbox"]:checked');
        if (workCategories.length === 0) {
            alert('Please select at least one work category.');
            isValid = false;
        }
    }
    
    return isValid;
}

// Save current step data
function saveCurrentStepData() {
    const currentStepElement = document.getElementById(`onboarding-step-${onboardingCurrentStep}`);
    if (!currentStepElement) return;
    
    switch (onboardingCurrentStep) {
        case 1: // Language preferences
            const primaryLanguage = document.getElementById('primary-language').value;
            const additionalLanguages = Array.from(currentStepElement.querySelectorAll('.language-checkboxes input[type="checkbox"]:checked')).map(cb => cb.value);
            const autoTranslation = document.getElementById('enable-auto-translation').checked;
            
            onboardingData.language = {
                primary: primaryLanguage,
                additional: additionalLanguages,
                autoTranslation: autoTranslation
            };
            break;
            
        case 2: // Location & Currency
            const country = document.getElementById('country-select').value;
            const timezone = document.getElementById('timezone-select').value;
            const currency = document.getElementById('currency-select').value;
            const showFTNS = document.getElementById('show-ftns-equivalent').checked;
            
            onboardingData.location = {
                country: country,
                timezone: timezone,
                currency: currency,
                showFTNSEquivalent: showFTNS
            };
            break;
            
        case 3: // Work preferences
            const workCategories = Array.from(currentStepElement.querySelectorAll('.work-categories input[type="checkbox"]:checked')).map(cb => cb.value);
            const availability = document.getElementById('availability-select').value;
            const experience = document.getElementById('experience-level').value;
            
            onboardingData.work = {
                categories: workCategories,
                availability: availability,
                experience: experience
            };
            break;
            
        case 4: // Notification settings
            const countryCode = document.getElementById('country-code').value;
            const phoneNumber = document.getElementById('notification-phone').value;
            const fullPhone = phoneNumber ? `${countryCode}${phoneNumber}` : '';
            
            const notifyNewJobs = document.getElementById('notify-new-jobs').checked;
            const notifyHighPaying = document.getElementById('notify-high-paying').checked;
            const notifyUrgent = document.getElementById('notify-urgent-jobs').checked;
            const notifyUpdates = document.getElementById('notify-job-updates').checked;
            
            const frequency = document.getElementById('notification-frequency').value;
            const quietStart = document.getElementById('quiet-start').value;
            const quietEnd = document.getElementById('quiet-end').value;
            
            onboardingData.notifications = {
                phone: fullPhone,
                preferences: {
                    newJobs: notifyNewJobs,
                    highPaying: notifyHighPaying,
                    urgent: notifyUrgent,
                    updates: notifyUpdates
                },
                frequency: frequency,
                quietHours: {
                    start: quietStart,
                    end: quietEnd
                }
            };
            break;
    }
}

// Complete the onboarding process
function completeOnboarding() {
    // Save onboarding data to localStorage
    localStorage.setItem('prsmOnboardingData', JSON.stringify(onboardingData));
    localStorage.setItem('prsmOnboardingCompleted', 'true');
    
    // Close modal
    const modal = document.getElementById('user-onboarding-modal');
    if (modal) {
        modal.style.display = 'none';
    }
    
    // Update button to show "Get Started" instead of onboarding trigger
    addOnboardingTrigger();
    
    // Show success message
    showActionFeedback('completed', 'User onboarding setup');
    
    // Apply user preferences
    applyUserPreferences();
    
    // Show data work content
    showDataWorkContent();
    
    console.log('Onboarding completed:', onboardingData);
}

// Apply user preferences to the UI
function applyUserPreferences() {
    const savedData = JSON.parse(localStorage.getItem('prsmOnboardingData'));
    if (!savedData) return;
    
    // Apply language preferences
    if (savedData.language && savedData.language.autoTranslation) {
        // Enable auto-translation in data work hub
        const autoTranslateBtn = document.querySelector('.auto-translate-btn');
        if (autoTranslateBtn) {
            autoTranslateBtn.classList.add('active');
        }
    }
    
    // Apply currency preferences
    if (savedData.location && savedData.location.currency) {
        // Update currency displays in job cards
        updateCurrencyDisplays(savedData.location.currency);
    }
    
    // Apply work category filters
    if (savedData.work && savedData.work.categories) {
        // Pre-filter jobs based on user categories
        filterJobsByCategories(savedData.work.categories);
    }
}

// Update currency displays based on user preference
function updateCurrencyDisplays(preferredCurrency) {
    const currencyElements = document.querySelectorAll('.currency-display');
    currencyElements.forEach(element => {
        // In a real app, this would convert FTNS to user's preferred currency
        const ftnsAmount = element.dataset.ftns;
        if (ftnsAmount) {
            const convertedAmount = convertFTNSToCurrency(ftnsAmount, preferredCurrency);
            element.textContent = `${convertedAmount} ${preferredCurrency}`;
        }
    });
}

// Filter jobs by user's selected categories
function filterJobsByCategories(categories) {
    const jobCards = document.querySelectorAll('.job-card');
    jobCards.forEach(card => {
        const jobCategory = card.dataset.category;
        if (jobCategory && categories.includes(jobCategory)) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

// Convert FTNS to user's preferred currency (mock function)
function convertFTNSToCurrency(ftnsAmount, currency) {
    // Mock conversion rates - in real app this would be from API
    const rates = {
        'USD': 0.0012,
        'EUR': 0.0011,
        'GBP': 0.0009,
        'NGN': 0.75,
        'KES': 0.15,
        'INR': 0.095,
        'BRL': 0.0065
    };
    
    const rate = rates[currency] || rates['USD'];
    return (ftnsAmount * rate).toFixed(2);
}

// Show data work content after onboarding
function showDataWorkContent() {
    // Switch to marketplace tab
    const marketplaceTab = document.querySelector('[data-target="marketplace-content"]');
    if (marketplaceTab) {
        marketplaceTab.click();
    }
    
    // Navigate directly to Data Work Hub
    setTimeout(() => {
        showDataWorkHub();
    }, 100); // Small delay to ensure tab switching completes
}

// Close onboarding modal when clicking outside
document.addEventListener('click', (e) => {
    const modal = document.getElementById('user-onboarding-modal');
    if (modal && e.target === modal) {
        modal.style.display = 'none';
    }
});

// Initialize onboarding when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('Onboarding initialization...');
        initializeOnboarding();
    });
} else {
    console.log('Onboarding initialization (DOM already loaded)...');
    initializeOnboarding();
}