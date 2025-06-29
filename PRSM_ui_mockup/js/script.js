document.addEventListener('DOMContentLoaded', () => {
    // --- DOM References ---
    const leftPanel = document.getElementById('left-panel');
    const rightPanel = document.getElementById('right-panel');
    const resizer = document.getElementById('drag-handle');
    const leftPanelToggleBtn = document.getElementById('left-panel-toggle');
    const logoImage = document.getElementById('logo-img'); // Logo in right header
    const rightPanelTabs = document.querySelectorAll('.right-panel-nav .nav-tab-btn');
    const rightPanelContentSections = document.querySelectorAll('.right-panel-content-area .content-section');

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
    applyTheme(savedTheme);


    // --- Profile Dropdown Logic ---
    const toggleProfileDropdown = (show) => {
        if (!profileDropdown) return;
        const shouldShow = show ?? !profileDropdown.classList.contains('show');
        profileDropdown.classList.toggle('show', shouldShow);
    };

    if (profileButton) {
        profileButton.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleProfileDropdown();
        });
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
            toggleTheme();
            toggleProfileDropdown(false);
        });
    }

    if (dropdownSettings) {
        dropdownSettings.addEventListener('click', (e) => {
            e.preventDefault();
            // console.log("Settings dropdown item clicked"); // Log click - Removed
            showRightPanelSection('settings-content');
            toggleProfileDropdown(false);
            // queryApiSettingFields(); // This is already called by showRightPanelSection logic if target is settings
        });
            // Activate the settings tab - THIS WAS LIKELY THE ISSUE FOR SETTINGS ACCESS
            // showRightPanelSection('settings-content'); // Keep commented out
            // Ensure setting fields are queried after potential dynamic loading
            // queryApiSettingFields(); // This is handled elsewhere now
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

    rightPanelTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.target; // Get targetId here
            // console.log(`Tab clicked: ${targetId}`); // Removed log
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

            // console.log(`Showing section: ${targetId}`); // Removed log
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
            const filesGrid = document.querySelector('.files-grid');
            if (filesGrid) {
                filesGrid.className = view === 'list' ? 'files-list' : 'files-grid';
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
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-left: 4px solid ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#007bff'};
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10000;
        max-width: 400px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.9rem;
        animation: slideIn 0.3s ease;
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