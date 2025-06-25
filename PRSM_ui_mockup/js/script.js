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
        if (!historySidebar) return;
        const shouldHide = hide ?? !historySidebar.classList.contains('hidden');
        
        // Toggle the hidden class
        // console.log(`Toggling history sidebar. Should hide: ${shouldHide}`); // Removed log
        historySidebar.classList.toggle('hidden', shouldHide);
        
        // Update the toggle button icon/title
        if (historyToggleBtn) {
            historyToggleBtn.title = shouldHide ? "Show History" : "Hide History";
        }
        
        // Save state to localStorage
        localStorage.setItem(historySidebarHiddenKey, shouldHide ? 'true' : 'false');
    };

    if (historyToggleBtn) {
        historyToggleBtn.addEventListener('click', () => toggleHistorySidebar());
    }

    // Apply saved history state on load, defaulting to visible
    const savedHistoryHidden = localStorage.getItem(historySidebarHiddenKey) === 'true';
    if (historySidebar) {
        // Explicitly set the class based on the saved state or default (visible)
        historySidebar.classList.toggle('hidden', savedHistoryHidden);
    }


    // --- Left Panel Toggle (UPDATED to handle history) ---
    const toggleLeftPanel = (collapse) => {
        if (!leftPanel || !rightPanel || !resizer) return;

        const shouldCollapse = collapse ?? !leftPanel.classList.contains('collapsed');
        const collapsedWidth = getCssVariable('--left-panel-collapsed-width') || '50px'; // Fallback
        const resizerWidth = resizer.offsetWidth;

        // console.log(`Toggling left panel. Should collapse: ${shouldCollapse}`); // Removed log
        if (shouldCollapse) {
            // Store current width before collapsing (if not already collapsed)
            if (!leftPanel.classList.contains('collapsed')) {
                 const currentWidth = leftPanel.offsetWidth > 0 ? `${leftPanel.offsetWidth}px` : (localStorage.getItem(leftPanelWidthKey) || '50%');
                 localStorage.setItem(leftPanelWidthKey, currentWidth);
            }

            leftPanel.classList.add('collapsed');
            leftPanel.style.width = collapsedWidth;
            rightPanel.style.width = `calc(100% - ${collapsedWidth})`;
            resizer.style.display = 'none';
            localStorage.setItem(leftPanelCollapsedKey, 'true');
            // Ensure history is hidden when main panel collapses
            if (historySidebar) historySidebar.classList.add('hidden');

        } else {
            leftPanel.classList.remove('collapsed');
            const restoredWidth = localStorage.getItem(leftPanelWidthKey) || '50%';
            leftPanel.style.width = restoredWidth;
            rightPanel.style.width = `calc(100% - ${restoredWidth} - ${resizerWidth}px)`;
            resizer.style.display = 'block';
            localStorage.setItem(leftPanelCollapsedKey, 'false');
            // Restore history sidebar state (it might have been open before collapsing)
             if (historySidebar && localStorage.getItem(historySidebarHiddenKey) !== 'true') {
                 historySidebar.classList.remove('hidden');
             }
        }
    };

    if (leftPanelToggleBtn) {
        leftPanelToggleBtn.addEventListener('click', () => toggleLeftPanel());
    }

     // Apply saved collapsed state on load (UPDATED)
     const savedCollapsed = localStorage.getItem(leftPanelCollapsedKey) === 'true';
     if (savedCollapsed && leftPanel && rightPanel && resizer) {
         const collapsedWidth = getCssVariable('--left-panel-collapsed-width') || '50px';
         leftPanel.classList.add('collapsed');
         leftPanel.style.width = collapsedWidth;
         rightPanel.style.width = `calc(100% - ${collapsedWidth})`;
         resizer.style.display = 'none';
         // Ensure history is hidden if main panel starts collapsed
         if (historySidebar) historySidebar.classList.add('hidden');
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

// Make functions available globally for onclick handlers
window.connectPlatform = connectPlatform;
window.disconnectPlatform = disconnectPlatform;
window.searchContent = searchContent;
window.startImport = startImport;
window.refreshImportHistory = refreshImportHistory;
window.checkSystemHealth = checkSystemHealth;