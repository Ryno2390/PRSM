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
    
    console.log('🎯 PRSM UI fully initialized with API integration');
});