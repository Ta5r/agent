document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // STATE
    let chatHistory = [];
    let knownFiles = ["refund_policy.pdf"];
    let previousTicketsCount = 0;

    // DOM ELEMENTS
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const logCountBadge = document.getElementById("log-count");
    const logsFeed = document.getElementById("logs-feed");
    
    const ordersTableBody = document.querySelector("#orders-table tbody");
    const ticketsTableBody = document.querySelector("#tickets-table tbody");
    
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const uploadStatus = document.getElementById("upload-status");
    const ragFilesList = document.getElementById("rag-files-list");

    // INIT
    fetchDatabaseState();
    renderFilesList();

    // TAB SWITCHING
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // CHAT SYSTEM
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        // Clear input & disable
        chatInput.value = "";
        setChatDisabled(true);

        // Add user message to UI
        appendMessage("user", text);

        // Show typing loader
        typingIndicator.style.display = "flex";
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    history: chatHistory
                })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Hide typing loader
            typingIndicator.style.display = "none";

            // Add agent message to UI
            appendMessage("model", data.response);

            // Update chat history
            chatHistory.push({ role: "user", text: text });
            chatHistory.push({ role: "model", text: data.response });

            // Render execution logs
            renderLogs(data.logs || []);
            
            // Refresh DB state to check for updates
            await fetchDatabaseState();

        } catch (err) {
            typingIndicator.style.display = "none";
            appendMessage("system", `Error: ${err.message}. Please check if the backend server is running and the API key is valid.`);
        } finally {
            setChatDisabled(false);
            chatInput.focus();
        }
    });

    clearChatBtn.addEventListener("click", () => {
        chatHistory = [];
        chatMessages.innerHTML = `
            <div class="message system-message">
                Chat session reset. Say hello to start a new simulation!
            </div>
            <div class="message agent-message">
                <div class="message-content">
                    Hello! I am your Quantum Tech Support assistant. How can I help you today?
                </div>
                <div class="message-meta">Agent • Just now</div>
            </div>
        `;
        renderLogs([]);
        chatInput.focus();
    });

    function setChatDisabled(disabled) {
        chatInput.disabled = disabled;
        chatForm.querySelector("button").disabled = disabled;
    }

    function appendMessage(role, text) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}-message`;
        
        if (role === "system") {
            messageDiv.className = "message system-message";
            messageDiv.textContent = text;
        } else {
            const contentDiv = document.createElement("div");
            contentDiv.className = "message-content";
            contentDiv.innerHTML = formatMarkdown(text);
            
            const metaDiv = document.createElement("div");
            metaDiv.className = "message-meta";
            metaDiv.textContent = role === "user" ? "You • Just now" : "Agent • Just now";
            
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(metaDiv);
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Simplistic Markdown parser (bold and code spans)
    function formatMarkdown(text) {
        let escaped = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
        
        // Match headers
        escaped = escaped.replace(/^### (.*?)$/gm, "<h3>$1</h3>");
        
        // Match bold **text**
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        
        // Match code spans `code`
        escaped = escaped.replace(/`(.*?)`/g, "<code>$1</code>");
        
        // Match linebreaks
        return escaped.replace(/\n/g, "<br>");
    }

    // RENDER REASONING LOGS
    function renderLogs(logs) {
        logCountBadge.textContent = logs.filter(l => l.action && l.action.startsWith("Calling")).length;
        
        if (logs.length === 0) {
            logsFeed.innerHTML = `
                <div class="empty-logs">
                    <i data-lucide="activity"></i>
                    <p>No tool execution logs yet. Send a message to see the agent reasoning in action.</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        logsFeed.innerHTML = "";
        
        // Group logs into card pairs (Call + Result)
        let stepCount = 1;
        for (let i = 0; i < logs.length; i++) {
            const current = logs[i];
            
            // Check if this is a tool call log
            if (current.action && current.action.startsWith("Calling")) {
                const logCard = document.createElement("div");
                logCard.className = "log-card";
                
                const nextLog = logs[i + 1] && logs[i + 1].action && logs[i + 1].action.includes("result") ? logs[i + 1] : null;
                
                // Format args nicely
                const argsStr = JSON.stringify(current.arguments || {}, null, 2);
                
                let resultHtml = "";
                if (nextLog) {
                    let resVal = nextLog.result;
                    if (typeof resVal === 'object') {
                        resVal = JSON.stringify(resVal, null, 2);
                    }
                    resultHtml = `
                        <div class="log-result">
                            <span>Returned Output:</span>
                            <p>${resVal}</p>
                        </div>
                    `;
                    i++; // skip next log since it's processed
                }
                
                logCard.innerHTML = `
                    <div class="log-header">
                        <span class="log-title">${current.action}</span>
                        <span class="log-step-tag">Step ${stepCount++}</span>
                    </div>
                    <div class="log-body">
                        <div class="log-payload">
                            <pre>${argsStr}</pre>
                        </div>
                        ${resultHtml}
                    </div>
                `;
                
                logsFeed.appendChild(logCard);
            } else if (current.error) {
                const errCard = document.createElement("div");
                errCard.className = "log-card";
                errCard.style.borderColor = "var(--error)";
                errCard.innerHTML = `
                    <div class="log-header">
                        <span class="log-title" style="color: var(--error)">API Error Encountered</span>
                    </div>
                    <div class="log-body">
                        <p style="font-size: 0.82rem; color: var(--text-secondary);">${current.error}</p>
                    </div>
                `;
                logsFeed.appendChild(errCard);
            }
        }
        
        lucide.createIcons();
    }

    // DATABASE CALLS
    async function fetchDatabaseState() {
        try {
            const [ordersRes, ticketsRes] = await Promise.all([
                fetch("/api/db/orders"),
                fetch("/api/db/tickets")
            ]);
            
            if (ordersRes.ok) {
                const orders = await ordersRes.json();
                renderOrders(orders);
            }
            
            if (ticketsRes.ok) {
                const tickets = await ticketsRes.json();
                renderTickets(tickets);
            }
        } catch (err) {
            console.error("Failed to load database tables:", err);
        }
    }

    function renderOrders(orders) {
        if (orders.length === 0) {
            ordersTableBody.innerHTML = `<tr><td colspan="7" class="loading-cell">No order data available.</td></tr>`;
            return;
        }
        
        ordersTableBody.innerHTML = orders.map(ord => `
            <tr>
                <td class="font-mono" style="color: var(--secondary); font-weight: 500;">${ord.order_id}</td>
                <td>${ord.customer_name}</td>
                <td>${ord.customer_email}</td>
                <td>${ord.product_name}</td>
                <td>
                    <span class="badge" style="
                        background-color: ${
                            ord.status === 'Delivered' ? 'rgba(16, 185, 129, 0.15)' :
                            ord.status === 'Shipped' ? 'rgba(6, 182, 212, 0.15)' :
                            ord.status === 'Cancelled' ? 'rgba(239, 68, 68, 0.15)' :
                            'rgba(245, 158, 11, 0.15)'
                        };
                        color: ${
                            ord.status === 'Delivered' ? 'var(--success)' :
                            ord.status === 'Shipped' ? 'var(--secondary)' :
                            ord.status === 'Cancelled' ? 'var(--error)' :
                            'var(--warning)'
                        };
                        border-color: transparent;
                    ">${ord.status}</span>
                </td>
                <td class="font-mono">${ord.tracking_number || '-'}</td>
                <td>${ord.purchase_date}</td>
            </tr>
        `).join("");
    }

    function renderTickets(tickets) {
        if (tickets.length === 0) {
            ticketsTableBody.innerHTML = `<tr><td colspan="6" class="loading-cell">No tickets filed yet.</td></tr>`;
            return;
        }

        const isNewTicketAdded = tickets.length > previousTicketsCount;
        
        ticketsTableBody.innerHTML = tickets.map((t, idx) => {
            // Flash row if it is a brand new ticket (index 0 since sorted DESC)
            const flashClass = (isNewTicketAdded && idx === 0) ? 'class="highlight-flash"' : '';
            return `
                <tr ${flashClass}>
                    <td class="font-mono">${t.id}</td>
                    <td>${t.customer_email}</td>
                    <td style="font-weight: 500; color: #fff;">${t.subject}</td>
                    <td>${t.description}</td>
                    <td>
                        <span class="badge" style="background-color: var(--success-bg); color: var(--success); border-color: transparent;">
                            ${t.status}
                        </span>
                    </td>
                    <td>${t.created_at}</td>
                </tr>
            `;
        }).join("");
        
        previousTicketsCount = tickets.length;
    }

    // RAG DOCUMENT UPLOADING
    uploadZone.addEventListener("click", () => fileInput.click());
    
    // Drag events
    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.remove("dragover");
        }, false);
    });

    uploadZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endswith(".pdf")) {
            showUploadStatus("Only PDF documents are allowed.", "error");
            return;
        }

        showUploadStatus(`Uploading and indexing '${file.name}'...`, "info");
        
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload-pdf", {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            
            if (res.ok) {
                showUploadStatus(data.message, "success");
                if (!knownFiles.includes(file.name)) {
                    knownFiles.push(file.name);
                }
                renderFilesList();
            } else {
                showUploadStatus(data.detail || "Upload failed", "error");
            }
        } catch (err) {
            showUploadStatus(`Error: ${err.message}`, "error");
        }
    }

    function showUploadStatus(msg, type) {
        uploadStatus.textContent = msg;
        uploadStatus.className = `upload-status ${type}`;
    }

    function renderFilesList() {
        ragFilesList.innerHTML = knownFiles.map(name => `
            <div class="rag-file-item">
                <div class="file-icon"><i data-lucide="file-text"></i></div>
                <div class="file-details">
                    <span class="file-name">${name}</span>
                    <span class="file-meta">Embedded • Ready</span>
                </div>
            </div>
        `).join("");
        lucide.createIcons();
    }
});
// Simple polyfill just in case JavaScript endswith doesn't support case/syntax check
String.prototype.endswith = function(suffix) {
    return this.indexOf(suffix, this.length - suffix.length) !== -1;
};
