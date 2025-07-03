/**
 * MCP Expert Chatbot Panel JavaScript
 * Handles chat functionality, API communication, and UI interactions
 */

class MCPExpertChatbot {
  constructor() {
    this.apiUrl = "http://localhost:8000";
    this.messages = [];
    this.isConnected = false;
    this.websocket = null;

    this.initializeElements();
    this.attachEventListeners();
    this.checkConnection();
    this.loadTheme();
  }

  initializeElements() {
    this.messagesContainer = document.getElementById("messages");
    this.messageInput = document.getElementById("message-input");
    this.sendButton = document.getElementById("send-btn");
    this.statusDot = document.getElementById("status-dot");
    this.statusText = document.getElementById("status-text");
    this.statsText = document.getElementById("stats-text");
    this.emptyState = document.getElementById("empty-state");
    this.examplesContainer = document.getElementById("examples-container");
    this.themeToggle = document.getElementById("theme-toggle");
    this.clearButton = document.getElementById("clear-chat");
    this.examplesToggle = document.getElementById("examples-toggle");
  }

  attachEventListeners() {
    // Send message events
    this.sendButton.addEventListener("click", () => this.sendMessage());
    this.messageInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    this.messageInput.addEventListener("input", () => {
      this.messageInput.style.height = "auto";
      this.messageInput.style.height =
        Math.min(this.messageInput.scrollHeight, 120) + "px";
    });

    // Theme toggle
    this.themeToggle.addEventListener("click", () => this.toggleTheme());

    // Clear chat
    this.clearButton.addEventListener("click", () => this.clearChat());

    // Examples toggle
    this.examplesToggle.addEventListener("click", () => this.toggleExamples());

    // Example and quick action clicks
    document.addEventListener("click", (e) => {
      if (
        e.target.matches(".example-item") ||
        e.target.matches(".quick-action")
      ) {
        const query = e.target.getAttribute("data-query");
        if (query) {
          this.messageInput.value = query;
          this.sendMessage();
        }
      }
    });

    // Code block actions
    document.addEventListener("click", (e) => {
      if (e.target.matches(".copy-code")) {
        this.copyCode(e.target);
      } else if (e.target.matches(".insert-code")) {
        this.insertCode(e.target);
      }
    });
  }

  async checkConnection() {
    try {
      const response = await fetch(`${this.apiUrl}/health`);
      const data = await response.json();
      this.isConnected = response.ok;

      if (this.isConnected) {
        this.statusDot.classList.remove("disconnected");
        this.statusText.textContent = "Connected";
        this.statsText.textContent = data.gemini_configured
          ? "Ready to help with MCP"
          : "Gemini not configured";
      } else {
        throw new Error("Health check failed");
      }
    } catch (error) {
      this.isConnected = false;
      this.statusDot.classList.add("disconnected");
      this.statusText.textContent = "Disconnected";
      this.statsText.textContent = "Backend server not available";
      console.error("Connection check failed:", error);
    }
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message || !this.isConnected) return;

    // Clear input and hide empty state
    this.messageInput.value = "";
    this.messageInput.style.height = "auto";
    this.hideEmptyState();

    // Add user message to chat
    this.addMessage("user", message);

    // Show loading indicator
    const loadingId = this.addMessage("assistant", "", true);

    // Disable send button
    this.sendButton.disabled = true;
    this.sendButton.textContent = "Sending...";

    try {
      const response = await this.callAPI("/chat", {
        message: message,
        context: this.getRecentContext(),
      });

      // Remove loading indicator
      this.removeMessage(loadingId);

      // Add assistant response
      this.addMessage("assistant", response.response, false, {
        code_samples: response.code_samples || [],
        references: response.references || [],
      });
    } catch (error) {
      // Remove loading indicator and show error
      this.removeMessage(loadingId);
      this.addMessage(
        "assistant",
        `Sorry, I encountered an error: ${error.message}`
      );
      console.error("Send message error:", error);
    } finally {
      // Re-enable send button
      this.sendButton.disabled = false;
      this.sendButton.textContent = "Send";
    }
  }

  async callAPI(endpoint, data) {
    const response = await fetch(`${this.apiUrl}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.statusText}`);
    }

    return await response.json();
  }

  addMessage(role, content, isLoading = false, metadata = {}) {
    const messageId = `msg-${Date.now()}-${Math.random()}`;
    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;
    messageEl.id = messageId;

    if (isLoading) {
      messageEl.innerHTML = `
                <div class="message-content">
                    <div class="loading">
                        <div class="spinner"></div>
                        <span>Thinking...</span>
                    </div>
                </div>
            `;
    } else {
      const messageContent = this.formatMessageContent(content, metadata);
      messageEl.innerHTML = `
                <div class="message-content">
                    ${messageContent}
                </div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            `;
    }

    this.messagesContainer.appendChild(messageEl);
    this.scrollToBottom();

    // Store message in history
    if (!isLoading) {
      this.messages.push({ role, content, timestamp: Date.now(), metadata });
    }

    return messageId;
  }

  removeMessage(messageId) {
    const messageEl = document.getElementById(messageId);
    if (messageEl) {
      messageEl.remove();
    }
  }

  formatMessageContent(content, metadata = {}) {
    let html = this.markdownToHTML(content);

    // Add code samples if available
    if (metadata.code_samples && metadata.code_samples.length > 0) {
      metadata.code_samples.forEach((sample) => {
        html += this.createCodeBlock(sample.code, sample.language || "python");
      });
    }

    // Add references if available
    if (metadata.references && metadata.references.length > 0) {
      html += `
                <div class="references">
                    <strong>References:</strong>
                    ${metadata.references
                      .map((ref) => `<span class="reference-tag">${ref}</span>`)
                      .join("")}
                </div>
            `;
    }

    return html;
  }

  createCodeBlock(code, language) {
    const codeId = `code-${Date.now()}-${Math.random()}`;
    return `
            <div class="code-block">
                <div class="code-header">
                    <span class="code-language">${language}</span>
                    <div class="code-actions">
                        <button class="btn copy-code" data-code-id="${codeId}">Copy</button>
                        <button class="btn insert-code" data-code-id="${codeId}">Insert</button>
                    </div>
                </div>
                <div class="code-content">
                    <pre id="${codeId}"><code>${this.escapeHtml(
      code
    )}</code></pre>
                </div>
            </div>
        `;
  }

  markdownToHTML(text) {
    // Basic markdown to HTML conversion
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br>");
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  async copyCode(button) {
    const codeId = button.getAttribute("data-code-id");
    const codeEl = document.getElementById(codeId);

    if (codeEl) {
      try {
        await navigator.clipboard.writeText(codeEl.textContent);
        button.textContent = "Copied!";
        setTimeout(() => {
          button.textContent = "Copy";
        }, 2000);
      } catch (error) {
        console.error("Copy failed:", error);
        button.textContent = "Failed";
        setTimeout(() => {
          button.textContent = "Copy";
        }, 2000);
      }
    }
  }

  async insertCode(button) {
    const codeId = button.getAttribute("data-code-id");
    const codeEl = document.getElementById(codeId);

    if (codeEl) {
      // In a real Cursor extension, this would use the Cursor API
      // For now, we'll copy to clipboard and show a message
      try {
        await navigator.clipboard.writeText(codeEl.textContent);
        button.textContent = "Copied!";
        setTimeout(() => {
          button.textContent = "Insert";
        }, 2000);

        // Show notification (in real extension, this would insert directly)
        this.showNotification(
          "Code copied to clipboard. Paste in your editor."
        );
      } catch (error) {
        console.error("Insert failed:", error);
        button.textContent = "Failed";
        setTimeout(() => {
          button.textContent = "Insert";
        }, 2000);
      }
    }
  }

  showNotification(message) {
    // Simple notification system
    const notification = document.createElement("div");
    notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--accent);
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 14px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
      notification.style.opacity = "0";
      notification.style.transition = "opacity 0.3s";
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 3000);
  }

  getRecentContext() {
    // Get last few messages for context
    const recentMessages = this.messages.slice(-4);
    return recentMessages
      .map((msg) => `${msg.role}: ${msg.content}`)
      .join("\n");
  }

  hideEmptyState() {
    if (this.emptyState) {
      this.emptyState.style.display = "none";
    }
  }

  showEmptyState() {
    if (this.emptyState && this.messages.length === 0) {
      this.emptyState.style.display = "flex";
    }
  }

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  clearChat() {
    this.messages = [];
    this.messagesContainer.innerHTML = "";
    this.showEmptyState();
  }

  toggleExamples() {
    this.examplesContainer.classList.toggle("show");
  }

  toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute("data-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";

    body.setAttribute("data-theme", newTheme);
    this.themeToggle.textContent = newTheme === "dark" ? "🌙" : "☀️";

    // Save theme preference
    localStorage.setItem("mcp-expert-theme", newTheme);
  }

  loadTheme() {
    const savedTheme = localStorage.getItem("mcp-expert-theme") || "dark";
    document.body.setAttribute("data-theme", savedTheme);
    this.themeToggle.textContent = savedTheme === "dark" ? "🌙" : "☀️";
  }
}

// Initialize the chatbot when the page loads
document.addEventListener("DOMContentLoaded", () => {
  window.mcpExpert = new MCPExpertChatbot();
});

// Export for use in Cursor extension context
if (typeof module !== "undefined" && module.exports) {
  module.exports = MCPExpertChatbot;
}
