// --- Configuration ---
const API_QUERY_URL = '/query';
const API_UPLOAD_URL = '/upload-pdf';

// Define the paths for your avatars (Update these paths!)
const USER_AVATAR = '/static/images/user_avatar.jpeg'; 
const BOT_AVATAR = '/static/images/bot_avatar.jpg';   

// --- DOM Elements ---
const messageHistory = document.getElementById('message-history');
const messageTemplate = document.getElementById('message-template');
const queryForm = document.getElementById('query-form');
const uploadForm = document.getElementById('upload-form');
const uploadStatus = document.getElementById('upload-status');
const queryStatus = document.getElementById('query-status');
const uploadButton = document.getElementById('upload-button');

// --- Utility Functions ---

/**
 * Displays a system status message (error or success).
 * @param {string} elementId - 'upload-status' or 'query-status'
 * @param {string} message - The message text.
 * @param {boolean} isError - True for error, false for success/info.
 */
function setStatus(element, message, isError = false) {
    element.textContent = message;
    element.style.display = 'block';
    element.style.backgroundColor = isError ? '#f8d7da' : '#d4edda';
    element.style.color = isError ? '#721c24' : '#155724';

    // Auto-hide after 5 seconds
    setTimeout(() => {
        element.style.display = 'none';
    }, 5000);
}

/**
 * Creates and appends a message bubble with an avatar to the chat history.
 * @param {string} sender - 'user' or 'bot'.
 * @param {string} text - The message content.
 */
function appendMessage(sender, text) {
    // 1. Clone the message template
    const clone = messageTemplate.content.cloneNode(true);
    const messageContainer = clone.querySelector('.message-container');
    const avatar = clone.querySelector('.avatar');
    const messageText = clone.querySelector('.message-text');

    messageText.innerHTML = formatBotResponse(text);

    // 2. Set classes and avatar source based on sender
    if (sender === 'user') {
        messageContainer.classList.add('outgoing');
        avatar.src = USER_AVATAR;
        avatar.alt = "User Avatar";
    } else { // 'bot'
        messageContainer.classList.add('incoming');
        avatar.src = BOT_AVATAR;
        avatar.alt = "Bot Avatar";
    }

    // 3. Append and scroll
    messageHistory.appendChild(clone);
    messageHistory.scrollTop = messageHistory.scrollHeight;
}

/**
 * Formats bot's text (markdown, citations) into HTML.
 * This is a basic formatter and may need more advanced logic for full markdown support.
 * @param {string} text - The raw text from the bot.
 * @returns {string} HTML formatted text.
 */
function formatBotResponse(text) {
    // Basic formatting for citations (e.g., [source_file: file.pdf - page: 12])
    const formatted = text.replace(
        /(\s)\[(source_file: [^\]]+)\]/g, 
        '$1<span class="citation" title="$2">🔍</span>'
    );
    // Basic line break conversion
    return formatted.replace(/\n/g, '<br>');
}


// --- Event Handlers ---

/**
 * Handles PDF file upload submission.
 */
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('pdf-file-input');
    const file = fileInput.files[0];

    if (!file) {
        setStatus(uploadStatus, "Please select a PDF file.", true);
        return;
    }

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);

    // Disable button and show loading
    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading & Re-indexing...";
    setStatus(uploadStatus, "Processing document and rebuilding RAG context. Please wait...", false);

    try {
        const response = await fetch(API_UPLOAD_URL, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        
        // Success message
        setStatus(uploadStatus, 
            `Successfully updated RAG context with "${result.filename}". ${result.total_chunks} chunks loaded.`, 
            false
        );
        
        // Clear file input
        fileInput.value = '';

    } catch (error) {
        setStatus(uploadStatus, `Upload failed: ${error.message}`, true);
    } finally {
        // Re-enable button
        uploadButton.disabled = false;
        uploadButton.textContent = "Upload PDF Context";
    }
});


/**
 * Handles chat query submission.
 */
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const userQueryInput = document.getElementById('user-query');
    const userQuery = userQueryInput.value.trim();

    if (!userQuery) return;

    // 1. Display user message
    appendMessage('user', userQuery);

    // Clear input field
    userQueryInput.value = '';
    queryStatus.style.display = 'none';

    // 2. Prepare payload
    const payload = {
        question: userQuery,
        top_k: 5, // Default retrieval limit
        summarize: false // Set to true if you want summaries returned
    };

    // 3. Display Bot placeholder/loading message
    const botLoadingMessage = appendMessage('bot', 'Assistant is thinking...');
    
    try {
        const response = await fetch(API_QUERY_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        // 4. Find and update the last message (the loading message)
        const botMessageElement = messageHistory.lastElementChild;
        const messageTextElement = botMessageElement.querySelector('.message-text');
        
        // Update the content with the final answer
        messageTextElement.innerHTML = formatBotResponse(result.answer);
        
        // Optional: Log sources for debugging
        console.log("Sources Used:", result.sources);

    } catch (error) {
        setStatus(queryStatus, `Query failed: ${error.message}`, true);
        
        // Update loading message to error message
        const botMessageElement = messageHistory.lastElementChild;
        const messageTextElement = botMessageElement.querySelector('.message-text');
        messageTextElement.textContent = `Error: Failed to get response. ${error.message}`;

    } finally {
        messageHistory.scrollTop = messageHistory.scrollHeight;
    }
});


// --- Initial Load Message ---
document.addEventListener('DOMContentLoaded', () => {
    // Add a welcome message on load
    appendMessage('bot', 
        "Hello! I am your RAG Assistant. Upload a PDF document above to set my context, and then ask me a question about it!"
    );
});