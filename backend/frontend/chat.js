const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');

function addMessage(message, isUser) {
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user' : 'bot'}`;
    div.textContent = message;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    messageInput.value = '';

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        addMessage(data.answer, false);
        console.log('Sources:', data.sources);
    } catch (error) {
        addMessage('Error: ' + error.message, false);
    }
}

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
