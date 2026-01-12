document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------
  // Your FastAPI backend (App Runner). Replace with your custom domain later.
  const API_BASE = "https://nyrqbf2z3k.ap-south-1.awsapprunner.com";

  // TEMP: Only for testing. In real production, DO NOT expose this in JS.
  // Best approach: WordPress proxy endpoint (server-side) to keep key secret.
  const CHAT_API_KEY = ""; // <-- paste your key here for testing (optional if backend key disabled)

  // Session id (simple). You can improve later (cookie/localStorage).
  const SESSION_ID = "default";
  // -------------------------------------------------------------------------

  const chatToggleButton = document.getElementById('chatToggleButton');
  const chatWindow = document.getElementById('chatWindow');
  const proactiveMessage = document.getElementById('proactiveMessage');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const chatMessages = document.getElementById('chatMessages');

  const proactiveMessages = [
    "Hi! Do you need any guidance?",
    "Hello! Are you looking for any course-related stuff?",
    "Need help with our services? Ask me!",
    "Curious about our courses? I can help."
  ];
  let proactiveIndex = 0;
  let proactiveInterval;

  // --- Proactive Messaging Logic ---
  function showProactiveMessage() {
    proactiveMessage.textContent = proactiveMessages[proactiveIndex];
    proactiveMessage.classList.add('visible');

    setTimeout(() => {
      proactiveMessage.classList.remove('visible');
    }, 5000);

    proactiveIndex = (proactiveIndex + 1) % proactiveMessages.length;
  }

  function startProactiveMessaging() {
    if (chatWindow.classList.contains('visible')) return;
    stopProactiveMessaging();
    proactiveInterval = setInterval(showProactiveMessage, 8000);
  }

  function stopProactiveMessaging() {
    clearInterval(proactiveInterval);
    proactiveMessage.classList.remove('visible');
  }

  // --- Chat Window Toggle Logic ---
  chatToggleButton.addEventListener('click', () => {
    const isVisible = chatWindow.classList.toggle('visible');
    if (isVisible) {
      stopProactiveMessaging();
      userInput.focus();
    } else {
      startProactiveMessaging();
    }
  });

  // --- Message Handling Logic ---
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const messageText = userInput.value.trim();
    if (!messageText) return;

    addMessage(messageText, 'user');
    userInput.value = '';
    fetchBotResponse(messageText);
  });

  function addMessage(text, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.textContent = text;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageElement;
  }

  async function fetchBotResponse(userMessage) {
    const botMessageElement = addMessage('...', 'bot');
    botMessageElement.classList.add('typing');

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (CHAT_API_KEY) headers['x-api-key'] = CHAT_API_KEY;

      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          message: userMessage,
          session_id: SESSION_ID
        }),
      });

      if (!response.ok) {
        // Try to show a helpful message
        let errText = `HTTP error: ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson?.detail) errText = errJson.detail;
        } catch (_) {}
        throw new Error(errText);
      }

      const data = await response.json();
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = '';
      typewriterEffect(botMessageElement, data.answer || "");

    } catch (error) {
      console.error('Error fetching bot response:', error);
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = "Sorry, something went wrong. Please try again.";
    }
  }

  function typewriterEffect(element, text) {
    let i = 0;
    element.textContent = '';
    const speed = 30;

    function type() {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i++;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        setTimeout(type, speed);
      }
    }
    type();
  }

  // --- Initial State ---
  setTimeout(startProactiveMessaging, 3000);
});
