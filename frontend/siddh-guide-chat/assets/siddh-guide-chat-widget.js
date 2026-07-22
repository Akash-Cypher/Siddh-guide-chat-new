document.addEventListener('DOMContentLoaded', () => {
  // For the live WordPress site, API_BASE must point to the proxy endpoint.
  const API_BASE = '/wp-json/siddh/v1';

  function generateSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return window.crypto.randomUUID();
    }
    return 'sess_' + Math.random().toString(36).slice(2) + Date.now();
  }

  // Keep one stable session id per browser tab so multi-turn continuity works
  // (the backend threads follow-ups like "how much is it?" by session id).
  // sessionStorage = one conversation per tab; survives reloads, clears when the
  // tab is closed. Falls back to an in-memory id if storage is unavailable.
  const SESSION_STORAGE_KEY = 'siddhGuideChatSessionId';

  function loadSessionId() {
    try {
      let id = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (!id) {
        id = generateSessionId();
        window.sessionStorage.setItem(SESSION_STORAGE_KEY, id);
      }
      return id;
    } catch (e) {
      return generateSessionId();
    }
  }

  let SESSION_ID = loadSessionId();

  const chatToggleButton = document.getElementById('chatToggleButton');
  const chatWindow = document.getElementById('chatWindow');
  const proactiveMessage = document.getElementById('proactiveMessage');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const chatMessages = document.getElementById('chatMessages');

  if (!chatToggleButton || !chatWindow || !proactiveMessage || !chatForm || !userInput || !chatMessages) {
    return;
  }

  const proactiveMessages = [
    'Hi! Do you need any guidance?',
    'Hello! Are you looking for any course-related help?',
    'Need help with our services? Ask me!',
    'Curious about our courses? I can help.'
  ];

  let proactiveIndex = 0;
  let proactiveInterval = null;

  function addMessage(text, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.textContent = text;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageElement;
  }

  function clearMessages() {
    chatMessages.innerHTML = '';
  }

  function resetChatSession() {
    clearMessages();
    SESSION_ID = generateSessionId();
    try {
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, SESSION_ID);
    } catch (e) {
      // storage unavailable — in-memory id still works for this page view
    }
  }

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
    if (proactiveInterval) {
      clearInterval(proactiveInterval);
      proactiveInterval = null;
    }
    proactiveMessage.classList.remove('visible');
  }

  function typewriterEffect(element, text) {
    let i = 0;
    element.textContent = '';
    const speed = 8;

    function type() {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i += 1;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        setTimeout(type, speed);
      }
    }

    type();
  }

  async function fetchBotResponse(userMessage) {
    const botMessageElement = addMessage('...', 'bot');
    botMessageElement.classList.add('typing');

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: SESSION_ID
        })
      });

      if (!response.ok) {
        let errText = `HTTP error: ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson && errJson.detail) {
            errText = errJson.detail;
          }
        } catch (e) {
          // ignore parse issue
        }
        throw new Error(errText);
      }

      const data = await response.json();
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = '';
      typewriterEffect(botMessageElement, data.answer || '');
    } catch (error) {
      console.error('Error fetching bot response:', error);
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = 'Sorry, something went wrong. Please try again.';
    }
  }

  chatToggleButton.addEventListener('click', () => {
    const isVisible = chatWindow.classList.toggle('visible');

    if (isVisible) {
      stopProactiveMessaging();
      userInput.focus();
    } else {
      // Closing the bubble should NOT start a new conversation — keep the same
      // session so reopening continues the thread (continuity). A fresh chat is
      // an explicit action via resetChatSession() if you add a "new chat" button.
      startProactiveMessaging();
    }
  });

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();

    const messageText = userInput.value.trim();
    if (!messageText) return;

    addMessage(messageText, 'user');
    userInput.value = '';
    fetchBotResponse(messageText);
  });

  setTimeout(startProactiveMessaging, 3000);
});