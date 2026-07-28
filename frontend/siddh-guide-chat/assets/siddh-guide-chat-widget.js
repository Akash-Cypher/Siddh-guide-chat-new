document.addEventListener('DOMContentLoaded', () => {
  // For the live WordPress site, API_BASE must point to the proxy endpoint.
  const API_BASE = '/wp-json/siddh/v1';

  function generateSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return window.crypto.randomUUID();
    }
    return 'sess_' + Math.random().toString(36).slice(2) + Date.now();
  }

  // A conversation lives exactly as long as this loaded page.
  //
  // The id is generated once, here, and held ONLY in this closure variable. It is
  // deliberately never written to localStorage or sessionStorage, so:
  //   - every message from this page carries the same id (continuity);
  //   - closing and reopening the bubble keeps that id (same page, same script);
  //   - a refresh re-runs this script and produces a brand-new id (isolation);
  //   - another page or tab runs its own script, so it gets its own id.
  // Old turns stay in the backend until the DynamoDB TTL expires them, but a new
  // page can never reach them because it never learns the previous id.
  const SESSION_ID = generateSessionId();

  const chatToggleButton = document.getElementById('chatToggleButton');
  const chatWindow = document.getElementById('chatWindow');
  const proactiveMessage = document.getElementById('proactiveMessage');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const chatMessages = document.getElementById('chatMessages');

  const submitButton = chatForm.querySelector('button[type="submit"]');

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

  // Only one request may be in flight per widget. Without this, a visitor who
  // submits twice quickly can have the two answers — and the two DynamoDB
  // history writes — land out of order, which corrupts the follow-up context of
  // every later turn in the conversation.
  let requestInFlight = false;

  function setBusy(busy) {
    requestInFlight = busy;
    userInput.disabled = busy;
    if (submitButton) {
      submitButton.disabled = busy;
    }
    if (!busy) {
      userInput.focus();
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
    setBusy(true);

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
    } finally {
      // Always restore the controls — on success AND on failure — so a network
      // error can never leave the widget permanently locked.
      setBusy(false);
    }
  }

  chatToggleButton.addEventListener('click', () => {
    const isVisible = chatWindow.classList.toggle('visible');

    if (isVisible) {
      stopProactiveMessaging();
      userInput.focus();
    } else {
      // Closing the bubble does NOT end the conversation: SESSION_ID is bound to
      // the page, not to the window being open, so reopening continues the same
      // thread. Only a page refresh starts a new conversation.
      startProactiveMessaging();
    }
  });

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();

    // Drop the submit outright while a request is open, so answers and their
    // history writes cannot interleave.
    if (requestInFlight) return;

    const messageText = userInput.value.trim();
    if (!messageText) return;

    addMessage(messageText, 'user');
    userInput.value = '';
    fetchBotResponse(messageText);
  });

  setTimeout(startProactiveMessaging, 3000);
});