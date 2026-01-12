# Chat Widget for Siddh Guide

This directory contains the frontend for a web-based chat widget. It is built with plain HTML, CSS, and JavaScript, designed to be embedded into any website.

## Overview

The widget consists of three main files:
- `index.html`: The HTML structure for the chat button and window.
- `style.css`: The stylesheet that defines the look and feel of the widget.
- `chat.js`: The JavaScript that handles user interaction, communication with the backend API, and renders messages.

## Configuration

Before the chat widget can be used, it must be configured to communicate with the backend API.

1.  **Open `chat.js`**.
2.  **Locate the `API_BASE_URL` constant** at the top of the file.
3.  **Update the URL** to point to your deployed backend service.

```javascript
// Replace this with the actual URL of your backend service on AWS App Runner
const API_BASE_URL = 'https://your-backend-service-url.com'; 
```

## How to Embed in a Website (e.g., WordPress)

You can add this chat widget to a WordPress site using the following steps:

### Step 1: Add the HTML Structure

Copy the HTML from the `<body>` of the `index.html` file. It will look like this:

```html
<!-- Chat Toggle Button -->
<div id="chatToggleButton">
    <img src="siddhanta-logo.png" alt="Chat">
    <div id="proactiveMessage"></div>
</div>

<!-- Chat Window -->
<div id="chatWindow">
    <div id="chatHeader">Siddh Guide</div>
    <div id="chatMessages"></div>
    <form id="chatForm">
        <input type="text" id="userInput" placeholder="Ask something..." autocomplete="off">
        <button type="submit">Send</button>
    </form>
</div>
```

Paste this HTML into your WordPress site. The easiest way is to use a "Custom HTML" block on the page or in the footer widget area where you want the chat button to appear.

### Step 2: Add the CSS and JavaScript

1.  **Link the Stylesheet:** Make the `style.css` file accessible (e.g., by uploading it to your server or hosting it with your frontend application) and add the following line to the `<head>` section of your website. You can typically do this in WordPress using a plugin like "Insert Headers and Footers".

    ```html
    <link rel="stylesheet" href="path/to/your/style.css">
    ```

2.  **Link the JavaScript:** Similarly, make the `chat.js` file accessible and add the following line just before the closing `</body>` tag of your website.

    ```html
    <script src="path/to/your/chat.js"></script>
    ```

    **Note:** Make sure the path to the `Siddhanta Logo .png` file is also correct and accessible from your WordPress site. You may need to upload it to the WordPress Media Library and update the `src` attribute in the HTML.

## API Contract

The frontend communicates with a backend service that exposes a `/chat` endpoint via POST request. The backend is a FastAPI application, which automatically generates interactive API documentation (Swagger UI) at its `/docs` endpoint. This can be shared with the WordPress team for their reference.
