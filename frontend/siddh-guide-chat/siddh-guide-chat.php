<?php
/**
 * Plugin Name: Siddh Guide Chat
 * Description: Siddh Guide chatbot widget with secure WordPress proxy.
 * Version: 1.0.0
 */

if (!defined('ABSPATH')) {
  exit;
}

define('SIDDH_GUIDE_CHAT_PLUGIN_URL', plugin_dir_url(__FILE__));
define('SIDDH_GUIDE_CHAT_PLUGIN_PATH', plugin_dir_path(__FILE__));

function siddh_guide_chat_backend_base_url() {
  return 'https://nyrqbf2z3k.ap-south-1.awsapprunner.com';
}

function siddh_guide_chat_api_key() {
  return 'siddh-guide-2026-8f3c9b2a71e4d5c6f9a0';
}

/**
 * REST proxy routes
 */
add_action('rest_api_init', function () {
  register_rest_route('siddh/v1', '/chat', [
    'methods'  => 'POST',
    'callback' => 'siddh_guide_chat_proxy',
    'permission_callback' => '__return_true',
  ]);

  register_rest_route('siddh/v1', '/history/(?P<session_id>[A-Za-z0-9._:-]+)', [
    'methods'  => 'GET',
    'callback' => 'siddh_guide_chat_history_proxy',
    'permission_callback' => '__return_true',
  ]);
});

function siddh_guide_chat_proxy(WP_REST_Request $request) {
  $payload = $request->get_json_params();
  if (!is_array($payload)) {
    $payload = [];
  }

  $message = isset($payload['message']) ? trim((string) $payload['message']) : '';
  $session_id = isset($payload['session_id']) ? trim((string) $payload['session_id']) : '';

  if ($message === '') {
    return new WP_REST_Response(['detail' => 'message is required'], 400);
  }

  if ($session_id === '') {
    return new WP_REST_Response(['detail' => 'session_id is required'], 400);
  }

  $response = wp_remote_post(siddh_guide_chat_backend_base_url() . '/chat', [
    'headers' => [
      'Content-Type' => 'application/json',
      'Accept'       => 'application/json',
      'x-api-key'    => siddh_guide_chat_api_key(),
    ],
    'body'    => wp_json_encode([
      'message'    => $message,
      'session_id' => $session_id,
    ]),
    'timeout' => 30,
  ]);

  if (is_wp_error($response)) {
    return new WP_REST_Response(['detail' => 'Proxy request failed'], 500);
  }

  $status = wp_remote_retrieve_response_code($response);
  $body   = wp_remote_retrieve_body($response);

  $decoded = json_decode($body, true);
  if ($decoded === null && json_last_error() !== JSON_ERROR_NONE) {
    return new WP_REST_Response(['detail' => 'Invalid response from backend'], 502);
  }

  return new WP_REST_Response($decoded, $status ?: 200);
}

function siddh_guide_chat_history_proxy(WP_REST_Request $request) {
  $session_id = trim((string) $request->get_param('session_id'));

  if ($session_id === '') {
    return new WP_REST_Response(['detail' => 'session_id is required'], 400);
  }

  $response = wp_remote_get(
    siddh_guide_chat_backend_base_url() . '/history/' . rawurlencode($session_id),
    [
      'headers' => [
        'Accept'    => 'application/json',
        'x-api-key' => siddh_guide_chat_api_key(),
      ],
      'timeout' => 20,
    ]
  );

  if (is_wp_error($response)) {
    return new WP_REST_Response(['detail' => 'Proxy request failed'], 500);
  }

  $status = wp_remote_retrieve_response_code($response);
  $body   = wp_remote_retrieve_body($response);

  $decoded = json_decode($body, true);
  if ($decoded === null && json_last_error() !== JSON_ERROR_NONE) {
    return new WP_REST_Response(['detail' => 'Invalid response from backend'], 502);
  }

  return new WP_REST_Response($decoded, $status ?: 200);
}

/**
 * Enqueue assets
 */
function siddh_guide_chat_enqueue_assets() {
  wp_enqueue_style(
    'siddh-guide-chat-style',
    SIDDH_GUIDE_CHAT_PLUGIN_URL . 'assets/style.css',
    [],
    null
  );

  wp_enqueue_script(
    'siddh-guide-chat-widget',
    SIDDH_GUIDE_CHAT_PLUGIN_URL . 'assets/siddh-guide-chat-widget.js',
    [],
    null,
    true
  );
}
add_action('wp_enqueue_scripts', 'siddh_guide_chat_enqueue_assets');

/**
 * Change asset version query param from ?ver= to ?v=
 */
function siddh_guide_chat_change_asset_version_param($src, $handle) {
  $handles = ['siddh-guide-chat-widget', 'siddh-guide-chat-style'];

  if (in_array($handle, $handles, true)) {
    $src = remove_query_arg('ver', $src);
    $src = add_query_arg('v', '1.0.0', $src);
  }

  return $src;
}
add_filter('script_loader_src', 'siddh_guide_chat_change_asset_version_param', 10, 2);
add_filter('style_loader_src', 'siddh_guide_chat_change_asset_version_param', 10, 2);

/**
 * Shortcode renderer
 */
function siddh_guide_chat_shortcode() {
  $logo_url = esc_url(SIDDH_GUIDE_CHAT_PLUGIN_URL . 'assets/siddhanta-logo.png');

  ob_start();
  ?>
  <div class="proactive-message" id="proactiveMessage"></div>

  <button class="chat-toggle-button" id="chatToggleButton" aria-label="Open chat">
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 16 16">
      <path d="M2.678 11.894a1 1 0 0 1 .287.801 11 11 0 0 1-.398 2c1.395-.323 2.247-.697 2.634-.893a1 1 0 0 1 .71-.074A8 8 0 0 0 8 14c3.996 0 7-2.807 7-6s-3.004-6-7-6-7 2.808-7 6c0 1.468.617 2.83 1.678 3.894m-.493 3.905a22 22 0 0 1-.713.129c-.2.032-.352-.176-.273-.362a10 10 0 0 0 .244-.637l.003-.01c.248-.72.45-1.548.524-2.319C.743 11.37 0 9.76 0 8c0-3.866 3.582-7 8-7s8 3.134 8 7-3.582 7-8 7a9 9 0 0 1-2.347-.306c-.52.263-1.639.742-3.468 1.105"/>
    </svg>
  </button>

  <div class="chat-window" id="chatWindow">
    <div class="chat-header">
      <img src="<?php echo $logo_url; ?>" alt="Siddhanta Logo">
      <span>Siddh Guide Assistant</span>
    </div>

    <div class="chat-messages" id="chatMessages"></div>

    <form class="chat-input-form" id="chatForm">
      <input type="text" id="userInput" placeholder="Ask a question..." autocomplete="off">
      <button type="submit" aria-label="Send message">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
        </svg>
      </button>
    </form>
  </div>
  <?php
  return ob_get_clean();
}
add_shortcode('siddh_guide_chat', 'siddh_guide_chat_shortcode');