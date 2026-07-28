<?php
/**
 * Plugin Name: Ask Sid
 * Description: Ask Sid chatbot widget with secure WordPress proxy.
 * Version: 1.1.0
 */

if (!defined('ABSPATH')) {
  exit;
}

define('SIDDH_GUIDE_CHAT_PLUGIN_URL', plugin_dir_url(__FILE__));
define('SIDDH_GUIDE_CHAT_PLUGIN_PATH', plugin_dir_path(__FILE__));

/**
 * Backend base URL — resolved dynamically (no hardcoded secrets in source):
 *   1. wp-config constant  SIDDH_CHAT_BACKEND_URL   (recommended for prod)
 *   2. WordPress option    siddh_chat_backend_url   (set via Settings screen)
 *   3. Sensible default    (a public URL, not a secret)
 */
function siddh_guide_chat_backend_base_url() {
  if (defined('SIDDH_CHAT_BACKEND_URL') && SIDDH_CHAT_BACKEND_URL) {
    return untrailingslashit(SIDDH_CHAT_BACKEND_URL);
  }

  $opt = trim((string) get_option('siddh_chat_backend_url', ''));
  if ($opt !== '') {
    return untrailingslashit($opt);
  }

  return 'https://nyrqbf2z3k.ap-south-1.awsapprunner.com';
}

/**
 * API key — resolved dynamically, never hardcoded:
 *   1. wp-config constant  SIDDH_CHAT_API_KEY   (recommended for prod)
 *   2. WordPress option    siddh_chat_api_key   (set via Settings screen)
 * Returns '' when unset so the UI can warn instead of leaking a default.
 */
function siddh_guide_chat_api_key() {
  if (defined('SIDDH_CHAT_API_KEY') && SIDDH_CHAT_API_KEY) {
    return (string) SIDDH_CHAT_API_KEY;
  }

  return (string) get_option('siddh_chat_api_key', '');
}

/** True when a key is configured by either method. */
function siddh_guide_chat_is_configured() {
  return siddh_guide_chat_api_key() !== '';
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
/**
 * Cache-busting version for an asset: the file's own modification time.
 *
 * A hardcoded version string means an updated widget can sit behind WordPress,
 * WP Rocket, or Cloudflare caches indefinitely — visitors keep running the old
 * script. Since the session id now lives only in page memory, a stale copy is
 * not cosmetic: it would keep the previous sessionStorage behaviour alive and
 * silently break the refresh-isolation guarantee. Keying on filemtime means the
 * URL changes the moment the file does.
 */
function siddh_guide_chat_asset_version($relative_path) {
  $full_path = SIDDH_GUIDE_CHAT_PLUGIN_PATH . $relative_path;
  $mtime = @filemtime($full_path);

  return $mtime ? (string) $mtime : '1.0.0';
}

function siddh_guide_chat_enqueue_assets() {
  wp_enqueue_style(
    'siddh-guide-chat-style',
    SIDDH_GUIDE_CHAT_PLUGIN_URL . 'assets/style.css',
    [],
    siddh_guide_chat_asset_version('assets/style.css')
  );

  wp_enqueue_script(
    'siddh-guide-chat-widget',
    SIDDH_GUIDE_CHAT_PLUGIN_URL . 'assets/siddh-guide-chat-widget.js',
    [],
    siddh_guide_chat_asset_version('assets/siddh-guide-chat-widget.js'),
    true
  );
}
add_action('wp_enqueue_scripts', 'siddh_guide_chat_enqueue_assets');

/**
 * Change asset version query param from ?ver= to ?v=
 */
function siddh_guide_chat_change_asset_version_param($src, $handle) {
  $map = [
    'siddh-guide-chat-widget' => 'assets/siddh-guide-chat-widget.js',
    'siddh-guide-chat-style'  => 'assets/style.css',
  ];

  if (isset($map[$handle])) {
    // Carry the real file version across, otherwise this filter would undo the
    // cache-busting done at enqueue time and pin every visitor to one URL.
    $version = siddh_guide_chat_asset_version($map[$handle]);
    $src = remove_query_arg('ver', $src);
    $src = remove_query_arg('v', $src);
    $src = add_query_arg('v', $version, $src);
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
      <span>Ask Sid</span>
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

/**
 * ---------------------------------------------------------------------------
 * Knowledge-base refresh: manual button (admin) + automatic on publish.
 *
 * When content is added/edited on the website, the chatbot's knowledge base
 * must be re-read from the live site so it can answer about the new content.
 * The backend already exposes POST /admin/refresh for exactly this. These
 * hooks call it securely (API key stays server-side).
 * ---------------------------------------------------------------------------
 */

/**
 * Server-side call to the backend refresh endpoint.
 * @param bool $wait  When true, block until the crawl+re-embed finishes.
 */
function siddh_guide_chat_trigger_refresh($wait = false) {
  $url = siddh_guide_chat_backend_base_url() . '/admin/refresh' . ($wait ? '?wait=1' : '');

  $response = wp_remote_post($url, [
    'headers'  => [
      'Accept'    => 'application/json',
      'x-api-key' => siddh_guide_chat_api_key(),
    ],
    'timeout'  => $wait ? 120 : 20,
    'blocking' => (bool) $wait,
  ]);

  if (is_wp_error($response)) {
    return ['ok' => false, 'error' => $response->get_error_message()];
  }

  return [
    'ok'   => true,
    'code' => wp_remote_retrieve_response_code($response),
    'body' => wp_remote_retrieve_body($response),
  ];
}

/**
 * Admin menu: "Ask Sid KB" with a one-click update button.
 */
add_action('admin_menu', function () {
  add_menu_page(
    'Ask Sid KB',
    'Ask Sid KB',
    'manage_options',
    'siddh-guide-kb',
    'siddh_guide_chat_admin_page',
    'dashicons-update',
    80
  );
});

function siddh_guide_chat_admin_page() {
  if (!current_user_can('manage_options')) {
    return;
  }

  $notice = '';

  // Save settings (only options that are not locked by a wp-config constant).
  if (isset($_POST['siddh_save_settings']) && check_admin_referer('siddh_settings_action')) {
    if (!defined('SIDDH_CHAT_BACKEND_URL')) {
      update_option('siddh_chat_backend_url', esc_url_raw(trim((string) ($_POST['siddh_backend_url'] ?? ''))));
    }
    // Only overwrite the key when a new value is typed; blank keeps the current one.
    if (!defined('SIDDH_CHAT_API_KEY')) {
      $new_key = sanitize_text_field(trim((string) ($_POST['siddh_api_key'] ?? '')));
      if ($new_key !== '') {
        update_option('siddh_chat_api_key', $new_key);
      }
    }
    $notice = '<div class="notice notice-success"><p><strong>✅ Settings saved.</strong></p></div>';
  }

  // Trigger a manual refresh.
  if (isset($_POST['siddh_refresh_kb']) && check_admin_referer('siddh_refresh_kb_action')) {
    if (!siddh_guide_chat_is_configured()) {
      $notice = '<div class="notice notice-error"><p><strong>❌ Set the API key first</strong> (below) before updating.</p></div>';
    } else {
      $res = siddh_guide_chat_trigger_refresh(false);
      if (!empty($res['ok'])) {
        $notice = '<div class="notice notice-success"><p><strong>✅ Knowledge base update started.</strong> '
          . 'The chatbot will answer about the new website content in about a minute. '
          . 'Refresh a chat and try a question about the new content.</p></div>';
      } else {
        $notice = '<div class="notice notice-error"><p><strong>❌ Could not start the update:</strong> '
          . esc_html($res['error'] ?? 'unknown error') . '</p></div>';
      }
    }
  }

  $url_locked = defined('SIDDH_CHAT_BACKEND_URL');
  $key_locked = defined('SIDDH_CHAT_API_KEY');
  $cur_url    = esc_attr(siddh_guide_chat_backend_base_url());
  $configured = siddh_guide_chat_is_configured();

  echo '<div class="wrap">';
  echo '<h1>Ask Sid — Chatbot Knowledge Base</h1>';
  echo $notice;

  // Status banner.
  echo $configured
    ? '<p><span style="color:#0a0">●</span> API key configured &nbsp; | &nbsp; Backend: <code>' . $cur_url . '</code></p>'
    : '<div class="notice notice-warning"><p><strong>⚠️ API key not set.</strong> Enter it in Settings below, '
      . 'or add <code>define(\'SIDDH_CHAT_API_KEY\', \'…\');</code> to wp-config.php.</p></div>';

  // Refresh button.
  echo '<h2>Update knowledge base</h2>';
  echo '<p>Click after you add or edit courses/pages on the website. It re-reads the live site '
     . 'and updates the chatbot so it can answer about the new content.</p>';
  echo '<form method="post">';
  wp_nonce_field('siddh_refresh_kb_action');
  echo '<p><button type="submit" name="siddh_refresh_kb" value="1" class="button button-primary button-hero"'
     . ($configured ? '' : ' disabled') . '>🔄 Update Chatbot Knowledge Now</button></p>';
  echo '</form>';
  echo '<p class="description">The chatbot also refreshes automatically once a day, and ~90 seconds '
     . 'after you <strong>publish or update</strong> a course, page, or post.</p>';

  // Settings form.
  echo '<hr><h2>Settings</h2>';
  echo '<form method="post">';
  wp_nonce_field('siddh_settings_action');
  echo '<table class="form-table" role="presentation"><tbody>';

  echo '<tr><th scope="row"><label for="siddh_backend_url">Backend URL</label></th><td>';
  if ($url_locked) {
    echo '<code>' . $cur_url . '</code> <span class="description">(locked by wp-config constant)</span>';
  } else {
    echo '<input name="siddh_backend_url" id="siddh_backend_url" type="url" class="regular-text" value="'
       . esc_attr(get_option('siddh_chat_backend_url', '')) . '" placeholder="https://your-backend.awsapprunner.com">';
  }
  echo '</td></tr>';

  echo '<tr><th scope="row"><label for="siddh_api_key">API key</label></th><td>';
  if ($key_locked) {
    echo '<span class="description">Set by wp-config constant <code>SIDDH_CHAT_API_KEY</code> (recommended).</span>';
  } else {
    $has_opt = get_option('siddh_chat_api_key', '') !== '';
    echo '<input name="siddh_api_key" id="siddh_api_key" type="password" class="regular-text" value="" '
       . 'placeholder="' . ($has_opt ? '•••••• (leave blank to keep current)' : 'paste API key') . '" autocomplete="off">';
    echo '<p class="description">Stored server-side. For best security, use the wp-config constant instead.</p>';
  }
  echo '</td></tr>';

  echo '</tbody></table>';
  if (!$url_locked || !$key_locked) {
    echo '<p><button type="submit" name="siddh_save_settings" value="1" class="button button-secondary">Save settings</button></p>';
  }
  echo '</form>';
  echo '</div>';
}

/**
 * Automatic refresh shortly after any course/page/post is published or updated.
 * Debounced through wp-cron so rapid edits trigger only one refresh.
 */
add_action('transition_post_status', function ($new_status, $old_status, $post) {
  if ($new_status !== 'publish') {
    return;
  }

  $watched_types = ['post', 'page', 'product'];
  if (!in_array($post->post_type, $watched_types, true)) {
    return;
  }

  if (!wp_next_scheduled('siddh_guide_chat_auto_refresh_event')) {
    wp_schedule_single_event(time() + 90, 'siddh_guide_chat_auto_refresh_event');
  }
}, 10, 3);

add_action('siddh_guide_chat_auto_refresh_event', function () {
  siddh_guide_chat_trigger_refresh(false);
});