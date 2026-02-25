<?php
/**
 * Plugin Name: Siddh Guide Chat Proxy
 * Description: Secure proxy endpoint for Siddh Guide chatbot (WordPress -> AWS App Runner)
 * Version: 1.2
 */

add_action('rest_api_init', function () {
  register_rest_route('siddh/v1', '/chat', [
    'methods'  => 'POST',
    'callback' => 'siddh_guide_chat_proxy',
    'permission_callback' => '__return_true',
  ]);

  register_rest_route('siddh/v1', '/history/(?P<session_id>[a-zA-Z0-9\-\_]+)', [
    'methods'  => 'GET',
    'callback' => 'siddh_guide_chat_history_proxy',
    'permission_callback' => '__return_true',
  ]);
});

function siddh_guide_allowed_origin_or_forbidden() {
  $origin = isset($_SERVER['HTTP_ORIGIN']) ? $_SERVER['HTTP_ORIGIN'] : '';

  $allowed_origins = [
    'https://siddhantaknowledge.org',
    'https://www.siddhantaknowledge.org',
    // Uncomment for local testing:
    // 'http://localhost',
    // 'http://127.0.0.1',
  ];

  if ($origin && !in_array($origin, $allowed_origins, true)) {
    return new WP_REST_Response(['detail' => 'Forbidden origin'], 403);
  }

  return true;
}

function siddh_guide_chat_proxy(WP_REST_Request $request) {
  $ok = siddh_guide_allowed_origin_or_forbidden();
  if ($ok !== true) return $ok;

  $api_url_base = "https://nyrqbf2z3k.ap-south-1.awsapprunner.com";

  $payload = $request->get_json_params();
  if (!is_array($payload)) $payload = [];

  $response = wp_remote_post($api_url_base . "/chat", [
    'headers' => [
      'Content-Type' => 'application/json',
      'Accept' => 'application/json',
      'x-api-key' => (defined('SIDDH_CHAT_API_KEY') ? SIDDH_CHAT_API_KEY : ''),
    ],
    'body' => wp_json_encode($payload),
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
  $ok = siddh_guide_allowed_origin_or_forbidden();
  if ($ok !== true) return $ok;

  $api_url_base = "https://nyrqbf2z3k.ap-south-1.awsapprunner.com";
  $session_id = (string) $request->get_param('session_id');
  if (!$session_id) $session_id = "default";

  $response = wp_remote_get($api_url_base . "/history/" . rawurlencode($session_id), [
    'headers' => [
      'Accept' => 'application/json',
      'x-api-key' => (defined('SIDDH_CHAT_API_KEY') ? SIDDH_CHAT_API_KEY : ''),
    ],
    'timeout' => 20,
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