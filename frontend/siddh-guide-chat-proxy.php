<?php
/**
 * Plugin Name: Siddh Guide Chat Proxy
 * Description: Secure proxy endpoint for Siddh Guide chatbot (WordPress -> AWS App Runner)
 * Version: 1.1
 */

add_action('rest_api_init', function () {
  register_rest_route('siddh/v1', '/chat', [
    'methods'  => 'POST',
    'callback' => 'siddh_guide_chat_proxy',
    'permission_callback' => '__return_true',
  ]);
});

function siddh_guide_chat_proxy(WP_REST_Request $request) {
  // ---------------------------------------------------------------------------
  // 1) Basic Origin Protection (prevents random external websites from abusing it)
  // ---------------------------------------------------------------------------
  $origin = isset($_SERVER['HTTP_ORIGIN']) ? $_SERVER['HTTP_ORIGIN'] : '';

  // Allow only your WP domains
  $allowed_origins = [
    'https://siddhantaknowledge.org',
    'https://www.siddhantaknowledge.org',
  ];

  // If Origin header exists and it's not allowed -> block
  if ($origin && !in_array($origin, $allowed_origins, true)) {
    return new WP_REST_Response(['detail' => 'Forbidden origin'], 403);
  }

  // ---------------------------------------------------------------------------
  // 2) Backend App Runner endpoint
  // ---------------------------------------------------------------------------
  $api_url = "https://nyrqbf2z3k.ap-south-1.awsapprunner.com/chat";

  // ---------------------------------------------------------------------------
  // 3) API Key (server-side only)
  // ---------------------------------------------------------------------------
  $api_key = defined('SIDDH_CHAT_API_KEY') ? SIDDH_CHAT_API_KEY : '';

  if (!$api_key) {
    return new WP_REST_Response(['detail' => 'Server API key is missing'], 500);
  }

  // ---------------------------------------------------------------------------
  // 4) Build payload from WP request
  // ---------------------------------------------------------------------------
  $payload = [
    'message'    => (string) $request->get_param('message'),
    'session_id' => (string) ($request->get_param('session_id') ?: 'default'),
  ];

  // Optional: reject empty message early
  if (!trim($payload['message'])) {
    return new WP_REST_Response(['detail' => 'message is required'], 400);
  }

  // ---------------------------------------------------------------------------
  // 5) Forward request to AWS backend
  // ---------------------------------------------------------------------------
  $response = wp_remote_post($api_url, [
    'headers' => [
      'Content-Type' => 'application/json',
      'x-api-key'    => $api_key,
    ],
    'body'    => wp_json_encode($payload),
    'timeout' => 20,
  ]);

  if (is_wp_error($response)) {
    return new WP_REST_Response(['detail' => 'Proxy request failed'], 500);
  }

  $status = wp_remote_retrieve_response_code($response);
  $body   = wp_remote_retrieve_body($response);

  // If backend returns non-JSON (rare), still avoid breaking WP
  $decoded = json_decode($body, true);
  if ($decoded === null && json_last_error() !== JSON_ERROR_NONE) {
    return new WP_REST_Response(['detail' => 'Invalid response from backend'], 502);
  }

  return new WP_REST_Response($decoded, $status ?: 200);
}
