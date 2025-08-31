
<?php
/**
 * DTRA Dark theme setup
 */
add_action('after_setup_theme', function(){
  add_theme_support('title-tag');
  add_theme_support('post-thumbnails');
  register_nav_menus([ 'primary' => __('Primary Menu','dtra-dark') ]);
});

add_action('wp_enqueue_scripts', function(){
  wp_enqueue_style('dtra-dark', get_stylesheet_uri(), [], '1.0.0');
});

// Simple helper to render a button
function dtra_button($href, $label){
  echo '<a class="btn" href="'.esc_url($href).'">'.esc_html($label).'</a>';
}
