
<?php
/* Template Name: DTRA Buy Page */
get_header(); ?>
<div class="container">
  <header class="header">
    <h1>Buy DTRA</h1>
    <div class="nav"><a href="<?php echo esc_url(home_url('/')); ?>">Home</a></div>
  </header>
  <section class="card">
    <?php echo do_shortcode('[dtra_buy]'); ?>
  </section>
</div>
<?php get_footer(); ?>
