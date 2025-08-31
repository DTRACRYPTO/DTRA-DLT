
<?php get_header(); ?>
<div class="container">
  <header class="header">
    <div>
      <h1>DTRA — Private Data Currency</h1>
      <div class="subtle">Fast • Private • Retail‑first — Smiles for all ICOs 😄</div>
    </div>
    <div class="nav">
      <a href="<?php echo esc_url(home_url('/')); ?>">Home</a>
      <a href="<?php echo esc_url(site_url('/buy-dtra')); ?>">Buy DTRA</a>
    </div>
  </header>

  <section class="hero card">
    <h2>Turn your crypto into DTRA</h2>
    <p class="subtle">We accept BTC, ETH, and HBAR. Fixed rate: $100 → 1 DTRA. Instant, private receipts.</p>
    <p><?php dtra_button(site_url('/buy-dtra'), 'Buy DTRA'); ?></p>
  </section>

  <section style="margin-top:22px" class="card">
    <h3 class="center">Marketplace (Demo)</h3>
    <?php echo do_shortcode('[dtra_marketplace]'); ?>
  </section>

  <footer class="footer center">
    © <?php echo date('Y'); ?> DTRA • Built with love for the crypto community
  </footer>
</div>
<?php get_footer(); ?>
