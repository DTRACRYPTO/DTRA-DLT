
<?php
/**
 * Plugin Name: DTRA Payments
 * Description: Accept BTC, ETH (via Coinbase Commerce) and HBAR for DTRA at a fixed USD rate. Provides [dtra_buy] and [dtra_marketplace] shortcodes.
 * Version: 0.1.1
 * Author: DTRA
 */

if (!defined('ABSPATH')) { exit; }

class DTRA_Payments {
  const OPT = 'dtra_payments_options';
  const CPT = 'dtra_order';
  const VERSION = '0.1.1';

  public function __construct(){
    add_action('init', [$this,'register_cpt']);
    add_action('admin_menu', [$this,'admin_menu']);
    add_action('admin_init', [$this,'register_settings']);
    add_shortcode('dtra_buy', [$this,'sc_buy']);
    add_shortcode('dtra_marketplace', [$this,'sc_marketplace']);
    add_action('init', [$this,'maybe_handle_coinbase_webhook']);
    add_action('rest_api_init', function(){
      register_rest_route('dtra/v1','/order-status/(?P<id>\d+)', [
        'methods'=>'GET',
        'callback'=>[$this,'rest_order_status'],
        'permission_callback'=>'__return_true'
      ]);
    });
    add_filter('cron_schedules', function($s){ $s['dtra_5min']=['interval'=>300,'display'=>'Every 5 Minutes']; return $s; });
    add_action('dtra_check_hbar', [$this,'cron_check_hbar']);
    if (!wp_next_scheduled('dtra_check_hbar')) {
      wp_schedule_event(time()+120, 'dtra_5min', 'dtra_check_hbar');
    }
  }

  // ---- Settings helpers ----
  public static function opts(){
    $defaults = [
      'usd_per_dtra' => 100,
      'coinbase_api_key' => '',
      'coinbase_webhook_secret' => '',
      'hedera_account' => '', // e.g. 0.0.12345
      'btc_address' => '',
      'eth_address' => ''
    ];
    return wp_parse_args(get_option(self::OPT, []), $defaults);
  }

  public function admin_menu(){
    add_options_page('DTRA Payments','DTRA Payments','manage_options','dtra-payments',[$this,'settings_page']);
  }

  public function register_settings(){
    register_setting(self::OPT, self::OPT);
    add_settings_section('main','Gateway Settings', function(){ echo '<p>Fill your keys for Coinbase Commerce and your Hedera account ID.</p>'; }, self::OPT);
    foreach (['usd_per_dtra','coinbase_api_key','coinbase_webhook_secret','hedera_account','btc_address','eth_address'] as $field){
      add_settings_field($field, ucwords(str_replace('_',' ',$field)), function() use ($field){
        $o = self::opts();
        $type = ($field==='usd_per_dtra')?'number':'text';
        $val = esc_attr($o[$field]);
        echo "<input type='$type' name='".self::OPT."[$field]' value='$val' class='regular-text'/>";
        if ($field==='usd_per_dtra') echo '<p class="description">$100 → 1 DTRA by default.</p>';
      }, self::OPT, 'main');
    }
  }

  public function settings_page(){
    echo '<div class="wrap"><h1>DTRA Payments</h1><form method="post" action="options.php">';
    settings_fields(self::OPT);
    do_settings_sections(self::OPT);
    submit_button();
    echo '<p><b>Webhook:</b> Set Coinbase Commerce webhook URL to <code>'.home_url('/?dtra_cc_webhook=1').'</code></p>';
    echo '</form></div>';
  }

  // ---- Custom Post Type for orders ----
  public function register_cpt(){
    register_post_type(self::CPT, [
      'label'=>'DTRA Orders','public'=>false,'show_ui'=>true,
      'supports'=>['title'],'menu_icon'=>'dashicons-tickets',
    ]);
  }

  private static function create_order($args){
    $order_id = wp_insert_post([
      'post_type'=>self::CPT,'post_status'=>'publish',
      'post_title'=>'Order '.date('Y-m-d H:i:s'),
    ]);
    foreach ($args as $k=>$v){ update_post_meta($order_id, $k, $v); }
    return $order_id;
  }

  public function rest_order_status($req){
    $id = intval($req['id']);
    $status = get_post_meta($id,'status',true) ?: 'pending';
    return ['id'=>$id,'status'=>$status];
  }

  // ---- Shortcodes ----
  public function sc_marketplace(){
    // Simple demo list; you can replace with products CPT later.
    $items = [
      ['id'=>'g_1','title'=>'Coffee','price'=>1.50],
      ['id'=>'g_2','title'=>'T-Shirt','price'=>12.00],
      ['id'=>'g_3','title'=>'Sticker Pack','price'=>0.75],
    ];
    ob_start();
    echo '<div class="market-grid">';
    foreach ($items as $it){
      echo '<div class="card"><div class="badge">Demo</div><h3>'.esc_html($it['title']).'</h3>';
      echo '<div class="price">'.$it['price'].' DTRA</div>';
      echo '<div class="small">ID: '.esc_html($it['id']).'</div>';
      echo '<p><a class="btn" href="'.esc_url(site_url('/buy-dtra')).'">Buy</a></p></div>';
    }
    echo '</div>';
    return ob_get_clean();
  }

  public function sc_buy(){
    $o = self::opts();
    $msg = '';
    if (!empty($_POST['dtra_buy_nonce']) && wp_verify_nonce($_POST['dtra_buy_nonce'],'dtra_buy')){
      $addr = sanitize_text_field($_POST['dtra_addr'] ?? '');
      $amount_dtra = floatval($_POST['dtra_amount'] ?? '0');
      $currency = sanitize_text_field($_POST['dtra_currency'] ?? 'BTC');
      $usd_total = floatval($o['usd_per_dtra']) * $amount_dtra;
      if ($amount_dtra<=0 || empty($addr)){
        $msg = '<div class="card" style="border-color:#ef4444">Please enter a valid amount and address.</div>';
      } else {
        if (in_array($currency,['BTC','ETH'])){
          $res = $this->coinbase_create_charge($usd_total, $currency, $amount_dtra, $addr);
          if ($res && !empty($res['data']['hosted_url'])){
            $charge_id = $res['data']['id'];
            $id = self::create_order([
              'status'=>'pending','currency'=>$currency,'amount_dtra'=>$amount_dtra,
              'usd_total'=>$usd_total,'addr'=>$addr,'charge_id'=>$charge_id
            ]);
            $url = esc_url($res['data']['hosted_url']);
            $msg = '<div class="card">Order #'.$id.' created. Redirecting… <script>location.href="'.$url.'";</script><p><a class="btn" href="'.$url.'">Continue to Coinbase</a></p></div>';
          } else {
            $msg = '<div class="card" style="border-color:#ef4444">Could not create Coinbase charge. Check API key.</div>';
          }
        } else { // HBAR
          $memo = 'DTRA#'.wp_generate_password(8,false,false);
          $price = $this->price_usd('hedera-hashgraph'); // per 1 coin in USD
          if ($price<=0){ $price = 0.1; } // fallback
          $hbar_amount = $usd_total / $price;
          $id = self::create_order([
            'status'=>'pending','currency'=>'HBAR','amount_dtra'=>$amount_dtra,'usd_total'=>$usd_total,
            'addr'=>$addr,'hbar_expected'=>$hbar_amount,'memo'=>$memo
          ]);
          $msg = '<div class="card"><h3>HBAR Payment Instructions</h3>
            <p>Send <b>'.number_format($hbar_amount,6).'</b> HBAR to <code>'.esc_html($o['hedera_account']).'</code> with memo <code>'.esc_html($memo).'</code>.</p>
            <p class="small">We’ll detect it automatically (checks every 5 min). Order #'.$id.'</p></div>';
        }
      }
    }

    ob_start(); ?>
    <form method="post" class="grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">
      <div><label>DTRA Amount</label><input name="dtra_amount" placeholder="1.5" required /></div>
      <div><label>Your DTRA Address</label><input name="dtra_addr" placeholder="0x…" required /></div>
      <div><label>Pay with</label>
        <select name="dtra_currency">
          <option>BTC</option><option>ETH</option><option>HBAR</option>
        </select>
      </div>
      <div><label>USD per DTRA</label><input value="<?php echo esc_attr($o['usd_per_dtra']); ?>" disabled /></div>
      <div><button class="btn">Buy DTRA</button></div>
      <?php wp_nonce_field('dtra_buy','dtra_buy_nonce'); ?>
    </form>
    <?php
      if ($msg) echo $msg;
      echo '<p class="small">Fixed rate: $'.esc_html($o['usd_per_dtra']).' → 1 DTRA. BTC/ETH via Coinbase Commerce; HBAR is direct transfer.</p>';
      return ob_get_clean();
  }

  // ---- Coinbase Commerce ----
  private function coinbase_create_charge($usd_total, $currency_label, $dtra_amount, $recv_addr){
    $o = self::opts();
    if (empty($o['coinbase_api_key'])) return false;
    $payload = [
      'name' => 'DTRA Purchase',
      'description' => 'DTRA: '.$dtra_amount.' to '.$recv_addr,
      'pricing_type' => 'fixed_price',
      'local_price' => ['amount'=>number_format($usd_total,2,'.',''),'currency'=>'USD'],
      'metadata' => ['recv_addr'=>$recv_addr, 'dtra'=>$dtra_amount],
      'redirect_url' => home_url('/buy-dtra'),
      'cancel_url' => home_url('/buy-dtra')
    ];
    $args = [
      'headers'=>[
        'X-CC-Api-Key'=>$o['coinbase_api_key'],
        'X-CC-Version'=>'2018-03-22',
        'Content-Type'=>'application/json'
      ],
      'body'=>json_encode($payload),
      'timeout'=>20
    ];
    $res = wp_remote_post('https://api.commerce.coinbase.com/charges', $args);
    if (is_wp_error($res)) return false;
    return json_decode(wp_remote_retrieve_body($res), true);
  }

  public function maybe_handle_coinbase_webhook(){
    if (!isset($_GET['dtra_cc_webhook'])) return;
    $o = self::opts();
    $raw = file_get_contents('php://input');
    $sig = $_SERVER['HTTP_X_CC_WEBHOOK_SIGNATURE'] ?? '';
    $computed = hash_hmac('sha256', $raw, $o['coinbase_webhook_secret']);
    if (!hash_equals($computed,$sig)){ status_header(401); exit('bad signature'); }
    $data = json_decode($raw, true);
    $event = $data['event']['type'] ?? '';
    $charge_id = $data['event']['data']['id'] ?? '';
    if ($event==='charge:confirmed' && $charge_id){
      $q = new WP_Query(['post_type'=>self::CPT,'meta_key'=>'charge_id','meta_value'=>$charge_id,'posts_per_page'=>1]);
      if ($q->have_posts()){ $q->the_post(); update_post_meta(get_the_ID(),'status','paid'); wp_reset_postdata(); }
    }
    echo 'ok'; exit;
  }

  // ---- Hedera Mirror Node polling ----
  public function cron_check_hbar(){
    $o = self::opts();
    if (empty($o['hedera_account'])) return;
    $q = new WP_Query(['post_type'=>self::CPT,'meta_key'=>'currency','meta_value'=>'HBAR','posts_per_page'=>50]);
    while($q->have_posts()){ $q->the_post();
      $id = get_the_ID();
      $status = get_post_meta($id,'status',true);
      if ($status==='paid') continue;
      $memo = get_post_meta($id,'memo',true);
      $expected = floatval(get_post_meta($id,'hbar_expected',true));
      if (!$memo || $expected<=0) continue;
      if ($this->hedera_seen_payment($o['hedera_account'], $memo, $expected)){
        update_post_meta($id,'status','paid');
      }
    }
    wp_reset_postdata();
  }

  private function hedera_seen_payment($accountId, $memo, $expectedHBAR){
    $url = 'https://mainnet-public.mirrornode.hedera.com/api/v1/transactions?account.id='.urlencode($accountId).'&limit=25&order=desc';
    $res = wp_remote_get($url, ['timeout'=>20]);
    if (is_wp_error($res)) return false;
    $body = json_decode(wp_remote_retrieve_body($res), true);
    if (empty($body['transactions'])) return false;
    $tiny_expected = $expectedHBAR * 100000000; // 1e8
    foreach ($body['transactions'] as $tx){
      $memo_b64 = $tx['memo_base64'] ?? '';
      $decoded = base64_decode($memo_b64);
      if (strpos($decoded, $memo) !== false){
        // crude check: look at the first transfer list entry to the account
        foreach (($tx['transfers'] ?? []) as $tr){
          if (($tr['account'] ?? '') === $accountId){
            $amt = intval($tr['amount'] ?? 0);
            if ($amt >= $tiny_expected * 0.98) return true; // allow 2% slippage
          }
        }
      }
    }
    return false;
  }

  private function price_usd($coingecko_id){
    $url = 'https://api.coingecko.com/api/v3/simple/price?ids='.$coingecko_id.'&vs_currencies=usd';
    $res = wp_remote_get($url, ['timeout'=>15]);
    if (is_wp_error($res)) return 0;
    $body = json_decode(wp_remote_retrieve_body($res), true);
    return floatval($body[$coingecko_id]['usd'] ?? 0);
  }
}
new DTRA_Payments();
