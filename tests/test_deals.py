import pytest

from lg_workflows.deals import stores
from lg_workflows.deals.config import Item, load_watchlist
from lg_workflows.deals.stores import (StoreError, parse_amazon_product, parse_amazon_search, parse_bestbuy_stock,
                                       parse_newegg_search, parse_price)
from lg_workflows.deals.watch import decide, format_price_alert, is_candidate

WATCH, GPU = load_watchlist()


@pytest.mark.parametrize("title,ok", [
    ("Apple Watch SE 3 (GPS + Cellular) 40mm Midnight Aluminum Case with Midnight Sport Band - S/M", True),
    ("Apple Watch SE 3 [GPS + Cellular 40mm] Smartwatch with Starlight Aluminum Case", True),
    ("Rogers Apple Watch SE 3 (GPS + Cellular) 40mm Midnight Aluminum Case - Monthly Financing", False),
    ("Refurbished (Excellent) - Apple Watch SE 3 (GPS + Cellular) 40mm Midnight Aluminum Case", False),
    ("Amazon Renewed Watch SE 3 [GPS + Cellular, 40mm] Starlight Aluminum Case", False),
    ("Apple Watch SE 3 (GPS + Cellular) 44mm Midnight Aluminum Case", False),
    ("Apple Watch SE 3 (GPS) 40mm Midnight Aluminum Case", False),
    ("Apple Watch SE (2nd Gen) [GPS + Cellular 40mm] Midnight Aluminum Case", False),
    ("Screen Protector Case Compatible with Apple Watch SE 3 40mm Cellular", False),
])
def test_watch_titles(title, ok):
    assert WATCH.title_matches(title) is ok


@pytest.mark.parametrize("title,price,ok", [
    ("GIGABYTE WindForce GeForce RTX 5070 Ti 16GB GDDR7 Graphics Card", 1799.99, True),
    ("ASUS TUF-RTX5070TI-O16G-GAMING", 1979.99, False),  # no "5070 Ti" phrase: model codes are not guessed
    ("MSI GeForce RTX 5070 12GB GDDR7 Video Card", 799.99, False),
    ("ASUS ROG Gaming Desktop PC, Core Ultra 7, RTX 5070 Ti", 2899.99, False),
    ("PNY GeForce RTX 5070 Ti 16GB", 159.99, False),
    ("PNY GeForce RTX 5070 Ti 16GB", None, False),
])
def test_gpu_matches(title, price, ok):
    assert GPU.matches(title, price) is ok


def test_out_of_stock_is_not_a_candidate():
    l = {"item": GPU.key, "title": "MSI GeForce RTX 5070 Ti Gaming Trio OC Plus 16GB", "price": 1189.95}
    assert is_candidate(GPU, l)
    assert not is_candidate(GPU, {**l, "in_stock": False})
    assert not is_candidate(WATCH, l)


def test_parse_price():
    assert parse_price("$1,849.99") == 1849.99
    assert parse_price("1,849") == 1849.0
    assert parse_price("") is None


NEWEGG = """<html><head><title>rtx | Newegg.ca</title></head><body>
<div class="menu"><div class="goods-container"><div class="goods-title">Header promo</div></div></div>
<div class="plp-results-section">
 <div data-itemnumber="14-932-769" class="goods-container"><a class="goods-title" href="https://www.newegg.ca/a/p/1">
   <span class="item-open-box-italic"></span> GIGABYTE Eagle RTX 5070 Ti 16GB</a>
   <div class="goods-price-current"><span class="goods-price-symbol">$</span>
   <span class="goods-price-value"><strong>1,849</strong><sup>.99</sup></span></div></div>
 <div data-itemnumber="14-126-756" class="goods-container"><a class="goods-title" href="https://www.newegg.ca/b/p/2">
   ASUS PRIME RTX 5070 Ti 16GB</a>
   <div class="goods-price-soldout">OUT OF STOCK Sold Price: $1,849.99</div></div>
</div></body></html>"""


def test_parse_newegg_skips_menus_and_reads_sold_out_as_unpriced():
    got = parse_newegg_search(NEWEGG)
    assert [(l["id"], l["price"]) for l in got] == [("14-932-769", 1849.99), ("14-126-756", None)]
    assert got[0]["title"] == "GIGABYTE Eagle RTX 5070 Ti 16GB"


def test_parse_newegg_cloudflare():
    with pytest.raises(StoreError):
        parse_newegg_search("<html><head><title>Just a moment...</title></head></html>")


AMAZON = """<html><body>
<div data-component-type="s-search-result" data-asin="B0FQFHDYTG"><h2 aria-label="Apple Watch SE 3 [GPS + Cellular 40mm]">
 <span>Apple Watch SE 3</span></h2></div>
<div data-component-type="s-search-result" data-asin="B0DVKG2PXW"><h2><span>PNY RTX 5070 Ti 16GB</span></h2>
 <span class="a-price a-text-price"><span class="a-offscreen">$2,199.99</span></span>
 <span class="a-price"><span class="a-offscreen">$1,899.97</span></span></div>
<div data-component-type="s-search-result" data-asin=""><h2>empty slot</h2></div>
</body></html>"""


def test_parse_amazon_search_takes_price_paid_not_list_price():
    got = parse_amazon_search(AMAZON)
    assert [(l["id"], l["title"], l["price"]) for l in got] == [
        ("B0FQFHDYTG", "Apple Watch SE 3 [GPS + Cellular 40mm]", None),
        ("B0DVKG2PXW", "PNY RTX 5070 Ti 16GB", 1899.97),
    ]
    assert got[1]["url"] == "https://www.amazon.ca/dp/B0DVKG2PXW"


def test_parse_amazon_captcha():
    with pytest.raises(StoreError):
        parse_amazon_search('<form action="/errors/validateCaptcha"></form>')


def test_parse_amazon_product():
    new = """<div id="corePriceDisplay_desktop_feature_div"><span class="a-price priceToPay">
      <span class="a-offscreen"> </span><span class="a-price-whole">379<span class="a-price-decimal">.</span></span>
      <span class="a-price-fraction">99</span></span></div>"""
    assert parse_amazon_product(new) == 379.99
    used_only = '<div id="usedOnlyBuybox">Buy Used: $379.99</div><div id="corePriceDisplay_desktop_feature_div"></div>'
    assert parse_amazon_product(used_only) is None


PAGE_TEXT = """Results
ASUS ROG Strix GeForce RTX 5070 12GB GDDR7 OC Edition Graphics Card
4.8 out of 5 stars
Price, product page
$1,551.62
$1,551
.
62
GIGABYTE WindForce GeForce RTX 5070 Ti 16GB GDDR7 Graphics Card
$1,799
.99
More options from $1,799.99 - $3,299.99
PNY GeForce RTX 5070 Ti 16GB
OUT OF STOCK
Sold Price: $1,849.99
ZOTAC Gaming GeForce RTX 5070 Ti Solid SFF
$2,119.99"""


@pytest.mark.parametrize("title,price,expected", [
    ("GIGABYTE WindForce GeForce RTX 5070 Ti 16GB GDDR7", 1799.99,
     ("GIGABYTE WindForce GeForce RTX 5070 Ti 16GB GDDR7 Graphics Card", 1799.99)),
    # cents dropped by the model
    ("GIGABYTE WindForce GeForce RTX 5070 Ti", 1799.0, ("GIGABYTE WindForce GeForce RTX 5070 Ti 16GB GDDR7 Graphics Card", 1799.99)),
    # misread title: the page's own line wins, so it no longer looks like a 5070 Ti
    ("ASUS ROG Strix GeForce RTX 5070 Ti 16GB GDDR7", 1551.62,
     ("ASUS ROG Strix GeForce RTX 5070 12GB GDDR7 OC Edition Graphics Card", 1551.62)),
    ("GIGABYTE WindForce GeForce RTX 5070 Ti 16GB", 1299.0, None),  # price of nothing near that title
    ("MSI Gefore RTX 5070 Ti 16 GB Graphics Card", 1399.0, None),  # invented listing
    ("PNY GeForce RTX 5070 Ti 16GB", None, ("PNY GeForce RTX 5070 Ti 16GB", None)),
    # sold out: neither its old price nor the next card's counts
    ("PNY GeForce RTX 5070 Ti 16GB", 1849.99, ("PNY GeForce RTX 5070 Ti 16GB", None)),
    ("PNY GeForce RTX 5070 Ti 16GB", 2119.99, ("PNY GeForce RTX 5070 Ti 16GB", None)),
])
def test_ground_listing(title, price, expected):
    assert stores.ground_listing(title, price, PAGE_TEXT) == expected


def test_parse_bestbuy_stock_handles_bom():
    text = ('\ufeff{"availabilities":[{"sku":"1","shipping":{"purchasable":false},"pickup":{"purchasable":true}},'
            '{"sku":"2","shipping":{"purchasable":false},"pickup":{"purchasable":false}}]}')
    assert parse_bestbuy_stock(text.encode().decode("utf-8-sig")) == {"1": True, "2": False}


def test_amazon_details_only_for_matching_unpriced(monkeypatch):
    fetched = []
    listings = [{"title": "Apple Watch SE 3 [GPS + Cellular 40mm]", "price": None, "url": "u1"},
                {"title": "Apple Watch SE 3 [GPS 40mm]", "price": None, "url": "u2"}]
    stores._amazon_details(WATCH, listings, lambda url: fetched.append(url) or '<div id="usedOnlyBuybox"></div>')
    assert fetched == ["u1"] and listings[0]["price"] is None


def _listing(price, store="bestbuy", id_="1"):
    return {"item": WATCH.key, "store": store, "id": id_, "title": "Apple Watch SE 3 GPS + Cellular 40mm",
            "price": price, "url": f"https://x/{id_}", "method": "http"}


def test_decide_uses_watchlist_normal_until_history():
    assert decide(WATCH, [_listing(389.99)], [], None) is None  # 2.5% off 399.99
    a = decide(WATCH, [_listing(379.99)], [], None)
    assert a and a["normal"] == 399.99 and a["best"]["price"] == 379.99


def test_decide_history_median_wins():
    history = [349.99, 349.99, 349.99, 349.99]
    assert decide(WATCH, [_listing(379.99)], history, None) is None
    a = decide(WATCH, [_listing(329.0)], history, None)
    assert a and a["normal"] == 349.99 and a["low"] == 349.99


def test_decide_without_any_normal_price_needs_target():
    assert decide(GPU, [{**_listing(1200.0), "item": GPU.key}], [], None) is None
    item = Item(**{**GPU.__dict__, "target_price": 1500.0})
    assert decide(item, [{**_listing(1499.0), "item": GPU.key}], [], None)["hit_target"]


def test_decide_realert_only_when_cheaper():
    assert decide(WATCH, [_listing(369.99)], [], 369.99) is None
    assert decide(WATCH, [_listing(366.0)], [], 369.99) is not None


def test_format_price_alert_lists_other_stores():
    a = decide(WATCH, [_listing(379.99), _listing(389.0, "amazon", "2"), _listing(379.99)], [], None)
    title, body = format_price_alert(a)
    assert title == "降价提醒：Apple Watch SE 3 40mm GPS + Cellular"
    assert "$379.99 @ Best Buy（常价 $399.99，便宜 5%）" in body
    assert "Amazon $389.00" in body and body.count("https://x/1") == 1
