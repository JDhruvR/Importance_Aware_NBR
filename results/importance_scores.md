Output of Compute Importance - 
--- alpha_idf statistics ---
Mean: 1.116541
Std:  0.510633
25th: 0.807086
50th: 1.021458
75th: 1.298054

--- raw_importance (delta_bar) statistics ---
Mean: 0.106674
Std:  0.045278
25th: 0.080031
50th: 0.100237
75th: 0.123701

--- idf_factor statistics ---
Mean: 10.598360
Std:  1.853930
25th: 9.374300
50th: 10.831309
75th: 12.058539

Output of Check Importance - 

────────────────────────────────────────────────────────────
  Check 1 — File structure
────────────────────────────────────────────────────────────
  ✓ PASS: File exists: /home/dhruv/Qwen-VLA/sandbox/Importance_Aware_NBR/data/processed/instacart/importance_scores.npz
  ✓ PASS: All expected keys present: ['alpha_idf', 'idf_factor', 'raw_importance']
  ✓ PASS: All arrays have consistent shape: (47975,)
  ✓ PASS: Arrays are 1-D with 47975 items
  ✓ PASS:   alpha_idf: dtype=float32
  ✓ PASS:   raw_importance: dtype=float32
  ✓ PASS:   idf_factor: dtype=float32

────────────────────────────────────────────────────────────
  Check 2 — Value ranges
────────────────────────────────────────────────────────────
  ✓ PASS: No NaN, Inf, or negative values in any array
  ✓ PASS: raw_importance max = 0.804409 (≤ 1.0 as expected)

────────────────────────────────────────────────────────────
  Check 3 — Consistency with dataset
────────────────────────────────────────────────────────────
  ✓ PASS: Score array covers all items: 47975 slots ≥ 47975 unique items in baskets
  ✓ PASS: Coverage: 47969/47975 items have non-zero α_idf (100.0%)
  ✓ PASS: α_idf ≈ raw × idf (max absolute diff = 0.00e+00)
  ✓ PASS: Zero patterns are consistent across arrays

────────────────────────────────────────────────────────────
  Check 4 — Distribution statistics
────────────────────────────────────────────────────────────

  alpha_idf (n=47969 non-zero / 47975 total)
    Mean:  1.116541
    Std:   0.510633
    Min:   0.158184
     5th:  0.537619
    25th:  0.807086
    50th:  1.021458
    75th:  1.298054
    95th:  2.011705
    Max:   10.863911

  raw_importance (n=47969 non-zero / 47975 total)
    Mean:  0.106674
    Std:   0.045278
    Min:   0.011986
     5th:  0.048815
    25th:  0.080031
    50th:  0.100237
    75th:  0.123701
    95th:  0.185110
    Max:   0.804409

  idf_factor (n=47969 non-zero / 47975 total)
    Mean:  10.598360
    Std:   1.853930
    Min:   1.914454
     5th:  7.216114
    25th:  9.374300
    50th:  10.831309
    75th:  12.058539
    95th:  13.099993
    Max:   14.891752
  ✓ PASS: α_idf is right-skewed (mean=1.1165 > median=1.0215) — expected

────────────────────────────────────────────────────────────
  Check 5 — Top 30 and bottom 10 items
────────────────────────────────────────────────────────────

  Top 30 items by α_idf:
  Rank  ID      α_idf      raw_Δ      IDF      Name
  ───────────────────────────────────────────────────────────────────────────
  1     29965   10.8639    0.8044     13.51    Dry Ice
  2     13083   9.3006     0.7713     12.06    California Champagne
  3     932     8.2796     0.6462     12.81    Blue Label Year of the Ram
  4     5048    7.9969     0.5921     13.51    Elastic Bandage With Clips for Customized Compression
  5     10403   7.4472     0.6602     11.28    Organic Raspberry Black Tea
  6     10850   7.2600     0.5890     12.33    Brut Premier
  7     1680    7.2239     0.5349     13.51    Absolutely Ageless Restorative Night Cream
  8     35591   6.9938     0.6544     10.69    Laxative Tablets
  9     40559   6.8967     0.5559     12.41    Milk Chocolate Mini Chocolate Candies
  10    16698   6.8810     0.6156     11.18    Infant Formula With Iron
  11    20053   6.4728     0.5000     12.95    Chocolate Brownie Ice Cream Cake
  12    27449   6.3865     0.4933     12.95    Chocolate Truffle Cake
  13    29325   6.2235     0.4512     13.79    Black Elderberry Dietary Supplement Syrup Original Formula
  14    47729   6.1523     0.4887     12.59    Sweep & Vac Floor Vacuum Starter Kit
  15    17295   6.1163     0.4605     13.28    La Grand Dame Brut Champagne
  16    23588   5.9813     0.5289     11.31    Tropical Turmeric Jun-Kombucha Made With Honey Organic
  17    47776   5.9559     0.4832     12.33    The Original Celebration Ice Cream Cake
  18    33072   5.9549     0.4483     13.28    Hearts of Palm, Sliced
  19    22113   5.9459     0.4311     13.79    Natural Champagne
  20    6010    5.9234     0.4460     13.28    Fireworks Original Scent In Wash Laundry Booster Beads
  21    4950    5.9085     0.5212     11.34    Soy Powder Infant Formula
  22    40649   5.7118     0.4458     12.81    Expert Care Neosure Infant Formula Powder
  23    28784   5.6535     0.4665     12.12    White & Blue Thermoscan Ear Thermometer
  24    42301   5.6526     0.4752     11.90    Birthday Cake Ice Cream
  25    9385    5.6050     0.4802     11.67    Sorta Sweet Straight Up Iced Tea
  26    10534   5.5894     0.4052     13.79    Big Sur Woods airEffects Air Freshener
  27    8874    5.5848     0.4205     13.28    Kinderhook Creek Mini a Pure Sheep's Milk Cheese
  28    17382   5.5643     0.4978     11.18    UltraGel Personal Lubricant
  29    34336   5.5492     0.4473     12.41    Citron Vodka
  30    42442   5.5470     0.4440     12.49    Blanc De Noirs Sparkling Wine

  Bottom 10 items by α_idf (non-zero):
  Rank  ID      α_idf      raw_Δ      IDF      Name
  ───────────────────────────────────────────────────────────────────────────
  1     11979   0.1582     0.0636     2.49     Organic Strawberries
  2     40352   0.1619     0.0120     13.51    Alive! Once Daily Men's 50+ Ultra Potency Multivitamin
  3     20965   0.1628     0.0123     13.28    Chocolate Sea Salt Stars Shortbread Cookies
  4     12808   0.1674     0.0647     2.59     Organic Baby Spinach
  5     3855    0.1719     0.0125     13.79    Good Friends Cereal
  6     17429   0.1803     0.0576     3.13     Limes
  7     32955   0.1851     0.0508     3.65     Organic Grape Tomatoes
  8     40377   0.1852     0.0603     3.07     Large Lemon
  9     35075   0.1854     0.0143     12.95    Light 50 Cranberry
  10    47580   0.1883     0.0142     13.28    Yolk Free Medium Noodles

────────────────────────────────────────────────────────────
  Check 6 — Component correlations
────────────────────────────────────────────────────────────
  Pearson correlations (non-zero items only, n=47969):
    raw_importance ↔ idf_factor : -0.1672
    raw_importance ↔ alpha_idf  : +0.9210
    idf_factor     ↔ alpha_idf  : +0.1970

────────────────────────────────────────────────────────────
  Validation complete.
────────────────────────────────────────────────────────────
