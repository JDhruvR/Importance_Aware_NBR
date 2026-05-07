Output of Compute Importance for Instacart - 
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

Output of Check Importance for Instacart - 

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

Output of Compute Importance for Dunnhumby - 
--- alpha_idf statistics ---
Mean: 1.149258
Std:  0.579086
25th: 0.807933
50th: 0.985545
75th: 1.283139

--- raw_importance (delta_bar) statistics ---
Mean: 0.159690
Std:  0.075830
25th: 0.115662
50th: 0.137090
75th: 0.174024

--- idf_factor statistics ---
Mean: 7.154244
Std:  0.842102
25th: 6.658957
50th: 7.292957
75th: 7.779310

Output of Check Importance for Dunnhumby -

────────────────────────────────────────────────────────────
  Check 1 — File structure
────────────────────────────────────────────────────────────
  ✓ PASS: File exists: /home/dhruv/Qwen-VLA/sandbox/Importance_Aware_NBR/data/processed/dunnhumby/importance_scores.npz
  ✓ PASS: All expected keys present: ['alpha_idf', 'idf_factor', 'raw_importance']
  ✓ PASS: All arrays have consistent shape: (4997,)
  ✓ PASS: Arrays are 1-D with 4997 items
  ✓ PASS:   alpha_idf: dtype=float32
  ✓ PASS:   raw_importance: dtype=float32
  ✓ PASS:   idf_factor: dtype=float32

────────────────────────────────────────────────────────────
  Check 2 — Value ranges
────────────────────────────────────────────────────────────
  ✓ PASS: No NaN, Inf, or negative values in any array
  ✓ PASS: raw_importance max = 0.987157 (≤ 1.0 as expected)

────────────────────────────────────────────────────────────
  Check 3 — Consistency with dataset
────────────────────────────────────────────────────────────
  ✓ PASS: Score array covers all items: 4997 slots ≥ 4997 unique items in baskets
  ✓ PASS: Coverage: 4997/4997 items have non-zero α_idf (100.0%)
  ✓ PASS: α_idf ≈ raw × idf (max absolute diff = 0.00e+00)
  ✓ PASS: Zero patterns are consistent across arrays

────────────────────────────────────────────────────────────
  Check 4 — Distribution statistics
────────────────────────────────────────────────────────────

  alpha_idf (n=4997 non-zero / 4997 total)
    Mean:  1.149258
    Std:   0.579086
    Min:   0.243102
     5th:  0.610320
    25th:  0.807933
    50th:  0.985545
    75th:  1.283139
    95th:  2.297984
    Max:   7.752421

  raw_importance (n=4997 non-zero / 4997 total)
    Mean:  0.159690
    Std:   0.075830
    Min:   0.068364
     5th:  0.095057
    25th:  0.115662
    50th:  0.137090
    75th:  0.174024
    95th:  0.317822
    Max:   0.987157

  idf_factor (n=4997 non-zero / 4997 total)
    Mean:  7.154244
    Std:   0.842102
    Min:   1.872807
     5th:  5.632869
    25th:  6.658957
    50th:  7.292957
    75th:  7.779310
    95th:  8.253329
    Max:   8.638747
  ✓ PASS: α_idf is right-skewed (mean=1.1493 > median=0.9855) — expected

────────────────────────────────────────────────────────────
  Check 5 — Top 30 and bottom 10 items
────────────────────────────────────────────────────────────

  Top 30 items by α_idf:
  Rank  ID      α_idf      raw_Δ      IDF      Name
  ───────────────────────────────────────────────────────────────────────────
  1     653     7.7524     0.9752     7.95     PRD0900654
  2     2610    7.3614     0.9872     7.46     PRD0902611
  3     2763    6.9162     0.9497     7.28     PRD0902764
  4     903     6.1501     0.8086     7.61     PRD0900904
  5     9       5.7939     0.7311     7.92     PRD0900010
  6     3326    5.3992     0.7974     6.77     PRD0903327
  7     4089    5.3163     0.6631     8.02     PRD0904090
  8     4262    5.0852     0.7156     7.11     PRD0904263
  9     3502    4.9795     0.6074     8.20     PRD0903503
  10    1074    4.4430     0.6055     7.34     PRD0901075
  11    2490    4.4016     0.5788     7.61     PRD0902491
  12    688     4.3872     0.5483     8.00     PRD0900689
  13    962     4.2699     0.6453     6.62     PRD0900963
  14    1200    4.2628     0.5637     7.56     PRD0901201
  15    1422    4.2524     0.5608     7.58     PRD0901423
  16    821     4.1701     0.5879     7.09     PRD0900822
  17    3788    4.1455     0.5754     7.20     PRD0903789
  18    3263    4.0841     0.5504     7.42     PRD0903264
  19    1075    4.0726     0.5437     7.49     PRD0901076
  20    2816    3.9988     0.6242     6.41     PRD0902817
  21    1961    3.9906     0.5336     7.48     PRD0901962
  22    112     3.9108     0.5667     6.90     PRD0900113
  23    4765    3.8985     0.5496     7.09     PRD0904766
  24    2147    3.8687     0.5926     6.53     PRD0902148
  25    2651    3.8420     0.4887     7.86     PRD0902652
  26    2736    3.8066     0.4948     7.69     PRD0902737
  27    4548    3.8059     0.4819     7.90     PRD0904549
  28    4684    3.7923     0.5134     7.39     PRD0904685
  29    820     3.7861     0.4608     8.22     PRD0900821
  30    4336    3.7854     0.5060     7.48     PRD0904337

  Bottom 10 items by α_idf (non-zero):
  Rank  ID      α_idf      raw_Δ      IDF      Name
  ───────────────────────────────────────────────────────────────────────────
  1     1264    0.2431     0.0858     2.83     PRD0901265
  2     829     0.3002     0.0962     3.12     PRD0900830
  3     3051    0.3109     0.1660     1.87     PRD0903052
  4     3080    0.3142     0.0732     4.29     PRD0903081
  5     3992    0.3266     0.0750     4.35     PRD0903993
  6     4975    0.3272     0.0947     3.46     PRD0904976
  7     172     0.3393     0.1012     3.35     PRD0900173
  8     4886    0.3444     0.0965     3.57     PRD0904887
  9     4249    0.3464     0.0889     3.90     PRD0904250
  10    3073    0.3475     0.0983     3.53     PRD0903074

────────────────────────────────────────────────────────────
  Check 6 — Component correlations
────────────────────────────────────────────────────────────
  Pearson correlations (non-zero items only, n=4997):
    raw_importance ↔ idf_factor : +0.1064
    raw_importance ↔ alpha_idf  : +0.9712
    idf_factor     ↔ alpha_idf  : +0.3207
  ✓ PASS: Both raw_importance and idf_factor contribute meaningfully to α_idf

────────────────────────────────────────────────────────────
  Validation complete.
────────────────────────────────────────────────────────────

Output of Compute Importance for Tafeng -

--- alpha_idf statistics ---
Mean: 1.673635
Std:  1.001201
25th: 1.039983
50th: 1.421600
75th: 1.993604

--- raw_importance (delta_bar) statistics ---
Mean: 0.187879
Std:  0.095500
25th: 0.127874
50th: 0.165226
75th: 0.220439

--- idf_factor statistics ---
Mean: 8.734407
Std:  1.223634
25th: 7.970380
50th: 8.908649
75th: 9.719580

Output of Check Importance for Tafeng -

────────────────────────────────────────────────────────────
  Check 1 — File structure
────────────────────────────────────────────────────────────
  ✓ PASS: File exists: /home/dhruv/Qwen-VLA/sandbox/Importance_Aware_NBR/data/processed/tafeng/importance_scores.npz
  ✓ PASS: All expected keys present: ['alpha_idf', 'idf_factor', 'raw_importance']
  ✓ PASS: All arrays have consistent shape: (15743,)
  ✓ PASS: Arrays are 1-D with 15743 items
  ✓ PASS:   alpha_idf: dtype=float32
  ✓ PASS:   raw_importance: dtype=float32
  ✓ PASS:   idf_factor: dtype=float32

────────────────────────────────────────────────────────────
  Check 2 — Value ranges
────────────────────────────────────────────────────────────
  ✓ PASS: No NaN, Inf, or negative values in any array
  ✓ PASS: raw_importance max = 1.000000 (≤ 1.0 as expected)

────────────────────────────────────────────────────────────
  Check 3 — Consistency with dataset
────────────────────────────────────────────────────────────
  ✓ PASS: Score array covers all items: 15743 slots ≥ 15743 unique items in baskets
  ✓ PASS: Coverage: 15624/15743 items have non-zero α_idf (99.2%)
  ✓ PASS: α_idf ≈ raw × idf (max absolute diff = 0.00e+00)
  ✓ PASS: Zero patterns are consistent across arrays

────────────────────────────────────────────────────────────
  Check 4 — Distribution statistics
────────────────────────────────────────────────────────────

  alpha_idf (n=15624 non-zero / 15743 total)
    Mean:  1.673635
    Std:   1.001201
    Min:   0.303290
     5th:  0.695685
    25th:  1.039983
    50th:  1.421600
    75th:  1.993604
    95th:  3.481934
    Max:   11.105874

  raw_importance (n=15624 non-zero / 15743 total)
    Mean:  0.187879
    Std:   0.095500
    Min:   0.027746
     5th:  0.088196
    25th:  0.127874
    50th:  0.165226
    75th:  0.220439
    95th:  0.361708
    Max:   1.000000

  idf_factor (n=15624 non-zero / 15743 total)
    Mean:  8.734407
    Std:   1.223634
    Min:   2.782508
     5th:  6.480901
    25th:  7.970380
    50th:  8.908649
    75th:  9.719580
    95th:  10.412727
    Max:   11.105874
  ✓ PASS: α_idf is right-skewed (mean=1.6736 > median=1.4216) — expected

────────────────────────────────────────────────────────────
  Check 5 — Top 30 and bottom 10 items
────────────────────────────────────────────────────────────

  Top 30 items by α_idf:
  Rank  ID      α_idf      raw_Δ      IDF      Name
  ───────────────────────────────────────────────────────────────────────────
  1     13394   11.1059    1.0000     11.11    300302
  2     14995   11.1059    1.0000     11.11    510512
  3     7321    11.1059    1.0000     11.11    300711
  4     388     11.1059    1.0000     11.11    320402
  5     6986    11.1059    1.0000     11.11    730603
  6     8185    11.1059    1.0000     11.11    760157
  7     9111    11.1059    1.0000     11.11    300115
  8     9301    11.1059    1.0000     11.11    110333
  9     12430   11.1059    1.0000     11.11    760687
  10    2311    11.1059    1.0000     11.11    520422
  11    10617   11.1059    1.0000     11.11    780405
  12    11132   11.1059    1.0000     11.11    730303
  13    12344   11.1059    1.0000     11.11    760950
  14    1137    10.4127    1.0000     10.41    510411
  15    12981   10.4127    1.0000     10.41    720506
  16    9376    10.4127    1.0000     10.41    530116
  17    12613   10.4127    1.0000     10.41    730718
  18    13164   9.4684     0.9093     10.41    720515
  19    9306    9.2484     0.8882     10.41    100322
  20    1276    9.1636     0.8800     10.41    751101
  21    1302    9.1344     0.9128     10.01    711702
  22    12169   9.0929     0.8187     11.11    730703
  23    10364   8.8220     0.8816     10.01    780202
  24    10195   8.8097     0.8803     10.01    100109
  25    7695    8.7784     0.7904     11.11    560349
  26    10492   8.6452     0.7784     11.11    730103
  27    8112    8.5900     0.8250     10.41    730145
  28    14634   8.5587     0.7706     11.11    300416
  29    12884   8.5045     0.7658     11.11    100605
  30    179     8.4140     0.9186     9.16     470611

  Bottom 10 items by α_idf (non-zero):
  Rank  ID      α_idf      raw_Δ      IDF      Name
  ───────────────────────────────────────────────────────────────────────────
  1     9769    0.3033     0.0291     10.41    720135
  2     610     0.3081     0.0277     11.11    760554
  3     12780   0.3323     0.0332     10.01    760551
  4     10865   0.3437     0.0309     11.11    510327
  5     45      0.3552     0.0320     11.11    530114
  6     10021   0.3667     0.0330     11.11    501101
  7     9185    0.3688     0.0369     10.01    510110
  8     13753   0.3699     0.0355     10.41    760201
  9     11113   0.3709     0.0334     11.11    500305
  10    5151    0.3728     0.0579     6.44     110103

────────────────────────────────────────────────────────────
  Check 6 — Component correlations
────────────────────────────────────────────────────────────
  Pearson correlations (non-zero items only, n=15624):
    raw_importance ↔ idf_factor : +0.2791
    raw_importance ↔ alpha_idf  : +0.9774
    idf_factor     ↔ alpha_idf  : +0.4513
  ✓ PASS: Both raw_importance and idf_factor contribute meaningfully to α_idf

────────────────────────────────────────────────────────────
  Validation complete.
────────────────────────────────────────────────────────────
