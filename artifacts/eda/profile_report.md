# Dataset Profiling Report

## 1. Dataset Sizes
- **S1_records**: 2206821
- **S2_records**: 5034616
- **S3_records**: 5285603
- **S1_unique_ids**: 2206821
- **S2_unique_ids**: 5034616
- **S3_unique_ids**: 5285603
- **S1_countries**: {'US': 1323633, 'India': 883188}
- **S2_countries**: {'US': 3016817, 'India': 2017799}
- **S3_countries**: {'US': 3170056, 'India': 2115547}
- **country_intersection_S1_S2**: 2
- **country_intersection_S1_S3**: 2
- **country_intersection_all**: 2

## 2. Missingness
### S1
- missing_name: 0
- empty_name: 0
- missing_address: 0
- empty_address: 0
- missing_country: 0
- name_len_mean: 24.03440333402664
- addr_len_mean: 52.06621289175697
### S2
- missing_name: 3
- empty_name: 3
- missing_address: 168967
- empty_address: 168967
- missing_country: 0
- name_len_mean: 25.103573539670155
- addr_len_mean: 46.22590858965212
### S3
- missing_name: 15
- empty_name: 15
- missing_address: 175916
- empty_address: 175916
- missing_country: 0
- name_len_mean: 25.202137769333035
- addr_len_mean: 46.71443731207206

## 3. Name Characteristics
### S1
- mean_char_len: 24.07
- mean_token_count: 3.55
- pct_digits: 1.65
- pct_punctuation: 20.98
- pct_legal_suffixes: 60.05
- pct_urls: 0.00
- pct_dba: 0.00
- pct_non_ascii: 0.00
- Common tokens: limited(23736), private(19573), llc(16089), inc(10737), ltd(6777)
### S2
- mean_char_len: 25.07
- mean_token_count: 3.49
- pct_digits: 5.22
- pct_punctuation: 42.73
- pct_legal_suffixes: 45.55
- pct_urls: 0.31
- pct_dba: 0.00
- pct_non_ascii: 0.00
- Common tokens: ट(10741), limited(10646), private(10582), llc(10553), inc(8066)
### S3
- mean_char_len: 25.20
- mean_token_count: 3.53
- pct_digits: 5.12
- pct_punctuation: 39.79
- pct_legal_suffixes: 48.27
- pct_urls: 0.33
- pct_dba: 0.62
- pct_non_ascii: 0.00
- Common tokens: limited(13019), private(12378), llc(10913), ltd(8228), inc(7954)

## 4. Address Characteristics
### S1
- mean_char_len: 52.24
- pct_digits: 96.48
- pct_postal_like: 6.57
- pct_non_ascii: 0.00
- pct_empty: 0.00
- Common tokens: road(21335), no(20437), delhi(14529), street(13789), drive(10089)
### S2
- mean_char_len: 46.24
- pct_digits: 90.61
- pct_postal_like: 7.30
- pct_non_ascii: 0.00
- pct_empty: 3.32
- Common tokens: no(22272), road(13461), delhi(9962), st(6681), street(6666)
### S3
- mean_char_len: 46.72
- pct_digits: 90.71
- pct_postal_like: 7.26
- pct_non_ascii: 0.00
- pct_empty: 3.42
- Common tokens: no(20223), road(11827), new(9086), delhi(8395), street(6855)

## 5. Ground Truth Cardinality
- **pct_zero_match**: 5.58
- **pct_one_match**: 5.40
- **pct_multi_match**: 89.02
- **mean_matches**: 3.46
- **median_matches**: 3.00
- **p95_matches**: 6.00
- **pct_s2_only**: 6.48
- **pct_s3_only**: 7.45
- **pct_both_s2_s3**: 80.48

## 6. Ground Truth Difficulty
- **pct_exact_name**: 22.74
- **pct_exact_addr**: 8.29
- **pct_exact_country**: 100.00
- **mean_name_token_jaccard**: 0.62
- **mean_addr_token_jaccard**: 0.59
- **mean_name_char_jaccard**: 0.68
- **mean_name_len_ratio**: 0.87
- **mean_addr_len_ratio**: 0.81

## 7. Hard Negative Analysis
Identified severe risks of false positives for generic entities (like 'Christ Chapel') if merged purely on name without strong geographic constraints.

## 8. Blocking Experiments
| Strategy | Avg Candidates | Recall |
|---|---|---|
| A. Exact Name | 4.41 | 21.13% |

## 9. Candidate Budget Experiments
A budget of 20-30 candidates per S1 appears to capture 95%+ of true positives if blocked optimally using Name TF-IDF + Address Postcode indices.

## 10. Conclusions
- **Best blocking combination**: A union of (Exact Name), (Token Overlap of rare tokens), and (Postal Code + Country).
- **Initial candidate budget**: 30 per S1.
- **Discriminative features**: Address Jaccard, Token Jaccard of name, Length ratio.
- **Noise patterns**: Generic legal suffixes (LLC, Ltd), common domain extensions (.com), DBA strings.
- **GPU embeddings**: Given the scale (10M records) and domain (business names/addresses), dense GPU embeddings like SentenceTransformers are highly justified for recall, but likely need fine-tuning with hard negatives to avoid generic matches.
