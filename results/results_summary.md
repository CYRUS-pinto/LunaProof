# LunaProof benchmark (real LOLA terrain, held-out regions)

| method                  |   0 |   30 |   60 |   90 |   120 |   150 |   180 |
|:------------------------|----:|-----:|-----:|-----:|------:|------:|------:|
| LoFTR (pretrained)      | 100 |  100 |  100 |   88 |    50 |    31 |    44 |
| PC + learned descriptor | 100 |  100 |  100 |   94 |    94 |   100 |   100 |
| PC-SIFT                 | 100 |  100 |   94 |   62 |    62 |   100 |   100 |
| SIFT                    | 100 |  100 |    0 |    0 |     0 |     0 |     0 |

| method                  |   pairs |   success_pct |   median_err_px |   median_err_ok_only |   median_inliers |   match_precision |   seconds |
|:------------------------|--------:|--------------:|----------------:|---------------------:|-----------------:|------------------:|----------:|
| LoFTR (pretrained)      |     112 |         73.21 |            0.72 |                 0.36 |            265.5 |              0.48 |      0.4  |
| PC + learned descriptor |     112 |         98.21 |            0.59 |                 0.58 |            133   |              0.35 |      0.22 |
| PC-SIFT                 |     112 |         88.39 |            0.51 |                 0.43 |             98.5 |              0.41 |      0.93 |
| SIFT                    |     112 |         28.57 |          268.05 |                 0.14 |              6   |              0.21 |      1.4  |

|                                         |   pairs |   actually_correct_pct |
|:----------------------------------------|--------:|-----------------------:|
| ('LoFTR (pretrained)', 'DEGRADED')      |      12 |                   25   |
| ('LoFTR (pretrained)', 'REFUSED')       |       2 |                    0   |
| ('LoFTR (pretrained)', 'SUCCESS')       |      98 |                   80.6 |
| ('PC + learned descriptor', 'DEGRADED') |       1 |                  100   |
| ('PC + learned descriptor', 'REFUSED')  |      12 |                   83.3 |
| ('PC + learned descriptor', 'SUCCESS')  |      99 |                  100   |
| ('PC-SIFT', 'DEGRADED')                 |      22 |                  100   |
| ('PC-SIFT', 'REFUSED')                  |      21 |                   38.1 |
| ('PC-SIFT', 'SUCCESS')                  |      69 |                  100   |
| ('SIFT', 'DEGRADED')                    |       1 |                  100   |
| ('SIFT', 'REFUSED')                     |      80 |                    0   |
| ('SIFT', 'SUCCESS')                     |      31 |                  100   |