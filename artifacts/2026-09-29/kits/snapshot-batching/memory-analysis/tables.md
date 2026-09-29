## Q1. Per block, per rank (settled updates; MiB)

Values: max over the block's settled updates of per-update peak allocated / peak reserved; mean end-of-update allocated / reserved; max inactive-split peak; mean cache (post-learn) and collection-cache (post-collect) bytes; mean private-graph pool bytes (all identities); heads.

| block | arm | updates | rank | peak alloc max | peak res max | end alloc mean | end res mean | end res max | inact split peak max | cache mean | coll-cache mean | graph ids | graph pool mean | heads |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | off | 872-879 | 0 | 3,503 | 4,640 | 1,730 | 4,464 | 4,640 | 0 | 1,344 | 2,315 | 13-15 | 422 | 0.0 |
| 0 | off | 872-879 | 1 | 3,511 | 4,506 | 1,690 | 4,401 | 4,506 | 0 | 1,301 | 2,253 | 13-15 | 417 | 0.0 |
| 0 | off | 872-879 | 2 | 4,199 | 5,062 | 1,681 | 4,592 | 4,906 | 0 | 1,296 | 2,279 | 13-15 | 424 | 0.0 |
| 0 | off | 872-879 | 3 | 3,891 | 4,770 | 1,568 | 4,318 | 4,420 | 0 | 1,180 | 2,110 | 13-15 | 423 | 0.0 |
| 1 | on | 882-889 | 0 | 4,202 | 5,144 | 1,699 | 5,129 | 5,144 | 0 | 1,329 | 2,296 | 1-1 | 42 | 32.8 |
| 1 | on | 882-889 | 1 | 3,448 | 4,600 | 1,647 | 4,598 | 4,600 | 0 | 1,277 | 2,249 | 1-1 | 42 | 32.8 |
| 1 | on | 882-889 | 2 | 3,375 | 4,974 | 1,629 | 4,936 | 4,974 | 0 | 1,267 | 2,204 | 1-1 | 22 | 32.8 |
| 1 | on | 882-889 | 3 | 3,384 | 5,140 | 1,659 | 5,038 | 5,140 | 0 | 1,286 | 2,207 | 1-1 | 42 | 32.8 |
| 2 | off | 892-899 | 0 | 4,330 | 5,364 | 1,759 | 4,750 | 4,904 | 0 | 1,260 | 2,221 | 13-15 | 420 | 0.0 |
| 2 | off | 892-899 | 1 | 4,115 | 5,372 | 1,729 | 4,683 | 4,736 | 0 | 1,236 | 2,192 | 12-14 | 423 | 0.0 |
| 2 | off | 892-899 | 2 | 3,568 | 4,884 | 1,797 | 4,683 | 4,884 | 0 | 1,306 | 2,270 | 12-14 | 424 | 0.0 |
| 2 | off | 892-899 | 3 | 4,264 | 5,528 | 1,821 | 4,915 | 5,022 | 0 | 1,327 | 2,313 | 14-16 | 432 | 0.0 |
| 3 | on | 902-909 | 0 | 4,449 | 5,444 | 1,734 | 5,372 | 5,444 | 0 | 1,267 | 2,243 | 1-1 | 42 | 32.8 |
| 3 | on | 902-909 | 1 | 3,535 | 5,308 | 1,740 | 5,306 | 5,308 | 0 | 1,277 | 2,229 | 1-1 | 42 | 32.8 |
| 3 | on | 902-909 | 2 | 4,221 | 5,372 | 1,786 | 4,864 | 5,372 | 0 | 1,330 | 2,274 | 1-1 | 22 | 32.8 |
| 3 | on | 902-909 | 3 | 4,237 | 5,442 | 1,720 | 5,442 | 5,442 | 0 | 1,253 | 2,202 | 1-1 | 42 | 32.8 |
| 4 | off* | 912-914 | 0 | 4,098 | 5,334 | 1,723 | 4,671 | 4,728 | 0 | 1,223 | 2,123 | 13-14 | 411 | 0.0 |
| 4 | off* | 912-914 | 1 | 4,342 | 5,340 | 1,779 | 4,812 | 4,912 | 0 | 1,278 | 2,279 | 14-16 | 422 | 0.0 |
| 4 | off* | 912-914 | 2 | 3,579 | 4,796 | 1,793 | 4,655 | 4,688 | 0 | 1,304 | 2,237 | 13-14 | 429 | 0.0 |
| 4 | off* | 912-914 | 3 | 4,450 | 5,484 | 1,918 | 4,956 | 5,058 | 0 | 1,424 | 2,478 | 13-14 | 416 | 0.0 |
| 5 | on* | 917-919 | 0 | 3,662 | 4,936 | 1,871 | 4,919 | 4,936 | 0 | 1,403 | 2,420 | 1-1 | 42 | 32.8 |
| 5 | on* | 917-919 | 1 | 4,152 | 5,250 | 1,673 | 5,071 | 5,250 | 0 | 1,208 | 2,155 | 1-1 | 42 | 32.8 |
| 5 | on* | 917-919 | 2 | 3,339 | 4,594 | 1,650 | 4,594 | 4,594 | 0 | 1,195 | 2,106 | 1-1 | 22 | 32.8 |
| 5 | on* | 917-919 | 3 | 4,289 | 5,438 | 1,720 | 5,438 | 5,438 | 0 | 1,258 | 2,183 | 1-1 | 42 | 32.8 |

`*` = profiled block. `peak GB` in ab-summary = max over all 4 ranks and settled updates of `cuda_peak_reserved_bytes` / 1e9 (decimal GB, reserved, reset every update).

## Q1-Q2. Sums across ranks vs nvidia-smi (settled updates; MiB)

| block | arm | updates | sum peak res (max) | sum end res mean | sum end alloc mean | sum end res-alloc mean | sum peak res - end res mean | ranks trimmed/upd | captures/upd (4 ranks) | smi max | smi mean | smi min | smi max - sum peak res (max over upd) | smi_end - sum end res (mean) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | off | 872-879 | 18,420 | 17,775 | 6,670 | 11,105 | 374 | 2.38 | 646.38 | 20,458 | 17,764 | 15,490 | 2,058 | -231 |
| 1 | on | 882-889 | 19,858 | 19,700 | 6,633 | 13,066 | 0 | 0.00 | 0.00 | 21,870 | 21,698 | 21,406 | 2,012 | 1,150 |
| 2 | off | 892-899 | 20,700 | 19,031 | 7,106 | 11,925 | 836 | 3.00 | 590.62 | 21,250 | 19,145 | 16,810 | 1,654 | 429 |
| 3 | on | 902-909 | 21,566 | 20,983 | 6,981 | 14,002 | 0 | 0.00 | 0.00 | 23,578 | 22,970 | 22,602 | 2,012 | 1,339 |
| 4 | off* | 912-914 | 20,266 | 19,094 | 7,213 | 11,881 | 925 | 3.67 | 594.33 | 21,238 | 19,077 | 17,018 | 1,200 | 1,190 |
| 5 | on* | 917-919 | 20,218 | 20,022 | 6,914 | 13,108 | 0 | 0.00 | 0.00 | 22,230 | 22,010 | 21,586 | 2,012 | 2,012 |

### Pooled by arm (settled updates)

| arm | set | n upd | sum end res mean | sum end alloc mean | sum end res-alloc mean | ranks trimmed/upd | captures/upd | smi mean (upd means) | smi_end - sum end res mean | sd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | unprofiled | 16 | 18,403 | 6,888 | 11,515 | 2.69 | 618.50 | 18,454 | 99 | 1,630 |
| on | unprofiled | 16 | 20,341 | 6,807 | 13,534 | 0.00 | 0.00 | 22,334 | 1,245 | 2,161 |
| off | all | 19 | 18,512 | 6,939 | 11,573 | 2.84 | 614.68 | 18,553 | 271 | 1,647 |
| on | all | 19 | 20,291 | 6,824 | 13,467 | 0.00 | 0.00 | 22,283 | 1,366 | 1,994 |

First update 845: sum end reserved 4,702 MiB, nearest smi sample 7,468 MiB (dt +1.3 s) -> outside-allocator estimate 2,766 MiB.

### Sum over ranks of per-update peak allocated (settled; MiB; ranks peak at different times, so an upper bound on simultaneous live bytes)

| arm | set | n upd | mean | max |
|---|---|---:|---:|---:|
| off | unprofiled | 16 | 14,615 | 15,923 |
| on | unprofiled | 16 | 14,192 | 15,274 |
| off | all | 19 | 14,724 | 15,923 |
| on | all | 19 | 14,239 | 15,274 |

### Per-sample smi(t) minus the sum of the 4 ranks' peak reserved of the update containing t (MiB; post-warmup)

This is a lower bound on memory outside the PyTorch allocator at t (exact when every rank's reserved equals its peak at t).

| arm | samples | max | p90 | median | min | samples within 20 MiB of max |
|---|---:|---:|---:|---:|---:|---:|
| off | 105 | 2,058 | 1,522 | -952 | -4,968 | 1 |
| on | 76 | 2,012 | 2,012 | 2,012 | 1,454 | 58 |

## Reserved trimming vs private-graph captures (per rank-update, updates after warmup)

| arm | >=1 capture in update | rank-updates | with end reserved < peak reserved | mean drop MiB (when dropped) |
|---|---|---:|---:|---:|
| off | True | 100 | 72 | 246 |
| on | False | 100 | 0 | - |

## Q3. Time structure of nvidia-smi samples

Off-arm maximum over all post-warmup off updates (settling included): 21,250 MiB.

Samples above it: 70 of 181 post-warmup samples.

| epoch-offset s | update | arm | block pos | phase | smi MiB | excess over off max |
|---:|---:|---|---:|---|---:|---:|
| 636 | 882 | on | 2 | collect | 21,406 | 156 |
| 641 | 882 | on | 2 | learn | 21,438 | 188 |
| 646 | 882 | on | 2 | learn | 21,458 | 208 |
| 651 | 883 | on | 3 | collect | 21,620 | 370 |
| 656 | 883 | on | 3 | learn | 21,620 | 370 |
| 661 | 883 | on | 3 | learn | 21,640 | 390 |
| 666 | 884 | on | 4 | collect | 21,640 | 390 |
| 671 | 884 | on | 4 | learn | 21,642 | 392 |
| 676 | 884 | on | 4 | learn | 21,642 | 392 |
| 681 | 885 | on | 5 | collect | 21,692 | 442 |
| 686 | 885 | on | 5 | learn | 21,708 | 458 |
| 691 | 886 | on | 6 | collect | 21,708 | 458 |
| 696 | 886 | on | 6 | collect | 21,726 | 476 |
| 701 | 886 | on | 6 | learn | 21,746 | 496 |
| 706 | 887 | on | 7 | collect | 21,746 | 496 |
| 711 | 887 | on | 7 | collect | 21,802 | 552 |
| 716 | 887 | on | 7 | learn | 21,802 | 552 |
| 721 | 888 | on | 8 | collect | 21,802 | 552 |
| 726 | 888 | on | 8 | collect | 21,826 | 576 |
| 731 | 888 | on | 8 | learn | 21,826 | 576 |
| 736 | 889 | on | 9 | collect | 21,826 | 576 |
| 741 | 889 | on | 9 | collect | 21,870 | 620 |
| 746 | 889 | on | 9 | learn | 21,870 | 620 |
| 961 | 900 | on | 0 | collect | 21,778 | 528 |
| 966 | 900 | on | 0 | learn | 21,920 | 670 |
| 971 | 900 | on | 0 | learn | 21,920 | 670 |
| 976 | 901 | on | 1 | collect | 21,944 | 694 |
| 981 | 901 | on | 1 | collect | 22,502 | 1,252 |
| 986 | 901 | on | 1 | learn | 22,502 | 1,252 |
| 991 | 902 | on | 2 | collect | 22,602 | 1,352 |
| 996 | 902 | on | 2 | collect | 22,772 | 1,522 |
| 1001 | 902 | on | 2 | learn | 22,780 | 1,530 |
| 1006 | 903 | on | 3 | collect | 22,786 | 1,536 |
| 1011 | 903 | on | 3 | collect | 22,798 | 1,548 |
| 1016 | 903 | on | 3 | learn | 22,798 | 1,548 |
| 1021 | 904 | on | 4 | collect | 22,818 | 1,568 |
| 1026 | 904 | on | 4 | collect | 22,818 | 1,568 |
| 1031 | 904 | on | 4 | learn | 22,818 | 1,568 |
| 1036 | 904 | on | 4 | learn | 22,818 | 1,568 |
| 1041 | 905 | on | 5 | collect | 22,818 | 1,568 |
| 1046 | 905 | on | 5 | learn | 22,824 | 1,574 |
| 1051 | 905 | on | 5 | learn | 22,824 | 1,574 |
| 1056 | 906 | on | 6 | collect | 22,940 | 1,690 |
| 1061 | 906 | on | 6 | collect | 22,954 | 1,704 |
| 1066 | 906 | on | 6 | learn | 22,954 | 1,704 |
| 1071 | 907 | on | 7 | collect | 22,954 | 1,704 |
| 1076 | 907 | on | 7 | collect | 22,954 | 1,704 |
| 1081 | 907 | on | 7 | learn | 22,954 | 1,704 |
| 1086 | 908 | on | 8 | collect | 22,954 | 1,704 |
| 1091 | 908 | on | 8 | collect | 23,254 | 2,004 |
| 1096 | 908 | on | 8 | learn | 23,254 | 2,004 |
| 1101 | 909 | on | 9 | collect | 23,494 | 2,244 |
| 1106 | 909 | on | 9 | collect | 23,578 | 2,328 |
| 1111 | 909 | on | 9 | learn | 23,578 | 2,328 |
| 1221 | 915 | on | 0 | collect | 21,276 | 26 |
| 1226 | 915 | on | 0 | collect | 21,416 | 166 |
| 1231 | 915 | on | 0 | learn | 21,474 | 224 |
| 1236 | 916 | on | 1 | collect | 21,474 | 224 |
| 1241 | 916 | on | 1 | collect | 21,536 | 286 |
| 1246 | 916 | on | 1 | learn | 21,540 | 290 |
| 1251 | 916 | on | 1 | learn | 21,540 | 290 |
| 1256 | 917 | on | 2 | collect | 21,586 | 336 |
| 1261 | 917 | on | 2 | collect | 21,690 | 440 |
| 1266 | 917 | on | 2 | learn | 21,690 | 440 |
| 1271 | 918 | on | 3 | collect | 22,170 | 920 |
| 1276 | 918 | on | 3 | collect | 22,182 | 932 |
| 1281 | 918 | on | 3 | learn | 22,182 | 932 |
| 1286 | 919 | on | 4 | collect | 22,182 | 932 |
| 1291 | 919 | on | 4 | collect | 22,182 | 932 |
| 1296 | 919 | on | 4 | learn | 22,230 | 980 |

### Per-block sample distribution (all updates of the block, settling included)

| block | arm | updates | samples | collect n / mean / max | learn n / mean / max | p50 | p90 | max | n at max |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| 0 | off | 870-879 | 42 | 29 / 17,009 / 18,526 | 13 / 19,498 / 20,458 | 17,417 | 19,964 | 20,458 | 1 |
| 1 | on | 880-889 | 29 | 14 / 21,583 / 21,870 | 15 / 21,487 / 21,870 | 21,642 | 21,826 | 21,870 | 2 |
| 2 | off | 890-899 | 42 | 29 / 18,154 / 20,898 | 13 / 20,708 / 21,250 | 18,711 | 21,022 | 21,250 | 1 |
| 3 | on | 900-909 | 31 | 18 / 22,818 / 23,578 | 13 / 22,765 / 23,578 | 22,818 | 23,254 | 23,578 | 2 |
| 4 | off* | 910-914 | 21 | 15 / 18,061 / 20,684 | 6 / 20,585 / 21,238 | 18,280 | 21,100 | 21,238 | 1 |
| 5 | on* | 915-919 | 16 | 10 / 21,769 / 22,182 | 6 / 21,776 / 22,230 | 21,638 | 22,182 | 22,230 | 1 |

### Arm switches: nvidia-smi around each switch (MiB)

| switch at update | from -> to | last 3 samples before | first 3 samples after | sum end res before -> after 1st upd |
|---:|---|---|---|---|
| 880 | off -> on | 18,022, 20,038, 20,458 | 20,528, 20,952, 20,952 | 18,026 -> 18,940 |
| 890 | on -> off | 21,826, 21,870, 21,870 | 14,978, 17,586, 17,474 | 19,858 -> 18,444 |
| 900 | off -> on | 18,156, 20,728, 21,048 | 21,778, 21,920, 21,920 | 19,036 -> 19,908 |
| 910 | on -> off | 23,494, 23,578, 23,578 | 17,926, 17,144, 17,438 | 21,566 -> 19,088 |
| 915 | off -> on | 17,018, 17,732, 20,936 | 21,276, 21,416, 21,474 | 18,924 -> 19,462 |

## Q4. Confounders and regression (per update, settled, after warmup)

| block | arm | resident (4 ranks) mean | mean prefix (rank mean) | max prefix max | cache MiB sum mean | coll-cache MiB sum mean | store tokens sum mean | graph pool MiB sum mean | heads MiB sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | off | 54.4 | 585 | 2729 | 5,120 | 8,957 | 601,955 | 1,686 | 0.0 |
| 1 | on | 51.4 | 582 | 2142 | 5,159 | 8,955 | 594,498 | 148 | 131.3 |
| 2 | off | 54.0 | 595 | 2402 | 5,129 | 8,996 | 607,915 | 1,698 | 0.0 |
| 3 | on | 53.9 | 584 | 2447 | 5,128 | 8,948 | 601,990 | 148 | 131.3 |
| 4 | off* | 54.3 | 600 | 2216 | 5,228 | 9,116 | 614,344 | 1,678 | 0.0 |
| 5 | on* | 53.3 | 587 | 2445 | 5,064 | 8,864 | 597,360 | 148 | 131.3 |

OLS on settled post-warmup updates (n = 38). Coefficients per unit; t = coef / se (ordinary SE, updates are autocorrelated, so t is optimistic).

| y | model | arm coef (t) | other coefs (t) | R2 |
|---|---|---|---|---:|
| sum_res | arm only | 1,779 (8.3) | - | 0.66 |
| sum_res | arm + collection cache | 1,814 (8.1) | sum_ccache 0.569 (0.7) | 0.66 |
| sum_res | arm + prefix + resident + update | 1,460 (7.5) | sum_mean_prefix 0.186 (0.0); sum_resident 11.1 (0.2); update 36.5 (5.9) | 0.85 |
| sum_res | no arm: ccache + prefix + resident + update | - | sum_ccache 1.23 (1.0); sum_mean_prefix -35.2 (-2.3); sum_resident -144 (-2.2); update 60.4 (6.9) | 0.61 |
| sum_alloc | arm only | -115 (-1.6) | - | 0.07 |
| sum_alloc | arm + collection cache | -74 (-1.1) | sum_ccache 0.681 (2.6) | 0.22 |
| sum_alloc | arm + prefix + resident + update | -176 (-4.0) | sum_mean_prefix 5.46 (3.1); sum_resident 6.98 (0.7); update 12.4 (9.0) | 0.82 |
| sum_alloc | no arm: ccache + prefix + resident + update | - | sum_ccache 0.496 (2.6); sum_mean_prefix 4.43 (1.9); sum_resident 24.9 (2.5); update 10.3 (7.6) | 0.77 |
| sum_res_minus_alloc_end | arm only | 1,894 (11.9) | - | 0.80 |
| sum_res_minus_alloc_end | arm + collection cache | 1,887 (11.4) | sum_ccache -0.112 (-0.2) | 0.80 |
| sum_res_minus_alloc_end | arm + prefix + resident + update | 1,636 (9.7) | sum_mean_prefix -5.27 (-0.8); sum_resident 4.13 (0.1); update 24.1 (4.6) | 0.88 |
| sum_res_minus_alloc_end | no arm: ccache + prefix + resident + update | - | sum_ccache 0.73 (0.6); sum_mean_prefix -39.7 (-2.5); sum_resident -169 (-2.5); update 50.2 (5.5) | 0.55 |
| smi_mean | arm only | 3,730 (15.0) | - | 0.86 |
| smi_mean | arm + collection cache | 3,786 (14.8) | sum_ccache 0.906 (0.9) | 0.87 |
| smi_mean | arm + prefix + resident + update | 3,387 (13.2) | sum_mean_prefix -2.88 (-0.3); sum_resident 17.4 (0.3); update 37.6 (4.6) | 0.92 |
| smi_mean | no arm: ccache + prefix + resident + update | - | sum_ccache 2.91 (1.2); sum_mean_prefix -85.6 (-2.8); sum_resident -341 (-2.6); update 93.1 (5.3) | 0.54 |
| smi_max | arm only | 1,767 (8.3) | - | 0.66 |
| smi_max | arm + collection cache | 1,799 (8.1) | sum_ccache 0.526 (0.6) | 0.66 |
| smi_max | arm + prefix + resident + update | 1,447 (7.3) | sum_mean_prefix -0.25 (-0.0); sum_resident 9.13 (0.2); update 36 (5.8) | 0.85 |
| smi_max | no arm: ccache + prefix + resident + update | - | sum_ccache 1.19 (1.0); sum_mean_prefix -35.1 (-2.3); sum_resident -144 (-2.2); update 59.6 (6.8) | 0.61 |

### Adjacent-block differences (on minus neighbouring off, settled block means)

| pair | d smi mean | d smi max | d sum end res | d sum end alloc | d sum end res-alloc | d coll-cache | d mean prefix | d resident |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| on 882-889 - off 872-879 | 3,934 | 1,412 | 1,924 | -37 | 1,961 | -2 | -2.7 | -3.0 |
| on 882-889 - off 892-899 | 2,553 | 620 | 669 | -472 | 1,141 | -41 | -12.7 | -2.6 |
| on 902-909 - off 892-899 | 3,825 | 2,328 | 1,952 | -125 | 2,077 | -49 | -11.0 | -0.1 |
| on 902-909 - off 912-914 | 3,893 | 2,340 | 1,889 | -232 | 2,121 | -168 | -16.3 | -0.5 |
| on 917-919 - off 912-914 | 2,934 | 992 | 928 | -299 | 1,227 | -252 | -13.5 | -1.0 |
