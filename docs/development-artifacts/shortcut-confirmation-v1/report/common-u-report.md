# Paired common-knowledge retention

These are descriptive controls, not additional confirmatory tests. Both phases use the intersection of unchanged U query IDs and both baselines must answer correctly. same_truth_common_known additionally requires identical truth in low/high worlds; common_known retains phase-specific truths and is reported separately. U_unseen excludes the union of both replay pools; U_heldout intersects the independently selected heldout pools. None replaces the phase-own U results. Both-old-correct is a post-training subset affected by the prevalence intervention; this conditional control is not the primary causal effect of prevalence on retention.

Case-macro averages defined case ratios within each world, then equally weights worlds with defined values. Pooled first divides summed broken by shared-known counts within each world, then equally weights defined worlds. Counts below are repeated case/query instances across worlds, never independent sample sizes. Undefined zero-known cases remain missing, not zero. Learning is not repeated by edit supports.

All four editing nodes and all eight worlds, with coverage and numerator/denominator counts, are preserved in [common-u-world-checks.csv](common-u-world-checks.csv). The table below shows step 512 for every control, kind, pool, and stratum.

| Control / kind / pool / stratum | Low macro % | High macro % | Change pp | Low pooled % | High pooled % | Joint-known / eligible / pool instances | Valid cases low/high | Valid worlds low/high |
|---|---:|---:|---:|---:|---:|---|---|---|
| common_known / coherent / U_full / -1 | 0.6417 | 0.8303 | 0.1885 | 0.6413 | 0.8265 | 3757987/3930816/3930816 | 192/192 | 8/8 |
| common_known / coherent / U_full / 0 | 8.5069 | 4.3142 | -4.1927 | 8.2678 | 4.0172 | 1076/1152/1152 | 192/192 | 8/8 |
| common_known / coherent / U_full / 1 | 94.2755 | 7.2207 | -87.0548 | 94.3215 | 7.0421 | 7643/8064/8064 | 192/192 | 8/8 |
| common_known / coherent / U_full / 2 | 1.6503 | 1.8344 | 0.1841 | 1.6494 | 1.8297 | 144229/149760/149760 | 192/192 | 8/8 |
| common_known / coherent / U_full / 3 | 0.5452 | 1.3865 | 0.8413 | 0.5426 | 1.3756 | 1074726/1136064/1136064 | 192/192 | 8/8 |
| common_known / coherent / U_full / 4 | 0.3398 | 0.5186 | 0.1788 | 0.3397 | 0.5160 | 2530313/2635776/2635776 | 192/192 | 8/8 |
| common_known / coherent / U_unseen / -1 | 0.7935 | 1.0261 | 0.2325 | 0.7930 | 1.0212 | 2993383/3144384/3144384 | 192/192 | 8/8 |
| common_known / coherent / U_unseen / 0 | 11.0243 | 5.3385 | -5.6858 | 10.5485 | 4.8525 | 849/912/912 | 192/192 | 8/8 |
| common_known / coherent / U_unseen / 1 | 94.2755 | 7.2207 | -87.0548 | 94.3215 | 7.0421 | 7643/8064/8064 | 192/192 | 8/8 |
| common_known / coherent / U_unseen / 2 | 2.0891 | 2.3468 | 0.2577 | 2.0886 | 2.3409 | 110277/114804/114804 | 192/192 | 8/8 |
| common_known / coherent / U_unseen / 3 | 0.6365 | 1.6281 | 0.9916 | 0.6333 | 1.6136 | 911943/968922/968922 | 192/192 | 8/8 |
| common_known / coherent / U_unseen / 4 | 0.4262 | 0.6501 | 0.2239 | 0.4261 | 0.6469 | 1962671/2051682/2051682 | 192/192 | 8/8 |
| common_known / coherent / U_heldout / -1 | 0.7042 | 1.1938 | 0.4895 | 0.7043 | 1.1864 | 1304795/1401024/1401024 | 192/192 | 8/8 |
| common_known / coherent / U_heldout / 0 | 13.0208 | 5.9896 | -7.0312 | 12.5113 | 5.5901 | 359/384/384 | 192/192 | 8/8 |
| common_known / coherent / U_heldout / 1 | 94.1189 | 7.1199 | -86.9990 | 94.2388 | 6.9405 | 1626/1728/1728 | 192/192 | 8/8 |
| common_known / coherent / U_heldout / 2 | 1.9821 | 2.4602 | 0.4782 | 1.9826 | 2.4535 | 42218/44736/44736 | 192/192 | 8/8 |
| common_known / coherent / U_heldout / 3 | 0.5216 | 1.3824 | 0.8608 | 0.5195 | 1.3615 | 484614/527040/527040 | 192/192 | 8/8 |
| common_known / coherent / U_heldout / 4 | 0.5481 | 0.9981 | 0.4501 | 0.5490 | 0.9943 | 775978/827136/827136 | 192/192 | 8/8 |
| common_known / exception / U_full / -1 | 0.6170 | 0.8477 | 0.2307 | 0.6165 | 0.8433 | 3757987/3930816/3930816 | 192/192 | 8/8 |
| common_known / exception / U_full / 0 | 6.2587 | 4.2622 | -1.9965 | 6.1188 | 4.1888 | 1076/1152/1152 | 192/192 | 8/8 |
| common_known / exception / U_full / 1 | 87.3777 | 5.9595 | -81.4182 | 87.4138 | 5.8361 | 7643/8064/8064 | 192/192 | 8/8 |
| common_known / exception / U_full / 2 | 1.5934 | 1.7556 | 0.1622 | 1.5924 | 1.7497 | 144229/149760/149760 | 192/192 | 8/8 |
| common_known / exception / U_full / 3 | 0.5494 | 1.3965 | 0.8471 | 0.5466 | 1.3849 | 1074726/1136064/1136064 | 192/192 | 8/8 |
| common_known / exception / U_full / 4 | 0.3265 | 0.5482 | 0.2217 | 0.3261 | 0.5451 | 2530313/2635776/2635776 | 192/192 | 8/8 |
| common_known / exception / U_unseen / -1 | 0.7667 | 1.0515 | 0.2849 | 0.7661 | 1.0458 | 2993383/3144384/3144384 | 192/192 | 8/8 |
| common_known / exception / U_unseen / 0 | 8.0556 | 5.3646 | -2.6910 | 7.8445 | 5.0582 | 849/912/912 | 192/192 | 8/8 |
| common_known / exception / U_unseen / 1 | 87.3777 | 5.9595 | -81.4182 | 87.4138 | 5.8361 | 7643/8064/8064 | 192/192 | 8/8 |
| common_known / exception / U_unseen / 2 | 2.0520 | 2.2705 | 0.2185 | 2.0523 | 2.2626 | 110277/114804/114804 | 192/192 | 8/8 |
| common_known / exception / U_unseen / 3 | 0.6429 | 1.6422 | 0.9993 | 0.6396 | 1.6266 | 911943/968922/968922 | 192/192 | 8/8 |
| common_known / exception / U_unseen / 4 | 0.4127 | 0.6913 | 0.2786 | 0.4122 | 0.6872 | 1962671/2051682/2051682 | 192/192 | 8/8 |
| common_known / exception / U_heldout / -1 | 0.6606 | 1.2393 | 0.5787 | 0.6606 | 1.2314 | 1304795/1401024/1401024 | 192/192 | 8/8 |
| common_known / exception / U_heldout / 0 | 10.1562 | 6.5104 | -3.6458 | 10.0691 | 6.1397 | 359/384/384 | 192/192 | 8/8 |
| common_known / exception / U_heldout / 1 | 86.0212 | 5.7868 | -80.2344 | 86.1770 | 5.6543 | 1626/1728/1728 | 192/192 | 8/8 |
| common_known / exception / U_heldout / 2 | 1.9005 | 2.3820 | 0.4815 | 1.9017 | 2.3732 | 42218/44736/44736 | 192/192 | 8/8 |
| common_known / exception / U_heldout / 3 | 0.5160 | 1.4397 | 0.9237 | 0.5141 | 1.4192 | 484614/527040/527040 | 192/192 | 8/8 |
| common_known / exception / U_heldout / 4 | 0.5009 | 1.0453 | 0.5444 | 0.5013 | 1.0407 | 775978/827136/827136 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_full / -1 | 0.3840 | 0.6382 | 0.2542 | 0.3835 | 0.6350 | 3431455/3586752/3930816 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_full / 0 | 8.5069 | 4.3142 | -4.1927 | 8.2678 | 4.0172 | 1076/1152/1152 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_full / 1 | Undefined | Undefined | Undefined | Undefined | Undefined | 0/0/8064 | 0/0 | 0/0 |
| same_truth_common_known / coherent / U_full / 2 | 1.6621 | 1.8174 | 0.1553 | 1.6610 | 1.8131 | 136500/141624/149760 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_full / 3 | 0.4369 | 0.9110 | 0.4740 | 0.4348 | 0.9022 | 919103/972096/1136064 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_full / 4 | 0.2870 | 0.4646 | 0.1776 | 0.2869 | 0.4625 | 2374776/2471880/2635776 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_unseen / -1 | 0.4805 | 0.8029 | 0.3224 | 0.4800 | 0.7988 | 2666851/2800320/3144384 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_unseen / 0 | 11.0243 | 5.3385 | -5.6858 | 10.5485 | 4.8525 | 849/912/912 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_unseen / 1 | Undefined | Undefined | Undefined | Undefined | Undefined | 0/0/8064 | 0/0 | 0/0 |
| same_truth_common_known / coherent / U_unseen / 2 | 2.1382 | 2.3622 | 0.2240 | 2.1372 | 2.3573 | 102548/106668/114804 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_unseen / 3 | 0.5236 | 1.0997 | 0.5761 | 0.5211 | 1.0874 | 756320/804954/968922 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_unseen / 4 | 0.3642 | 0.5903 | 0.2261 | 0.3641 | 0.5878 | 1807134/1887786/2051682 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_heldout / -1 | 0.5565 | 1.0993 | 0.5429 | 0.5567 | 1.0926 | 1239162/1331808/1401024 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_heldout / 0 | 13.0208 | 5.9896 | -7.0312 | 12.5113 | 5.5901 | 359/384/384 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_heldout / 1 | Undefined | Undefined | Undefined | Undefined | Undefined | 0/0/1728 | 0/0 | 0/0 |
| same_truth_common_known / coherent / U_heldout / 2 | 2.0113 | 2.4379 | 0.4266 | 2.0123 | 2.4309 | 40755/43188/44736 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_heldout / 3 | 0.4709 | 1.1724 | 0.7014 | 0.4691 | 1.1524 | 452903/493584/527040 | 192/192 | 8/8 |
| same_truth_common_known / coherent / U_heldout / 4 | 0.5238 | 0.9843 | 0.4605 | 0.5248 | 0.9812 | 745145/794652/827136 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_full / -1 | 0.3720 | 0.6605 | 0.2885 | 0.3715 | 0.6569 | 3431455/3586752/3930816 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_full / 0 | 6.2587 | 4.2622 | -1.9965 | 6.1188 | 4.1888 | 1076/1152/1152 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_full / 1 | Undefined | Undefined | Undefined | Undefined | Undefined | 0/0/8064 | 0/0 | 0/0 |
| same_truth_common_known / exception / U_full / 2 | 1.6022 | 1.7262 | 0.1241 | 1.6007 | 1.7213 | 136500/141624/149760 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_full / 3 | 0.4347 | 0.9357 | 0.5010 | 0.4326 | 0.9270 | 919103/972096/1136064 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_full / 4 | 0.2751 | 0.4922 | 0.2171 | 0.2747 | 0.4897 | 2374776/2471880/2635776 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_unseen / -1 | 0.4697 | 0.8355 | 0.3658 | 0.4692 | 0.8308 | 2666851/2800320/3144384 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_unseen / 0 | 8.0556 | 5.3646 | -2.6910 | 7.8445 | 5.0582 | 849/912/912 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_unseen / 1 | Undefined | Undefined | Undefined | Undefined | Undefined | 0/0/8064 | 0/0 | 0/0 |
| same_truth_common_known / exception / U_unseen / 2 | 2.0985 | 2.2694 | 0.1709 | 2.0981 | 2.2635 | 102548/106668/114804 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_unseen / 3 | 0.5226 | 1.1325 | 0.6098 | 0.5202 | 1.1199 | 756320/804954/968922 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_unseen / 4 | 0.3525 | 0.6298 | 0.2773 | 0.3521 | 0.6266 | 1807134/1887786/2051682 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_heldout / -1 | 0.5198 | 1.1490 | 0.6292 | 0.5201 | 1.1419 | 1239162/1331808/1401024 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_heldout / 0 | 10.1562 | 6.5104 | -3.6458 | 10.0691 | 6.1397 | 359/384/384 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_heldout / 1 | Undefined | Undefined | Undefined | Undefined | Undefined | 0/0/1728 | 0/0 | 0/0 |
| same_truth_common_known / exception / U_heldout / 2 | 1.9304 | 2.3708 | 0.4405 | 1.9313 | 2.3624 | 40755/43188/44736 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_heldout / 3 | 0.4599 | 1.2371 | 0.7772 | 0.4585 | 1.2178 | 452903/493584/527040 | 192/192 | 8/8 |
| same_truth_common_known / exception / U_heldout / 4 | 0.4753 | 1.0306 | 0.5553 | 0.4759 | 1.0268 | 745145/794652/827136 | 192/192 | 8/8 |
