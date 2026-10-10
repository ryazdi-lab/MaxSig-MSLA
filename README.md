MaxSig: A Learnable Maximum-Significance Layer and MaxSig-Aware Loss for Breast Ultrasound Segmentation

Reza Yazdi, Mehdi Yaghoubi, Soheila Karbasi

Department of Computer Engineering, Golestan University, Gorgan, Iran


Hassan Khotanlou

Department of Computer Engineering, Bu-Ali Sina University, Hamadan, Iran

#Abstract

Segmenting breast lesions in ultrasound is challenging due to speckle noise and ill-defined boundaries. Existing architectures lack explicit reasoning about a pixel’s local statistical significance. We employ MaxSig, a recently introduced lightweight layer applying a max-sigmoid transform over local patches, aggregating features via learnable center, median, and mean statistics. MaxSig integrates into diverse backbones through a zero-initialized gated residual, adding fewer than 1% parameters. We also propose the MaxSig-Aware Loss (MSAL) to emphasize boundaries. Evaluated across seven breast ultrasound datasets and six diverse backbones, MaxSig improves test Dice in most settings, yielding gains up to 18.5%. A heterogeneous composite benchmark confirms consistent mixed-domain improvements.
While pairing MaxSig with MSAL boosts performance on several datasets, statistical tests indicate its additional benefit depends on the architecture and dataset. Overall, MaxSig offers a robust mechanism for injecting neighborhood-significance reasoning into segmentation networks.
