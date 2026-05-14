我正在使用 ViT + MLP + PixelShuffle 来估计深度图。这是我们论文的一项关键技术贡献，因为我们希望证明 DPT 计算量过大，而使用 ViT + MLP + PixelShuffle 可以达到与 DPT 相似的结果（定量和定性方面）。然而，目前我们的定性结果并不理想。正如.@projects/vggt/models/heads/linear_variants/imageData_2.png @projects/vggt/models/heads/linear_variants/error1.png  所示，我观察到了明显的伪影。您认为这是否可能与仅使用 ViT 的最后一层 token 而未使用多尺度特征有关？



我该如何解决这个问题？例如，取第 4、7、11 和 23 层的输出，将它们连接起来，然后使用 MLP + PixelShuffle 来获得结果？或者你有更好的想法的话请与我讨论。如果可能的话，我不想使用conv layers



